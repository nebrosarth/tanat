"""Collect a side-balanced, batched AI-42 intervention-DAgger generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .build_ai42_dataset import MatchSpec, deterministic_seed_schedule
from .build_ai42_dataset_go import complete_go_staged_match, merge_go_staging
from .collect_dagger_ai42 import (
    MarginInterventionGate,
    NativeDaggerWriter,
    TEACHER_LABEL_STATUSES,
    _atomic_write,
    _candidate_indices,
    _controllers,
)
from .env import AI40_ROSTER, AI42_DAGGER_PROTOCOL_VERSION, HERO_COUNT, AssaultEnvProcess
from .evaluate_ai42 import load_actor
from .export_ai42 import export_ai42_actor
from .inference_ai42_onnx import AI42ONNXEvaluationActor, create_onnx_session
from .trajectory_ai42 import canonical_json_bytes


@dataclass(slots=True)
class _MatchRuntime:
    spec: MatchSpec
    candidate_side: int
    env: AssaultEnvProcess
    writer: NativeDaggerWriter
    observation: Any
    gate: MarginInterventionGate
    ticks: int = 0
    interventions: int = 0
    candidate_labels: int = 0
    opponent_labels: int = 0
    invalid_actions: int = 0
    candidate_reward: float = 0.0
    writer_stdout: str = ""
    writer_stderr: str = ""
    done: bool = False


def _roster(seed: int, index: int) -> tuple[int, ...]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0xDA66E2]))
    return tuple(int(value) for value in rng.permutation(AI40_ROSTER))


def build_dagger_schedule(
    *, seed: int, matches: int, max_steps: int, lineage: dict[str, Any],
    threshold: float, min_gap_ticks: int, split_seed: int,
    validation_fraction: float,
) -> tuple[dict[str, Any], tuple[MatchSpec, ...]]:
    if isinstance(matches, bool) or not isinstance(matches, int) or matches < 1:
        raise ValueError("matches must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between zero and one")
    MarginInterventionGate(HERO_COUNT // 2, threshold, min_gap_ticks)
    seeds = deterministic_seed_schedule(seed, matches)
    specs = tuple(
        MatchSpec(
            index=index,
            seed=match_seed,
            match_id=f"ai42-dagger-{seed}-{index:06d}",
            scenario="ai42_dagger_vs_ai30",
            controller_by_slot=_controllers(index % 2 + 1),
            roster_ids=_roster(match_seed, index),
        )
        for index, match_seed in enumerate(seeds)
    )
    schedule = {
        "backend": "onnxruntime-dagger-batched-v1",
        "command": "tanat-ai42-collect-dagger-generation",
        "contract_version": "AI42-dagger-collector-v1",
        "intervention_policy": {
            "metric": "minimum-masked-top2-logit-margin",
            "threshold": threshold,
            "min_gap_ticks": min_gap_ticks,
        },
        "match_schedule": [
            {
                "controller_by_slot": list(spec.controller_by_slot),
                "index": spec.index,
                "match_id": spec.match_id,
                "roster_ids": list(spec.roster_ids),
                "scenario": spec.scenario,
                "seed": spec.seed,
                "side_by_slot": list(spec.side_by_slot),
            }
            for spec in specs
        ],
        "max_steps": max_steps,
        "policy_lineage": lineage,
        "seed": seed,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
    }
    return schedule, specs


def _start_runtime(
    spec: MatchSpec,
    *,
    env_executable: Path,
    writer_executable: Path,
    schedule_path: Path,
    staging: Path,
    max_steps: int,
    threshold: float,
    min_gap_ticks: int,
) -> _MatchRuntime:
    side = spec.index % 2 + 1
    env = AssaultEnvProcess(env_executable, AI42_DAGGER_PROTOCOL_VERSION)
    match_root = staging / "matches" / f"match-{spec.index:06d}"
    writer: NativeDaggerWriter | None = None
    try:
        writer = NativeDaggerWriter(
            writer_executable, schedule_path, match_root, spec.index,
            min(max_steps, 2048),
        )
        observation = env.reset(
            seed=spec.seed, max_steps=max_steps, roster=spec.roster_ids,
            controllers=spec.controller_by_slot,
        )
    except BaseException:
        if writer is not None:
            writer.abort()
            writer.close()
        env.close()
        raise
    assert writer is not None
    return _MatchRuntime(
        spec=spec, candidate_side=side, env=env, writer=writer,
        observation=observation,
        gate=MarginInterventionGate(HERO_COUNT // 2, threshold, min_gap_ticks),
    )


def _finish_runtime(
    runtime: _MatchRuntime,
    staging: Path,
    schedule: dict[str, Any],
) -> None:
    stdout, stderr = runtime.writer.finish()
    runtime.writer_stdout = stdout.decode(errors="replace").strip()
    runtime.writer_stderr = stderr.decode(errors="replace").strip()
    match_root = staging / "matches" / f"match-{runtime.spec.index:06d}"
    complete_go_staged_match(match_root, runtime.spec, schedule)
    runtime.writer.close()
    runtime.env.close()
    runtime.done = True


def _abort_runtimes(runtimes: list[_MatchRuntime]) -> str:
    diagnostics: list[str] = []
    for runtime in runtimes:
        if not runtime.done:
            try:
                _stdout, stderr = runtime.writer.abort()
                if stderr:
                    diagnostics.append(stderr.decode(errors="replace").strip())
            finally:
                runtime.writer.close()
                runtime.env.close()
    return " | ".join(item for item in diagnostics if item)


def _collect_batch(
    specs: tuple[MatchSpec, ...],
    *,
    actor_hidden_size: int,
    session: Any,
    env_executable: Path,
    writer_executable: Path,
    schedule_path: Path,
    schedule: dict[str, Any],
    staging: Path,
    max_steps: int,
    threshold: float,
    min_gap_ticks: int,
) -> list[dict[str, Any]]:
    runtimes: list[_MatchRuntime] = []
    try:
        for spec in specs:
            runtimes.append(_start_runtime(
                spec, env_executable=env_executable,
                writer_executable=writer_executable,
                schedule_path=schedule_path, staging=staging,
                max_steps=max_steps, threshold=threshold,
                min_gap_ticks=min_gap_ticks,
            ))
    except BaseException as exc:
        diagnostic = _abort_runtimes(runtimes)
        raise RuntimeError(
            f"start batched DAgger workers: {exc}; "
            f"native writers: {diagnostic or '<no stderr>'}"
        ) from exc
    evaluator = AI42ONNXEvaluationActor(
        session, len(runtimes) * (HERO_COUNT // 2), actor_hidden_size,
    )
    indices = np.concatenate([
        worker * HERO_COUNT + _candidate_indices(runtime.candidate_side)
        for worker, runtime in enumerate(runtimes)
    ])
    opponent_indices = [
        np.setdiff1d(np.arange(HERO_COUNT), _candidate_indices(runtime.candidate_side))
        for runtime in runtimes
    ]
    try:
        with ThreadPoolExecutor(max_workers=len(runtimes)) as pool:
            while not all(runtime.done for runtime in runtimes):
                observations = [runtime.observation for runtime in runtimes]
                actions, controls, _model_controls = evaluator.act(observations, indices)
                margins = evaluator.last_decision_margin.reshape(len(runtimes), -1)
                action_rows = actions.reshape(len(runtimes), -1, 4)
                control_rows = controls.reshape(len(runtimes), -1)
                pending: dict[int, Any] = {}
                selections: dict[int, np.ndarray] = {}
                for worker, runtime in enumerate(runtimes):
                    if runtime.done:
                        continue
                    local = _candidate_indices(runtime.candidate_side)
                    alive = runtime.observation.hero[local, 9] < 0.5
                    selected = runtime.gate.select(margins[worker], alive, runtime.ticks)
                    wire = np.zeros((HERO_COUNT, 6), dtype=np.int16)
                    wire[local, 0] = control_rows[worker]
                    wire[local, 1:5] = action_rows[worker]
                    wire[local, 5] = selected.astype(np.int16)
                    pending[worker] = pool.submit(runtime.env.step, wire)
                    selections[worker] = selected
                for worker, future in pending.items():
                    runtime = runtimes[worker]
                    observation = future.result()
                    request_frame, result_frame = runtime.env.take_step_exchange()
                    runtime.writer.write(request_frame, result_frame)
                    local = _candidate_indices(runtime.candidate_side)
                    runtime.interventions += int(selections[worker].sum())
                    runtime.invalid_actions += int(observation.invalid[local].sum())
                    runtime.candidate_reward += float(observation.rewards[local].sum())
                    labels = np.isin(
                        np.asarray(observation.teacher_status, dtype=np.uint8),
                        TEACHER_LABEL_STATUSES,
                    )
                    runtime.candidate_labels += int(labels[local].sum())
                    runtime.opponent_labels += int(labels[opponent_indices[worker]].sum())
                    runtime.observation = observation
                    runtime.ticks += 1
                    if observation.done:
                        _finish_runtime(runtime, staging, schedule)
    except BaseException as exc:
        diagnostic = _abort_runtimes(runtimes)
        raise RuntimeError(
            f"batched DAgger collection failed: {exc}; "
            f"native writers: {diagnostic or '<no stderr>'}"
        ) from exc
    return [
        {
            "match_id": runtime.spec.match_id,
            "candidate_side": runtime.candidate_side,
            "ticks": runtime.ticks,
            "simulated_minutes": runtime.ticks / 5.0 / 60.0,
            "interventions": runtime.interventions,
            "intervention_rate": runtime.interventions / max(runtime.ticks * 5, 1),
            "candidate_teacher_labels": runtime.candidate_labels,
            "opponent_teacher_labels": runtime.opponent_labels,
            "invalid_actions": runtime.invalid_actions,
            "candidate_reward": runtime.candidate_reward,
            "writer_stdout": runtime.writer_stdout,
            "writer_stderr": runtime.writer_stderr,
        }
        for runtime in runtimes
    ]


def collect_dagger_generation(
    checkpoint: Path,
    config: Path,
    env_executable: Path,
    writer_executable: Path,
    output: Path,
    onnx_path: Path,
    *,
    seed: int,
    matches: int,
    workers: int,
    max_steps: int,
    intervention_margin: float,
    intervention_gap_ticks: int,
    split_seed: int,
    validation_fraction: float,
    device: torch.device,
    staging: Path | None = None,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    staging = staging or output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"DAgger staging already exists: {staging}")
    actor, lineage = load_actor(checkpoint, config, torch.device("cpu"))
    schedule, specs = build_dagger_schedule(
        seed=seed, matches=matches, max_steps=max_steps, lineage=lineage,
        threshold=intervention_margin, min_gap_ticks=intervention_gap_ticks,
        split_seed=split_seed, validation_fraction=validation_fraction,
    )
    export_ai42_actor(actor, onnx_path)
    session = create_onnx_session(str(onnx_path), cuda=device.type == "cuda")
    staging.joinpath("matches").mkdir(parents=True)
    schedule_path = staging / "schedule.json"
    _atomic_write(schedule_path, canonical_json_bytes(schedule))
    started = time.perf_counter()
    match_metrics: list[dict[str, Any]] = []
    active_workers = min(workers, matches)
    for start in range(0, matches, active_workers):
        match_metrics.extend(_collect_batch(
            specs[start:start + active_workers], actor_hidden_size=actor.hidden_size,
            session=session, env_executable=env_executable,
            writer_executable=writer_executable, schedule_path=schedule_path,
            schedule=schedule, staging=staging, max_steps=max_steps,
            threshold=intervention_margin,
            min_gap_ticks=intervention_gap_ticks,
        ))
    merge = merge_go_staging(
        output, staging, schedule=schedule, specs=specs,
        split_seed=split_seed, validation_fraction=validation_fraction,
    )
    elapsed = time.perf_counter() - started
    total_ticks = sum(int(item["ticks"]) for item in match_metrics)
    result = {
        "format": "AI42-intervention-dagger-generation-v1",
        "dataset": str(output.resolve()),
        "staging": str(staging.resolve()),
        "onnx": str(onnx_path.resolve()),
        "matches": matches,
        "workers": active_workers,
        "ticks": total_ticks,
        "elapsed_seconds": elapsed,
        "ticks_per_second": total_ticks / max(elapsed, 1e-9),
        "interventions": sum(int(item["interventions"]) for item in match_metrics),
        "candidate_teacher_labels": sum(int(item["candidate_teacher_labels"]) for item in match_metrics),
        "opponent_teacher_labels": sum(int(item["opponent_teacher_labels"]) for item in match_metrics),
        "invalid_actions": sum(int(item["invalid_actions"]) for item in match_metrics),
        "merge": merge,
        "match_metrics": match_metrics,
    }
    _atomic_write(output.with_name(output.name + ".collection.json"), canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--intervention-margin", type=float, required=True)
    parser.add_argument("--intervention-gap-ticks", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = collect_dagger_generation(
        args.checkpoint, args.config, args.env, args.writer, args.output, args.onnx,
        seed=args.seed, matches=args.matches, workers=args.workers,
        max_steps=args.max_steps, intervention_margin=args.intervention_margin,
        intervention_gap_ticks=args.intervention_gap_ticks,
        split_seed=args.split_seed, validation_fraction=args.validation_fraction,
        device=torch.device(args.device), staging=args.staging,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
