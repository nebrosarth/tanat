"""Collect one intervention-DAgger match through ONNX Runtime and native Go shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .env import (
    AI40_ROSTER,
    AI42_DAGGER_PROTOCOL_VERSION,
    CONTROLLER_AI30,
    CONTROLLER_AI40,
    HERO_COUNT,
    AssaultEnvProcess,
)
from .evaluate_ai42 import load_actor
from .export_ai42 import export_ai42_actor
from .inference_ai42_onnx import AI42ONNXEvaluationActor, create_onnx_session
from .trajectory_ai42 import canonical_json_bytes


TEACHER_LABEL_STATUSES = np.asarray((1, 2, 3, 4), dtype=np.uint8)


class NativeDaggerWriter:
    """One fail-closed Go shard writer fed by exact scalar protocol frames."""

    def __init__(
        self, executable: Path, schedule: Path, output: Path,
        match_index: int, reserve_ticks: int,
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(executable), "-schedule", str(schedule),
                "-output", str(output), "-match-index", str(match_index),
                "-reserve-ticks", str(reserve_ticks),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr

    def write(self, request_frame: bytes, result_frame: bytes) -> None:
        self.process.stdin.write(request_frame)
        self.process.stdin.write(result_frame)

    def _read_output(self) -> tuple[bytes, bytes]:
        stdout = b"" if self.process.stdout.closed else self.process.stdout.read()
        stderr = b"" if self.process.stderr.closed else self.process.stderr.read()
        return stdout, stderr

    def finish(self) -> tuple[bytes, bytes]:
        if not self.process.stdin.closed:
            self.process.stdin.close()
        exit_code = self.process.wait(timeout=60)
        stdout, stderr = self._read_output()
        if exit_code != 0:
            raise RuntimeError(
                f"native DAgger writer exited {exit_code}: "
                f"{stderr.decode(errors='replace')}"
            )
        return stdout, stderr

    def abort(self) -> tuple[bytes, bytes]:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        return self._read_output()

    def close(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


class MarginInterventionGate:
    """Deterministic per-hero cooldown over an empirically chosen margin."""

    def __init__(self, actors: int, threshold: float, min_gap_ticks: int) -> None:
        if isinstance(actors, bool) or not isinstance(actors, int) or actors < 1:
            raise ValueError("actors must be a positive integer")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("actors and finite non-negative threshold are required")
        if (
            isinstance(min_gap_ticks, bool)
            or not isinstance(min_gap_ticks, int)
            or min_gap_ticks < 1
        ):
            raise ValueError("min_gap_ticks must be a positive integer")
        self.threshold = float(threshold)
        self.min_gap_ticks = int(min_gap_ticks)
        self.last_tick = np.full(actors, -min_gap_ticks, dtype=np.int64)

    def select(self, margins: np.ndarray, alive: np.ndarray, tick: int) -> np.ndarray:
        margins = np.asarray(margins, dtype=np.float32)
        alive = np.asarray(alive, dtype=np.bool_)
        if margins.shape != self.last_tick.shape or alive.shape != margins.shape:
            raise ValueError("margin/alive vectors do not match intervention actors")
        eligible = (int(tick) - self.last_tick) >= self.min_gap_ticks
        selected = np.isfinite(margins) & (margins <= self.threshold) & alive & eligible
        self.last_tick[selected] = int(tick)
        return selected


def _controllers(candidate_side: int) -> tuple[int, ...]:
    if candidate_side == 1:
        return (CONTROLLER_AI40,) * 5 + (CONTROLLER_AI30,) * 5
    if candidate_side == 2:
        return (CONTROLLER_AI30,) * 5 + (CONTROLLER_AI40,) * 5
    raise ValueError("candidate_side must be 1 or 2")


def _candidate_indices(candidate_side: int) -> np.ndarray:
    start = 0 if candidate_side == 1 else HERO_COUNT // 2
    return np.arange(start, start + HERO_COUNT // 2, dtype=np.intp)


def _build_schedule(
    *, seed: int, max_steps: int, candidate_side: int, roster: tuple[int, ...],
    lineage: dict[str, Any], threshold: float, min_gap_ticks: int,
) -> dict[str, Any]:
    match_id = f"ai42-dagger-{seed}-side{candidate_side}"
    return {
        "backend": "onnxruntime-dagger-v1",
        "command": "tanat-ai42-collect-dagger",
        "contract_version": "AI42-dagger-collector-v1",
        "intervention_policy": {
            "metric": "minimum-masked-top2-logit-margin",
            "threshold": threshold,
            "min_gap_ticks": min_gap_ticks,
        },
        "match_schedule": [{
            "controller_by_slot": list(_controllers(candidate_side)),
            "index": 0,
            "match_id": match_id,
            "roster_ids": list(roster),
            "scenario": "ai42_dagger_vs_ai30",
            "seed": seed,
            "side_by_slot": [0] * 5 + [1] * 5,
        }],
        "max_steps": max_steps,
        "policy_lineage": lineage,
        "split_seed": seed,
        "validation_fraction": 0.0,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def collect_dagger_match(
    checkpoint: Path,
    config: Path,
    env_executable: Path,
    writer_executable: Path,
    output: Path,
    onnx_path: Path,
    *,
    seed: int,
    max_steps: int,
    candidate_side: int,
    intervention_margin: float,
    intervention_gap_ticks: int,
    device: torch.device,
    schedule_path: Path | None = None,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    schedule_path = schedule_path or output.with_name(output.name + ".schedule.json")
    actor, lineage = load_actor(checkpoint, config, torch.device("cpu"))
    export_ai42_actor(actor, onnx_path)
    session = create_onnx_session(str(onnx_path), cuda=device.type == "cuda")
    evaluator = AI42ONNXEvaluationActor(session, HERO_COUNT // 2, actor.hidden_size)
    indices = _candidate_indices(candidate_side)
    gate = MarginInterventionGate(indices.size, intervention_margin, intervention_gap_ticks)
    rng = np.random.default_rng(seed)
    roster = tuple(int(value) for value in rng.permutation(AI40_ROSTER))
    schedule = _build_schedule(
        seed=seed, max_steps=max_steps, candidate_side=candidate_side,
        roster=roster, lineage=lineage, threshold=intervention_margin,
        min_gap_ticks=intervention_gap_ticks,
    )
    _atomic_write(schedule_path, canonical_json_bytes(schedule))

    writer = NativeDaggerWriter(
        writer_executable, schedule_path, output, 0, min(max_steps, 2048),
    )
    started = time.perf_counter()
    interventions = 0
    candidate_labels = 0
    opponent_labels = 0
    invalid_actions = 0
    candidate_reward = 0.0
    ticks = 0
    writer_stdout = writer_stderr = b""
    try:
        with AssaultEnvProcess(env_executable, AI42_DAGGER_PROTOCOL_VERSION) as env:
            observation = env.reset(
                seed=seed, max_steps=max_steps, roster=roster,
                controllers=_controllers(candidate_side),
            )
            while True:
                actions, controls, _model_controls = evaluator.act([observation], indices)
                alive = observation.hero[indices, 9] < 0.5
                selected = gate.select(evaluator.last_decision_margin, alive, ticks)
                wire = np.zeros((HERO_COUNT, 6), dtype=np.int16)
                wire[indices, 0] = controls
                wire[indices, 1:5] = actions
                wire[indices, 5] = selected.astype(np.int16)
                observation = env.step(wire)
                request_frame, result_frame = env.take_step_exchange()
                writer.write(request_frame, result_frame)
                interventions += int(selected.sum())
                invalid_actions += int(observation.invalid[indices].sum())
                candidate_reward += float(observation.rewards[indices].sum())
                statuses = np.asarray(observation.teacher_status, dtype=np.uint8)
                labels = np.isin(statuses, TEACHER_LABEL_STATUSES)
                candidate_labels += int(labels[indices].sum())
                opponent_indices = np.setdiff1d(np.arange(HERO_COUNT), indices)
                opponent_labels += int(labels[opponent_indices].sum())
                ticks += 1
                if observation.done:
                    break
        writer_stdout, writer_stderr = writer.finish()
    except BaseException as exc:
        writer_stdout, writer_stderr = writer.abort()
        diagnostic = writer_stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"DAgger collection failed: {exc}; native writer: {diagnostic or '<no stderr>'}"
        ) from exc
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    metrics = {
        "format": "AI42-intervention-dagger-collection-v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset": str(output.resolve()),
        "schedule": str(schedule_path.resolve()),
        "onnx": str(onnx_path.resolve()),
        "seed": seed,
        "candidate_side": candidate_side,
        "ticks": ticks,
        "simulated_minutes": ticks / 5.0 / 60.0,
        "elapsed_seconds": elapsed,
        "ticks_per_second": ticks / max(elapsed, 1e-9),
        "interventions": interventions,
        "intervention_rate": interventions / max(ticks * indices.size, 1),
        "candidate_teacher_labels": candidate_labels,
        "opponent_teacher_labels": opponent_labels,
        "invalid_actions": invalid_actions,
        "candidate_reward": candidate_reward,
        "writer_stdout": writer_stdout.decode(errors="replace").strip(),
        "writer_stderr": writer_stderr.decode(errors="replace").strip(),
    }
    _atomic_write(output.with_name(output.name + ".collection.json"), canonical_json_bytes(metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--candidate-side", type=int, choices=(1, 2), required=True)
    parser.add_argument("--intervention-margin", type=float, required=True)
    parser.add_argument("--intervention-gap-ticks", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metrics = collect_dagger_match(
        args.checkpoint, args.config, args.env, args.writer, args.output, args.onnx,
        seed=args.seed, max_steps=args.max_steps, candidate_side=args.candidate_side,
        intervention_margin=args.intervention_margin,
        intervention_gap_ticks=args.intervention_gap_ticks,
        device=torch.device(args.device), schedule_path=args.schedule,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
