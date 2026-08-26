"""Training-free, resumable multi-match AI-30 cloning dataset collector.

The collector only launches the v13 headless Assault environment. It submits
zero projected actions so the server's AI-30 controller remains the teacher,
spools complete matches to disk, and publishes through the strict dataset
replay/hash gate. No learner, optimizer, torch, or training code is imported.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, ContextManager

import numpy as np

from .collect_ai42 import AI42Collector
from .dataset_ai42 import (
    AI42DatasetError,
    AI42DatasetStaging,
    SHARD_SCHEMA_VERSION,
    publish_staged_dataset,
)
from .env import (
    AI40_ROSTER,
    AI42_PROTOCOL_VERSION,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    AssaultVectorEnv,
    CONTROLLER_AI20,
    CONTROLLER_AI30,
    HERO_COUNT,
    StepResult,
)
from .trajectory_ai42 import canonical_json_bytes, hash_payload


COMMAND = "tanat-ai42-build-dataset"
STEP_SECONDS = 0.2
STEPS_PER_SECOND = int(round(1.0 / STEP_SECONDS))
DEFAULT_TIMEOUT_MINUTES = 15.0
DEFAULT_MAX_STEPS = int(DEFAULT_TIMEOUT_MINUTES * 60 * STEPS_PER_SECOND)
DEFAULT_WORKERS = 8
DEFAULT_QUEUE_SIZE = 2
DEFAULT_SHARD_SIZE = 1
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42

EnvFactory = Callable[[str | Path, int, int], ContextManager[Any]]


def validate_timeout(max_steps: int, timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES) -> int:
    """Validate an exact simulated timeout/cadence mapping.

    The production default is 15:00 at the authoritative 200 ms AssaultTick.
    Tests or explicitly planned scenarios may use another minute value, but
    the step count must still map exactly to that duration.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(timeout_minutes, bool) or not isinstance(timeout_minutes, (int, float)):
        raise ValueError("timeout_minutes must be numeric")
    timeout = float(timeout_minutes)
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_minutes must be positive and finite")
    expected = timeout * 60.0 * STEPS_PER_SECOND
    if not expected.is_integer() or int(expected) != max_steps:
        raise ValueError(
            f"max_steps={max_steps} does not map exactly to {timeout:g}:00 at "
            f"{STEPS_PER_SECOND} Hz; expected {int(expected) if expected.is_integer() else expected}"
        )
    return max_steps


def deterministic_seed_schedule(seed: int, match_count: int) -> tuple[int, ...]:
    """Return stable, non-zero signed-int64 seeds independent of worker count."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 1:
        raise ValueError("match_count must be a positive integer")
    result: list[int] = []
    for index in range(match_count):
        digest = hashlib.sha256(f"AI42-seed-v1\0{seed}\0{index}".encode("utf-8")).digest()
        result.append(int.from_bytes(digest[:8], "little") % ((1 << 63) - 1) + 1)
    return tuple(result)


def _scenario_name(value: str) -> str:
    if value not in {"ai30_mirror", "ai30_vs_ai20", "ai20_vs_ai30"}:
        raise ValueError("scenario must be 'ai30_mirror', 'ai30_vs_ai20', or 'ai20_vs_ai30'")
    return value


def _scenario_schedule(scenario: str | Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(scenario, str):
        return (_scenario_name(scenario),) * count
    values = tuple(_scenario_name(item) for item in scenario)
    if len(values) != count:
        raise ValueError("scenario schedule must contain one value per match")
    return values


def _controllers_for_scenario(scenario: str) -> tuple[int, ...]:
    if scenario == "ai30_mirror":
        return (CONTROLLER_AI30,) * HERO_COUNT
    if scenario == "ai30_vs_ai20":
        return (CONTROLLER_AI30,) * (HERO_COUNT // 2) + (CONTROLLER_AI20,) * (HERO_COUNT // 2)
    return (CONTROLLER_AI20,) * (HERO_COUNT // 2) + (CONTROLLER_AI30,) * (HERO_COUNT // 2)


def _roster_for_match(seed: int, index: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence([int(seed), int(index), 0xA142])
    rng = np.random.default_rng(sequence)
    return tuple(int(value) for value in rng.permutation(AI40_ROSTER).tolist())


def _lineage_ids(match_id: str, tick: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parent = tuple(
        f"{match_id}:root:{slot:02d}" if tick == 0
        else f"{match_id}:boundary:{tick - 1}:{slot:02d}"
        for slot in range(HERO_COUNT)
    )
    current = tuple(f"{match_id}:boundary:{tick}:{slot:02d}" for slot in range(HERO_COUNT))
    return parent, current


@dataclass(frozen=True, slots=True)
class MatchSpec:
    index: int
    seed: int
    match_id: str
    scenario: str
    controller_by_slot: tuple[int, ...]
    roster_ids: tuple[int, ...]
    side_by_slot: tuple[int, ...] = (0,) * (HERO_COUNT // 2) + (1,) * (HERO_COUNT // 2)


def build_match_specs(
    seed: int,
    match_count: int,
    scenario: str | Sequence[str],
    *,
    match_id_prefix: str = "ai42-match",
) -> tuple[MatchSpec, ...]:
    if not isinstance(match_id_prefix, str) or not match_id_prefix.strip():
        raise ValueError("match_id_prefix must be a non-empty string")
    seeds = deterministic_seed_schedule(seed, match_count)
    scenarios = _scenario_schedule(scenario, match_count)
    return tuple(
        MatchSpec(
            index=index,
            seed=match_seed,
            match_id=f"{match_id_prefix}-{index:06d}",
            scenario=match_scenario,
            controller_by_slot=_controllers_for_scenario(match_scenario),
            roster_ids=_roster_for_match(match_seed, index),
        )
        for index, (match_seed, match_scenario) in enumerate(zip(seeds, scenarios))
    )


def _runtime_manifest(
    specs: Sequence[MatchSpec],
    *,
    seed: int,
    max_steps: int,
    workers: int,
    shard_size: int,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    schedule = [
        {
            "index": spec.index,
            "match_id": spec.match_id,
            "seed": spec.seed,
            "scenario": spec.scenario,
            "controller_by_slot": list(spec.controller_by_slot),
            "roster_ids": list(spec.roster_ids),
            "side_by_slot": list(spec.side_by_slot),
        }
        for spec in specs
    ]
    return {
        "command": COMMAND,
        "contract_version": "AI42-collector-v1",
        "protocol_version": AI42_PROTOCOL_VERSION,
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "step_seconds": STEP_SECONDS,
        "steps_per_second": STEPS_PER_SECOND,
        "max_steps": max_steps,
        "timeout_seconds": max_steps * STEP_SECONDS,
        "seed": seed,
        "workers": workers,
        "shard_size": shard_size,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "match_schedule": schedule,
    }


def _default_env_factory(executable: str | Path, workers: int, protocol: int) -> AssaultVectorEnv:
    return AssaultVectorEnv(executable, workers, protocol)


def _safe_put(
    captures: queue.Queue[Any],
    capture: Any,
    writer_errors: list[BaseException],
) -> None:
    while True:
        if writer_errors:
            raise AI42DatasetError(f"staging writer failed: {writer_errors[0]}") from writer_errors[0]
        try:
            captures.put(capture, timeout=0.25)
            return
        except queue.Full:
            continue


def _capture_group(
    env: Any,
    specs: Sequence[MatchSpec],
    *,
    max_steps: int,
    runtime_manifest: Mapping[str, Any],
    captures: queue.Queue[Any],
    writer_errors: list[BaseException],
) -> None:
    workers = len(specs)
    env.reset(
        [spec.seed for spec in specs],
        max_steps=max_steps,
        controller_sets=[list(spec.controller_by_slot) for spec in specs],
        rosters=[list(spec.roster_ids) for spec in specs],
    )
    collectors = {
        worker: AI42Collector(
            match_id=spec.match_id,
            hero_ids=tuple(f"{spec.match_id}:hero:{slot:02d}" for slot in range(HERO_COUNT)),
            runtime_manifest=runtime_manifest,
            seed=spec.seed,
            scenario=spec.scenario,
            controller_by_slot=spec.controller_by_slot,
            roster_ids=spec.roster_ids,
            side_by_slot=spec.side_by_slot,
        )
        for worker, spec in enumerate(specs)
    }
    complete: set[int] = set()
    zero_actions = np.zeros((workers, HERO_COUNT, 4), dtype=np.uint16)
    tick_counts = [0] * workers
    while len(complete) < workers:
        results = env.step(zero_actions)
        if len(results) != workers:
            raise AI42DatasetError("vector environment returned an incomplete worker batch")
        for worker, result in enumerate(results):
            if worker in complete:
                continue
            if not isinstance(result, StepResult):
                raise AI42DatasetError("vector environment returned a non-v13 StepResult")
            tick = tick_counts[worker]
            previous, current = _lineage_ids(specs[worker].match_id, tick)
            collectors[worker].record_tick_fast(
                result, zero_actions[worker], previous, current,
            )
            tick_counts[worker] += 1
            if result.done:
                _safe_put(captures, collectors[worker].finish_fast(), writer_errors)
                complete.add(worker)
            elif tick_counts[worker] >= max_steps:
                raise AI42DatasetError("match did not terminate before the configured 15:00 timeout")


def build_ai42_dataset(
    executable: str | Path,
    destination: str | Path,
    *,
    match_count: int = 1,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES,
    scenario: str | Sequence[str] = "ai30_mirror",
    staging: str | Path | None = None,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Collect, resume, strictly validate and atomically publish a dataset."""

    validate_timeout(max_steps, timeout_minutes)
    if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 1:
        raise ValueError("match_count must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
        raise ValueError("queue_size must be a positive integer")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")
    specs = build_match_specs(seed, match_count, scenario)
    runtime_manifest = _runtime_manifest(
        specs,
        seed=seed,
        max_steps=max_steps,
        workers=workers,
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    destination_path = Path(destination)
    if destination_path.exists():
        raise AI42DatasetError("final dataset destination already exists; generations are immutable")
    staging_path = (
        Path(staging)
        if staging is not None
        else destination_path.parent / f".{destination_path.name}.ai42-staging"
    )
    contract = {
        "staging_schema_version": "AI42-staging-v1",
        "dataset_schema_version": "AI42-dataset-v1",
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "protocol_version": AI42_PROTOCOL_VERSION,
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_hash": hash_payload(runtime_manifest),
        "match_count": match_count,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_minutes": timeout_minutes,
        "workers": workers,
        "queue_size": queue_size,
        "shard_size": shard_size,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "match_ids": [spec.match_id for spec in specs],
    }
    spool = AI42DatasetStaging(staging_path, contract)
    staged_ids = spool.match_ids
    unexpected = sorted(set(staged_ids) - {spec.match_id for spec in specs})
    if unexpected:
        raise AI42DatasetError(f"staging contains match IDs outside the exact contract: {unexpected}")
    remaining = [spec for spec in specs if not spool.contains(spec.match_id)]
    writer_errors: list[BaseException] = []
    captures: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
    stage_seconds: list[float] = []

    def spool_writer() -> None:
        while True:
            item = captures.get()
            try:
                if item is None:
                    return
                started = time.perf_counter()
                spool.add_match(item)
                stage_seconds.append(time.perf_counter() - started)
            except BaseException as exc:  # propagate through the producer boundary
                writer_errors.append(exc)
                return
            finally:
                captures.task_done()

    thread = threading.Thread(target=spool_writer, name="ai42-staging-writer", daemon=True)
    thread.start()
    opened = _default_env_factory if env_factory is None else env_factory
    collection_started = time.perf_counter()
    try:
        for start in range(0, len(remaining), workers):
            group = remaining[start : start + workers]
            group_workers = len(group)
            with opened(executable, group_workers, AI42_PROTOCOL_VERSION) as env:
                _capture_group(
                    env, group, max_steps=max_steps, runtime_manifest=runtime_manifest,
                    captures=captures, writer_errors=writer_errors,
                )
                if writer_errors:
                    raise AI42DatasetError(f"staging writer failed: {writer_errors[0]}") from writer_errors[0]
        collection_seconds = time.perf_counter() - collection_started
    finally:
        if writer_errors:
            while True:
                try:
                    captures.get_nowait()
                except queue.Empty:
                    break
                else:
                    captures.task_done()
        else:
            captures.join()
        if thread.is_alive():
            captures.put(None)
            captures.join()
        thread.join(timeout=5)
    if writer_errors:
        raise AI42DatasetError(f"staging writer failed: {writer_errors[0]}") from writer_errors[0]
    publication_started = time.perf_counter()
    dataset = publish_staged_dataset(
        destination_path,
        spool,
        runtime_manifest=runtime_manifest,
        runtime_manifest_hash=hash_payload(runtime_manifest),
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    publication_seconds = time.perf_counter() - publication_started
    total_seconds = time.perf_counter() - collection_started
    return {
        "command": COMMAND,
        "dataset": str(destination_path),
        "staging": str(staging_path),
        "manifest_hash": dataset.manifest_hash,
        "runtime_manifest_hash": dataset.runtime_manifest_hash,
        "matches": len(dataset),
        "ticks": sum(int(entry["tick_count"]) for entry in dataset.manifest["matches"]),
        "workers": workers,
        "max_steps": max_steps,
        "timeout_minutes": timeout_minutes,
        "seed": seed,
        "resumed_matches": match_count - len(remaining),
        "timings_seconds": {
            "collection": collection_seconds,
            "staging": float(sum(stage_seconds)),
            "publication": publication_seconds,
            "total": total_seconds,
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--matches", dest="match_count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--timeout-minutes", type=float)
    parser.add_argument("--scenario", choices=("ai30_mirror", "ai30_vs_ai20", "ai20_vs_ai30"))
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    values: dict[str, Any] = {}
    if args.config is not None:
        values.update(_load_config(args.config))
    for name in (
        "match_count", "seed", "workers", "max_steps", "timeout_minutes", "scenario",
        "staging", "queue_size", "shard_size", "validation_fraction", "split_seed",
    ):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    summary = build_ai42_dataset(args.executable, args.destination, **values)
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMAND", "DEFAULT_MAX_STEPS", "DEFAULT_TIMEOUT_MINUTES", "MatchSpec",
    "build_ai42_dataset", "build_match_specs", "deterministic_seed_schedule",
    "main", "validate_timeout",
]
