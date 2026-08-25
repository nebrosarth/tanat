"""Deterministic, training-free AI-42 dataset capture smoke command.

This command exercises only the environment, collector, and durable dataset
APIs.  It deliberately has no optimizer, learner, trainer, or torch imports.
The reset result is transport setup evidence, not a policy tick, so it is
intentionally discarded before the first ten-action step is collected.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any, ContextManager, Sequence

import numpy as np

from .collect_ai42 import AI42Collector
from .dataset_ai42 import AI42DatasetError, load_dataset, write_dataset
from .env import (
    AI40_ROSTER,
    AI42_PROTOCOL_VERSION,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    AssaultEnvProcess,
    CONTROLLER_AI30,
    HERO_COUNT,
    HeroAction,
    StepResult,
)
from .trajectory_ai42 import Outcome, canonical_json_bytes, hash_payload


DEFAULT_MAX_STEPS = 8
"""The deliberately small positive default used by the smoke command."""


SMOKE_COMMAND = "tanat-ai42-dataset-smoke"

EnvFactory = Callable[[str | Path], ContextManager[Any]]


def _validate_max_steps(max_steps: int) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a small positive integer")
    return max_steps


def _runtime_manifest(seed: int, max_steps: int) -> dict[str, Any]:
    """Return the exact runtime contract expected on dataset reload."""

    return {
        "command": SMOKE_COMMAND,
        "protocol_version": AI42_PROTOCOL_VERSION,
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "seed": seed,
        "max_steps": max_steps,
        "roster_ids": [int(value) for value in AI40_ROSTER.tolist()],
        "controllers": [CONTROLLER_AI30] * HERO_COUNT,
    }


def _default_env_factory(executable: str | Path) -> AssaultEnvProcess:
    return AssaultEnvProcess(executable, protocol_version=AI42_PROTOCOL_VERSION)


def _lineage_ids(match_id: str, tick: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    previous = tuple(
        f"{match_id}:root:{slot:02d}" if tick == 0
        else f"{match_id}:boundary:{tick - 1}:{slot:02d}"
        for slot in range(HERO_COUNT)
    )
    current = tuple(f"{match_id}:boundary:{tick}:{slot:02d}" for slot in range(HERO_COUNT))
    return previous, current


def _validate_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise AI42DatasetError(
                "dataset destination must be absent or an empty directory; generations are immutable"
            )


def _assert_reloaded_coverage(
    dataset: Any,
    *,
    match_id: str,
    terminal_winner: int,
    runtime_manifest: Mapping[str, Any],
) -> int:
    if dataset.manifest.get("runtime_manifest") != dict(runtime_manifest):
        raise AI42DatasetError("reloaded dataset runtime manifest differs from the capture manifest")
    if dataset.match_ids() != (match_id,) or dataset.match_ids("validation"):
        raise AI42DatasetError("reloaded dataset does not contain exactly one training match")

    entry = next(item for item in dataset.manifest["matches"] if item["match_id"] == match_id)
    if len(entry["hero_ids"]) != HERO_COUNT or len(set(entry["hero_ids"])) != HERO_COUNT:
        raise AI42DatasetError("reloaded match does not contain ten unique hero slots")
    tick_count = int(entry["tick_count"])
    if tick_count < 1:
        raise AI42DatasetError("reloaded match has no policy ticks")

    parent_rows = entry["recurrent_parent_ids"]
    boundary_rows = entry["recurrent_boundary_ids"]
    if len(parent_rows) != tick_count or len(boundary_rows) != tick_count:
        raise AI42DatasetError("reloaded match has incomplete recurrent coverage")
    for tick in range(tick_count):
        expected_parent, expected_boundary = _lineage_ids(match_id, tick)
        if tuple(parent_rows[tick]) != expected_parent or tuple(boundary_rows[tick]) != expected_boundary:
            raise AI42DatasetError("reloaded match recurrent lineage is not deterministic")

    arrays = dataset.arrays_for_match(match_id)
    for name in ("hero", "abilities", "entities", "global", "projected_action", "rewards"):
        if arrays[name].shape[0] != tick_count or arrays[name].shape[1] != HERO_COUNT:
            raise AI42DatasetError(f"reloaded {name} array does not cover all ten slots")
    done = np.asarray(arrays["done"])
    winner = np.asarray(arrays["winner"])
    if done.shape != (tick_count,) or int(done[-1]) != 1 or np.any(done[:-1] != 0):
        raise AI42DatasetError("reloaded dataset does not have exactly one full terminal boundary")
    if winner.shape != (tick_count,) or int(winner[-1]) != terminal_winner:
        raise AI42DatasetError("reloaded terminal winner differs from the captured result")
    return tick_count


def run_dataset_smoke(
    executable: str | Path,
    destination: str | Path,
    *,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS,
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Capture one deterministic ten-slot AI-42 match and verify its reload."""

    max_steps = _validate_max_steps(max_steps)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    destination_path = Path(destination)
    _validate_destination(destination_path)

    runtime_manifest = _runtime_manifest(seed, max_steps)
    match_id = f"ai42-smoke:{seed}"
    hero_ids = tuple(f"hero:{slot:02d}" for slot in range(HERO_COUNT))
    collector = AI42Collector(
        match_id=match_id,
        hero_ids=hero_ids,
        runtime_manifest=runtime_manifest,
    )
    opened = _default_env_factory if env_factory is None else env_factory
    tick_count = 0
    terminal_winner: int | None = None

    # The context manager closes the real process, and injected fakes, on all
    # reset/step/collector failures as well as on successful completion.
    with opened(executable) as env:
        env.reset(
            seed,
            max_steps,
            # AssaultEnvProcess.reset normalizes ordinary sequences; passing
            # the NumPy constant itself would trigger ambiguous truth testing.
            roster=AI40_ROSTER.tolist(),
            controllers=[CONTROLLER_AI30] * HERO_COUNT,
        )
        while True:
            submitted_actions = [HeroAction() for _ in range(HERO_COUNT)]
            result = env.step(submitted_actions)
            if not isinstance(result, StepResult):
                raise AI42DatasetError("environment step did not return a v13 StepResult")
            previous, current = _lineage_ids(match_id, tick_count)
            outcomes = [
                Outcome(
                    reward=float(result.rewards[slot]),
                    terminal=bool(result.done),
                    winner=int(result.winner),
                )
                for slot in range(HERO_COUNT)
            ]
            collector.record_tick(result, submitted_actions, previous, current, outcomes)
            tick_count += 1
            if result.done:
                terminal_winner = int(result.winner)
                break
            if tick_count >= max_steps:
                raise AI42DatasetError("environment exceeded the configured positive max_steps")

    if terminal_winner is None:
        raise AI42DatasetError("smoke capture ended without a terminal result")
    capture = collector.finish()
    write_dataset(
        destination_path,
        [capture],
        runtime_manifest=runtime_manifest,
        validation_fraction=0.0,
        split_seed=seed,
    )
    reloaded = load_dataset(destination_path, expected_runtime_manifest=runtime_manifest)
    reloaded_ticks = _assert_reloaded_coverage(
        reloaded,
        match_id=match_id,
        terminal_winner=terminal_winner,
        runtime_manifest=runtime_manifest,
    )
    if reloaded_ticks != tick_count:
        raise AI42DatasetError("reloaded tick count differs from captured tick count")

    return {
        "command": SMOKE_COMMAND,
        "dataset": str(destination_path),
        "manifest_hash": reloaded.manifest_hash,
        "runtime_manifest_hash": hash_payload(runtime_manifest),
        "match_id": match_id,
        "seed": seed,
        "max_steps": max_steps,
        "protocol_version": AI42_PROTOCOL_VERSION,
        "hero_slots": HERO_COUNT,
        "ticks": tick_count,
        "terminal_ticks": 1,
        "terminal_winner": terminal_winner,
        "validation_fraction": 0.0,
    }


run_smoke = run_dataset_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="path to the Assault environment executable")
    parser.add_argument("destination", type=Path, help="absent or empty directory for the new dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"small positive episode limit (default: {DEFAULT_MAX_STEPS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_dataset_smoke(
        args.executable,
        args.destination,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_STEPS",
    "SMOKE_COMMAND",
    "main",
    "run_dataset_smoke",
    "run_smoke",
]
