"""Strict, training-free audit for published AI42 dataset generations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bridge_ai42 import (
    REJECTION_REASON_INVALID,
    REJECTION_REASON_MASKED,
    REJECTION_REASON_NONE,
    REJECTION_REASON_POLICY_ERROR,
    REJECTION_REASON_SAFETY,
    REJECTION_REASON_SERVER_REJECTED,
    REJECTION_REASON_TIMEOUT,
    REJECTION_REASON_UNKNOWN,
    TEACHER_STATUS_ACTION,
    TEACHER_STATUS_CANCEL,
    TEACHER_STATUS_HOLD,
    TEACHER_STATUS_NONE,
    TEACHER_STATUS_UNAVAILABLE,
    TEACHER_STATUS_WAIT,
)
from .dataset_ai42 import AI42DatasetError, load_dataset
from .go_shard_ai42 import load_go_dataset
from .trajectory_ai42 import RejectionReason, canonical_json_bytes


_STATUS_NAMES = {
    TEACHER_STATUS_NONE: "none",
    TEACHER_STATUS_ACTION: "action",
    TEACHER_STATUS_WAIT: "wait",
    TEACHER_STATUS_HOLD: "hold",
    TEACHER_STATUS_CANCEL: "cancel",
    TEACHER_STATUS_UNAVAILABLE: "unavailable",
}
_KIND_NAMES = {
    0: "wait", 1: "move", 2: "attack", 3: "skill_1", 4: "skill_2",
    5: "skill_3", 6: "skill_4", 7: "teleport",
}
_REJECTION_NAMES = {
    REJECTION_REASON_NONE: RejectionReason.NONE.value,
    REJECTION_REASON_MASKED: RejectionReason.MASKED.value,
    REJECTION_REASON_INVALID: RejectionReason.INVALID.value,
    REJECTION_REASON_SERVER_REJECTED: RejectionReason.SERVER_REJECTED.value,
    REJECTION_REASON_SAFETY: RejectionReason.SAFETY.value,
    REJECTION_REASON_TIMEOUT: RejectionReason.TIMEOUT.value,
    REJECTION_REASON_POLICY_ERROR: RejectionReason.POLICY_ERROR.value,
    REJECTION_REASON_UNKNOWN: RejectionReason.UNKNOWN.value,
}


def _counter(counter: Counter[str | int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda value: str(value))}


def _stats(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "total": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "total": float(array.sum()),
    }


def audit_ai42_dataset(
    root: str | Path,
    *,
    expected_runtime_manifest: Mapping[str, Any] | Any | None = None,
    split: str | None = None,
    format: str = "python",
) -> dict[str, Any]:
    """Load and audit one immutable generation, retaining one match at a time."""

    if split not in (None, "train", "validation"):
        raise ValueError("split must be 'train', 'validation', or omitted")
    if format == "python":
        dataset = load_dataset(root, expected_runtime_manifest=expected_runtime_manifest)
    elif format == "go":
        dataset = load_go_dataset(root, expected_runtime_manifest=expected_runtime_manifest)
    else:
        raise ValueError("format must be 'python' or 'go'")
    entries = {
        entry["match_id"]: entry
        for entry in dataset.manifest["matches"]
        if split is None or entry["split"] == split
    }
    status_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    roster_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    controller_counts: Counter[str] = Counter()
    duration_ticks: list[int] = []
    duration_seconds: list[float] = []
    terminal_ticks = 0
    terminal_exact = True
    tick_count = 0
    raw_bytes = sum(int(shard["raw_bytes"]) for shard in dataset.manifest["shards"])
    stored_bytes = sum(int(shard["stored_bytes"]) for shard in dataset.manifest["shards"])
    for match_id in sorted(entries):
        entry = entries[match_id]
        arrays = dataset.arrays_for_match(match_id)
        ticks = int(entry["tick_count"])
        tick_count += ticks
        duration_ticks.append(ticks)
        elapsed = np.asarray(arrays["elapsed"], dtype=np.float64)
        duration_seconds.append(float(elapsed[-1]) if elapsed.size else 0.0)
        done = np.asarray(arrays["done"])
        terminal = int(np.count_nonzero(done))
        terminal_ticks += terminal
        terminal_exact = terminal_exact and terminal == 1 and int(done[-1]) == 1
        winners = np.asarray(arrays["winner"])
        winner_counts[str(int(winners[-1]))] += 1
        statuses = np.asarray(arrays["teacher_status"], dtype=np.int64)
        for value in statuses.reshape(-1):
            status_counts[_STATUS_NAMES.get(int(value), f"unknown_{int(value)}")] += 1
        actions = np.asarray(arrays["teacher_action"])
        action_kinds = actions["kind"][statuses == TEACHER_STATUS_ACTION]
        for value in action_kinds.reshape(-1):
            name = _KIND_NAMES.get(int(value), f"unknown_{int(value)}")
            kind_counts[name] += 1
            if 3 <= int(value) <= 6:
                skill_counts[name] += 1
        rejections = np.asarray(arrays["rejection_reason"], dtype=np.int64)
        for value in rejections.reshape(-1):
            rejection_counts[_REJECTION_NAMES.get(int(value), f"unknown_{int(value)}")] += 1
        scenario_counts[entry["scenario"]] += 1
        for value in entry["roster_ids"]:
            roster_counts[str(value)] += 1
        for value in entry["side_by_slot"]:
            side_counts[str(value)] += 1
        for value in entry["controller_by_slot"]:
            controller_counts[str(value)] += 1

    return {
        "command": "tanat-ai42-audit-dataset",
        "dataset": str(root),
        "manifest_hash": dataset.manifest_hash,
        "runtime_manifest_hash": dataset.runtime_manifest_hash,
        "format": format,
        "split": split,
        "matches": len(entries),
        "ticks": tick_count,
        "bytes": {
            "raw": raw_bytes,
            "stored": stored_bytes,
            "compression": dataset.compression,
            "ratio": (float(raw_bytes) / float(stored_bytes)) if stored_bytes else None,
        },
        "durations": {
            "ticks": _stats(duration_ticks),
            "seconds": _stats(duration_seconds),
        },
        "winners": _counter(winner_counts),
        "teacher_statuses": _counter(status_counts),
        "action_kinds": _counter(kind_counts),
        "skills": _counter(skill_counts),
        "rejections": _counter(rejection_counts),
        "terminal": {
            "matches": len(entries),
            "terminal_ticks": terminal_ticks,
            "exactly_one_terminal_tick_per_match": terminal_exact,
        },
        "scenario": _counter(scenario_counts),
        "roster": _counter(roster_counts),
        "side": _counter(side_counts),
        "controllers": _counter(controller_counts),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--format", choices=("python", "go"), default="python")
    parser.add_argument("--split", choices=("train", "validation"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(canonical_json_bytes(audit_ai42_dataset(args.dataset, format=args.format, split=args.split)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_ai42_dataset", "main"]
