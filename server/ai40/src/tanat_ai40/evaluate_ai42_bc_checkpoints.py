"""Evaluate an AI-42 BC checkpoint curve on one frozen validation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from .dataset_ai42 import AI42DatasetError
from .learner_ai42 import AI42LearnerError, manifest_digest
from .train_ai42_bc import (
    AI42TrainingError,
    ProbeSummary,
    _atomic_write_json,
    _iter_plan_batches,
    _sha256_file,
    _training_config_defaults,
    build_parser as build_training_parser,
    evaluate_probe,
    prepare_training_context,
)


REPORT_FORMAT = "AI42-bc-checkpoint-curve-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="evaluate multiple exact AI-42 BC checkpoints with one dataset load",
    )
    parser.add_argument("--config", type=Path, required=True, help="strict AI-42 BC JSON config")
    parser.add_argument("--dataset", type=Path, required=True, help="validated AI-42 dataset directory")
    parser.add_argument("--profile", type=Path, required=True, help="frozen class-balance profile")
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="exact checkpoint to evaluate; repeat in training order",
    )
    parser.add_argument("--output", type=Path, required=True, help="atomic compact curve report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-hash")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--selection-matches", type=int,
        help="evaluate only this frozen prefix of validation matches for cheap candidate ranking",
    )
    parser.add_argument(
        "--evaluation-batch-size", type=int,
        help="pack independent validation sequences more densely without changing checkpoint lineage",
    )
    parser.add_argument(
        "--patience", type=int,
        help="report early-stop readiness after this many checkpoints since the best loss",
    )
    return parser


def _training_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_training_parser()
    parser.set_defaults(**_training_config_defaults(args.config))
    raw = [
        "--config", str(args.config),
        "--dataset", str(args.dataset),
        "--profile", str(args.profile),
        "--output", str(args.output.parent),
        "--device", str(args.device),
    ]
    if args.dataset_hash is not None:
        raw.extend(("--dataset-hash", str(args.dataset_hash)))
    return parser.parse_args(raw)


def compact_probe(summary: ProbeSummary) -> dict[str, Any]:
    metrics = summary.metrics if isinstance(summary.metrics, Mapping) else {}
    action = metrics.get("action", {}) if isinstance(metrics.get("action"), Mapping) else {}
    offset = metrics.get("offset", {}) if isinstance(metrics.get("offset"), Mapping) else {}
    balanced = metrics.get("balanced_accuracy", {}) if isinstance(metrics.get("balanced_accuracy"), Mapping) else {}
    macro_f1 = metrics.get("macro_f1", {}) if isinstance(metrics.get("macro_f1"), Mapping) else {}
    return {
        "loss": summary.loss,
        "batches": summary.batches,
        "supervised_count": summary.supervised_count,
        "action_count": summary.action_count,
        "control_count": summary.control_count,
        "end_to_end_accuracy": float(action.get("end_to_end_accuracy", 0.0)),
        "head_losses": dict(summary.head_losses),
        "head_accuracies": dict(summary.head_accuracies),
        "balanced_accuracy": dict(balanced),
        "macro_f1": dict(macro_f1),
        "offset": {
            key: offset[key]
            for key in (
                "count", "top1_accuracy", "top5_accuracy",
                "mean_manhattan_distance", "mean_manhattan_grid_distance",
            )
            if key in offset
        },
    }


def curve_summary(
    points: Sequence[Mapping[str, Any]], epsilon: float, patience: int | None = None,
) -> dict[str, Any]:
    if not points:
        raise AI42TrainingError("checkpoint curve is empty")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise AI42TrainingError("--epsilon must be finite and non-negative")
    if patience is not None and (isinstance(patience, bool) or patience < 1):
        raise AI42TrainingError("--patience must be a positive integer")
    ordered = sorted(points, key=lambda point: (int(point["step"]), str(point["path"])))
    losses = [float(point["validation"]["loss"]) for point in ordered]
    end_to_end = [float(point["validation"]["end_to_end_accuracy"]) for point in ordered]
    loss_regressions = [
        {
            "from_step": int(previous["step"]),
            "to_step": int(current["step"]),
            "delta": losses[index] - losses[index - 1],
        }
        for index, (previous, current) in enumerate(zip(ordered, ordered[1:]), start=1)
        if losses[index] > losses[index - 1] + epsilon
    ]
    end_to_end_regressions = [
        {
            "from_step": int(previous["step"]),
            "to_step": int(current["step"]),
            "delta": end_to_end[index] - end_to_end[index - 1],
        }
        for index, (previous, current) in enumerate(zip(ordered, ordered[1:]), start=1)
        if end_to_end[index] + epsilon < end_to_end[index - 1]
    ]
    best = min(ordered, key=lambda point: (float(point["validation"]["loss"]), -int(point["step"])))
    best_index = ordered.index(best)
    checkpoints_since_best = len(ordered) - best_index - 1
    return {
        "ordered_steps": [int(point["step"]) for point in ordered],
        "loss_non_increasing": not loss_regressions,
        "end_to_end_non_decreasing": not end_to_end_regressions,
        "loss_regressions": loss_regressions,
        "end_to_end_regressions": end_to_end_regressions,
        "checkpoints_since_best": checkpoints_since_best,
        "patience": patience,
        "early_stop_ready": patience is not None and checkpoints_since_best >= patience,
        "best": {
            "path": str(best["path"]),
            "step": int(best["step"]),
            "epoch": int(best["epoch"]),
            "loss": float(best["validation"]["loss"]),
            "end_to_end_accuracy": float(best["validation"]["end_to_end_accuracy"]),
        },
    }


def evaluate_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isfinite(float(args.epsilon)) or float(args.epsilon) < 0.0:
        raise AI42TrainingError("--epsilon must be finite and non-negative")
    if args.selection_matches is not None and (
        isinstance(args.selection_matches, bool) or args.selection_matches < 1
    ):
        raise AI42TrainingError("--selection-matches must be a positive integer")
    if args.evaluation_batch_size is not None and (
        isinstance(args.evaluation_batch_size, bool) or not 1 <= args.evaluation_batch_size <= 512
    ):
        raise AI42TrainingError("--evaluation-batch-size must be in [1, 512]")
    if args.patience is not None and (isinstance(args.patience, bool) or args.patience < 1):
        raise AI42TrainingError("--patience must be a positive integer")
    paths = [path.resolve() for path in args.checkpoint]
    if len(set(paths)) != len(paths):
        raise AI42TrainingError("checkpoint paths must be unique")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AI42TrainingError("checkpoint does not exist: " + ", ".join(missing))

    training_args = _training_args(args)
    context = prepare_training_context(training_args)
    evaluation_args = argparse.Namespace(**vars(training_args))
    evaluation_args.batch_size = args.evaluation_batch_size or training_args.validation_batch_size
    full_validation_match_ids = tuple(context.plan["validation_match_ids"])
    validation_match_ids = (
        full_validation_match_ids
        if args.selection_matches is None
        else full_validation_match_ids[:args.selection_matches]
    )
    probe_hash = hashlib.sha256(json.dumps(
        {
            "validation_probe_hash": context.plan["validation_probe_hash"],
            "match_ids": list(validation_match_ids),
            "evaluation_batch_size": evaluation_args.batch_size,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    started = time.monotonic()
    points: list[dict[str, Any]] = []
    for path in paths:
        state = context.learner.load_checkpoint(
            path, context.manifest, map_location=context.device, restore_rng=False,
        )
        if state.extra.get("validation_probe_hash") != context.plan["validation_probe_hash"]:
            raise AI42TrainingError(f"checkpoint validation probe is incompatible: {path}")
        validation = evaluate_probe(
            context.learner,
            _iter_plan_batches(
                context.dataset,
                validation_match_ids,
                "validation",
                evaluation_args,
                context.device,
            ),
        )
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        points.append({
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "step": state.step,
            "epoch": state.epoch,
            "validation": compact_probe(validation),
        })

    report = {
        "format": REPORT_FORMAT,
        "dataset_hash": context.dataset_hash,
        "manifest_digest": manifest_digest(context.manifest),
        "validation_probe_hash": context.plan["validation_probe_hash"],
        "evaluation_probe": {
            "kind": "full" if validation_match_ids == full_validation_match_ids else "selection",
            "match_count": len(validation_match_ids),
            "full_match_count": len(full_validation_match_ids),
            "match_ids": list(validation_match_ids),
            "hash": probe_hash,
            "training_batch_size": training_args.batch_size,
            "evaluation_batch_size": evaluation_args.batch_size,
        },
        "device": str(context.device),
        "checkpoint_count": len(points),
        "elapsed_seconds": time.monotonic() - started,
        "epsilon": float(args.epsilon),
        "curve": curve_summary(points, float(args.epsilon), args.patience),
        "checkpoints": points,
    }
    _atomic_write_json(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_checkpoints(args)
    except (AI42DatasetError, AI42LearnerError, OSError, RuntimeError, ValueError) as exc:
        print(f"AI-42 BC checkpoint evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_FORMAT", "build_parser", "compact_probe", "curve_summary",
    "evaluate_checkpoints", "main",
]
