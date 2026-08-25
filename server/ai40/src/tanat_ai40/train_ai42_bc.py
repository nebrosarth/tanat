"""AI-42 behavior-cloning preflight CLI.

The command is intentionally validation-only today.  ``--execute`` is an
explicit gate for the future training implementation and currently returns a
clear not-implemented error.  No optimizer step, environment, simulation,
collection, or legacy AI-40/AI-41 trainer is reachable from this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import torch

from .learner_ai42 import (
    AI42Learner,
    AI42LearnerConfig,
    AI42LearnerError,
    build_learner_manifest,
    iter_ai42_dataset_batches,
)
from .dataset_ai42 import AI42DatasetError, load_dataset
from .model_ai42_actor import AI42Actor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-42 actor-only behavior-cloning preflight")
    parser.add_argument("--config", type=Path, help="strict AI-42 BC preflight JSON config")
    parser.add_argument("--dataset", type=Path, help="validated AI-42 dataset directory; validation only")
    parser.add_argument("--checkpoint", type=Path, help="safe AI-42 checkpoint to validate and resume in memory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--entity-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--timing-bins", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-balance-power", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--dataset-hash", help="optional expected manifest hash")
    parser.add_argument("--execute", action="store_true", help="reserved explicit training gate (not implemented)")
    return parser


def _config_defaults(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AI42LearnerError(f"preflight config cannot be read: {exc}") from exc
    expected = {"protocol_version", "model", "recurrent_batch", "learner"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["protocol_version"] != 13:
        raise AI42LearnerError("preflight config field set/protocol version is invalid")
    model = payload["model"]
    recurrent = payload["recurrent_batch"]
    learner = payload["learner"]
    if not isinstance(model, dict) or set(model) != {
        "hidden_size", "model_width", "entity_layers", "num_heads", "ff_multiplier", "timing_bins",
    }:
        raise AI42LearnerError("preflight config.model is invalid")
    if not isinstance(recurrent, dict) or set(recurrent) != {"sequence_length", "batch_size"}:
        raise AI42LearnerError("preflight config.recurrent_batch is invalid")
    if not isinstance(learner, dict) or set(learner) != {
        "class_balance_power", "max_gradient_norm", "timing_loss_enabled", "optimizer_step_allowed_in_preflight",
    }:
        raise AI42LearnerError("preflight config.learner is invalid")
    if learner["timing_loss_enabled"] is not False or learner["optimizer_step_allowed_in_preflight"] is not False:
        raise AI42LearnerError("preflight config attempts to enable a prohibited operation")
    return {
        **model, **recurrent,
        "class_balance_power": learner["class_balance_power"],
        "max_gradient_norm": learner["max_gradient_norm"],
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.hidden_size < 1 or args.model_width < 1 or args.entity_layers < 1 or args.num_heads < 1 or args.ff_multiplier < 1 or args.timing_bins < 1:
        raise AI42LearnerError("model dimensions must be positive")
    if args.model_width % args.num_heads:
        raise AI42LearnerError("model-width must be divisible by num-heads")
    if args.sequence_length < 1 or args.batch_size < 1:
        raise AI42LearnerError("sequence-length and batch-size must be positive")
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    try:
        device = torch.device(requested_device)
    except RuntimeError as exc:
        raise AI42LearnerError(f"invalid device: {args.device}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AI42LearnerError("CUDA was requested but is unavailable")
    config = AI42LearnerConfig(
        class_balance_power=args.class_balance_power,
        max_gradient_norm=args.max_gradient_norm,
        model_kwargs={
        "hidden_size": args.hidden_size,
        "model_width": args.model_width,
        "entity_layers": args.entity_layers,
        "num_heads": args.num_heads,
        "ff_multiplier": args.ff_multiplier,
        "timing_bins": args.timing_bins,
        },
    )
    actor = AI42Actor(**config.model_kwargs).to(device)
    dataset_summary: dict[str, Any] = {"provided": False}
    dataset_hash = "0" * 64
    first_batch = None
    if args.dataset is not None:
        dataset = load_dataset(args.dataset)
        dataset_hash = dataset.manifest_hash
        if args.dataset_hash is not None and args.dataset_hash != dataset_hash:
            raise AI42LearnerError("--dataset-hash does not match the validated dataset manifest")
        split_ids = dataset.split_match_ids()
        if not split_ids["train"]:
            raise AI42LearnerError("validated dataset has no train matches")
        batches = iter_ai42_dataset_batches(
            dataset, split="train", sequence_length=args.sequence_length, batch_size=args.batch_size,
        )
        first_batch = next(iter(batches), None)
        if first_batch is None:
            raise AI42LearnerError("validated dataset produced no recurrent train batch")
        dataset_summary = {
            "provided": True,
            "path": str(args.dataset),
            "manifest_hash": dataset_hash,
            "matches": len(dataset),
            "train_matches": len(split_ids["train"]),
            "validation_matches": len(split_ids["validation"]),
            "batch_size": first_batch.batch_size,
            "sequence_length": first_batch.sequence_length,
        }
    elif args.dataset_hash is not None:
        raise AI42LearnerError("--dataset-hash requires --dataset")
    manifest = build_learner_manifest(actor, config, dataset_hash)
    learner = AI42Learner(actor, config)
    loss_summary: dict[str, Any] = {"checked": False}
    parameters_unchanged = True
    if first_batch is not None:
        before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        result = learner.backward(first_batch.to(device))
        gradient_norm = learner.clip_gradients()
        learner.optimizer.zero_grad(set_to_none=True)
        parameters_unchanged = all(torch.equal(value, before[name]) for name, value in actor.state_dict().items())
        if not parameters_unchanged:
            raise AI42LearnerError("BC preflight changed model parameters without an optimizer step")
        loss_summary = {"checked": True, "gradient_norm": gradient_norm, **result.to_dict()}
    checkpoint_summary: dict[str, Any] = {"provided": False}
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise AI42LearnerError(f"checkpoint path does not exist: {args.checkpoint}")
        resumed = learner.load_checkpoint(args.checkpoint, manifest)
        checkpoint_summary = {
            "provided": True, "path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size,
            "step": resumed.step, "epoch": resumed.epoch,
        }
    else:
        with tempfile.TemporaryDirectory(prefix="ai42-bc-preflight-") as directory:
            checkpoint = Path(directory) / "roundtrip.pt"
            learner.save_checkpoint(checkpoint, manifest)
            resumed = learner.load_checkpoint(checkpoint, manifest)
            checkpoint_summary = {
                "provided": False, "roundtrip_checked": True,
                "step": resumed.step, "epoch": resumed.epoch,
            }
    return {
        "mode": "preflight",
        "execute_required_for_training": True,
        "training_implemented": False,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "parameters_unchanged": parameters_unchanged,
        "manifest": manifest,
        "dataset": dataset_summary,
        "loss": loss_summary,
        "checkpoint": checkpoint_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args(raw)
    parser = build_parser()
    if known.config is not None:
        try:
            parser.set_defaults(**_config_defaults(known.config))
        except AI42LearnerError as exc:
            print(f"AI-42 preflight failed: {exc}", file=sys.stderr)
            return 2
    args = parser.parse_args(raw)
    if args.execute:
        print("AI-42 behavior-cloning execution is not implemented; no optimizer update was run.", file=sys.stderr)
        return 2
    try:
        result = preflight(args)
    except (AI42DatasetError, AI42LearnerError, OSError, RuntimeError) as exc:
        print(f"AI-42 preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "preflight"]
