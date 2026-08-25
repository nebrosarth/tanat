"""Deterministic, explicitly enabled AI-42 behavior-cloning training.

The default command remains a no-update preflight. A run that performs an
optimizer update must pass ``--execute`` and a run directory. Tensor
validation, loss calculation, and exact checkpoint serialization remain in
``learner_ai42``; this module owns the execution evidence and promotion gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from .dataset_ai42 import AI42DatasetError, load_dataset
from .learner_ai42 import (
    AI42Batch,
    AI42Learner,
    AI42LearnerConfig,
    AI42LearnerError,
    LossResult,
    build_learner_manifest,
    inspect_ai42_checkpoint,
    iter_ai42_dataset_batches,
    manifest_digest,
)
from .model_ai42_actor import AI42Actor


MAX_OPTIMIZER_SECONDS = 300.0
DEFAULT_SEED = 4242
DEFAULT_VALIDATION_BATCHES = 16
DEFAULT_CHECKPOINT_INTERVAL = 100
BATCH_PLAN_VERSION = "AI42-bc-batch-plan-v1"
ACCEPTED_POINTER_FORMAT = "AI42-bc-accepted-pointer-v1"
ACCEPTED_POINTER_FILENAME = "accepted_pointer.json"
CHECKPOINT_GENERATION_DIRNAME = "checkpoint_generations"


class AI42TrainingError(AI42LearnerError):
    """Raised for an invalid or unsuccessful executable BC run."""


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    """Count-weighted metrics over one immutable, deterministic probe."""

    loss: float
    batches: int
    supervised_count: int
    action_count: int
    control_count: int
    head_losses: Mapping[str, float]
    head_accuracies: Mapping[str, float] = field(default_factory=dict)
    head_denominators: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "batches": self.batches,
            "supervised_count": self.supervised_count,
            "action_count": self.action_count,
            "control_count": self.control_count,
            "head_losses": dict(self.head_losses),
            "head_accuracies": dict(self.head_accuracies),
            "head_denominators": dict(self.head_denominators),
        }


ValidationSummary = ProbeSummary


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AI42LearnerError(f"{description} cannot be read: {exc}") from exc


def _config_defaults(path: Path) -> dict[str, Any]:
    """Read the legacy validation-only config without weakening its schema."""

    payload = _read_json(path, "preflight config")
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


def _training_config_defaults(path: Path) -> dict[str, Any]:
    """Read the strict executable training config.

    Training settings are in a separate section so a preflight config cannot
    accidentally authorize optimizer updates.
    """

    payload = _read_json(path, "training config")
    expected = {"protocol_version", "model", "recurrent_batch", "learner", "training"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["protocol_version"] != 13:
        raise AI42LearnerError("training config field set/protocol version is invalid")
    model = payload["model"]
    recurrent = payload["recurrent_batch"]
    learner = payload["learner"]
    training = payload["training"]
    if not isinstance(model, dict) or set(model) != {
        "hidden_size", "model_width", "entity_layers", "num_heads", "ff_multiplier", "timing_bins",
    }:
        raise AI42LearnerError("training config.model is invalid")
    if not isinstance(recurrent, dict) or set(recurrent) != {"sequence_length", "batch_size"}:
        raise AI42LearnerError("training config.recurrent_batch is invalid")
    if not isinstance(learner, dict) or set(learner) != {
        "class_balance_power", "max_gradient_norm", "learning_rate", "weight_decay",
    }:
        raise AI42LearnerError("training config.learner is invalid")
    required_training = {
        "seed", "max_optimizer_seconds", "max_steps", "epochs", "validation_batches", "validation_epsilon",
    }
    optional_training = {"checkpoint_interval", "validation_matches"}
    if not isinstance(training, dict) or not required_training.issubset(training) or set(training) - required_training - optional_training:
        raise AI42LearnerError("training config.training is invalid")
    return {
        **model,
        **recurrent,
        "class_balance_power": learner["class_balance_power"],
        "max_gradient_norm": learner["max_gradient_norm"],
        "learning_rate": learner["learning_rate"],
        "weight_decay": learner["weight_decay"],
        **training,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-42 actor-only behavior-cloning preflight/training")
    parser.add_argument("--config", type=Path, help="strict AI-42 BC JSON config")
    parser.add_argument("--dataset", type=Path, help="validated AI-42 dataset directory")
    parser.add_argument("--checkpoint", type=Path, help="preflight checkpoint, or resume checkpoint with --execute")
    parser.add_argument("--resume", type=Path, help="exact AI-42 training checkpoint to resume")
    parser.add_argument("--output", "--run-dir", dest="output", type=Path, help="atomic BC run directory")
    parser.add_argument("--report", type=Path, help="run report path (default: <output>/run_report.json)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--entity-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--timing-bins", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--class-balance-power", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-optimizer-seconds", type=float, default=MAX_OPTIMIZER_SECONDS)
    parser.add_argument("--max-steps", type=int, default=1, help="total optimizer step target, including resumed steps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validation-batches", type=int, default=DEFAULT_VALIDATION_BATCHES)
    parser.add_argument("--validation-matches", type=int, default=None)
    parser.add_argument("--validation-epsilon", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--dataset-hash", help="optional expected manifest hash")
    parser.add_argument("--execute", action="store_true", help="explicitly authorize optimizer updates")
    return parser


def _device_from_arg(value: str) -> torch.device:
    requested = "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)
    try:
        device = torch.device(requested)
    except RuntimeError as exc:
        raise AI42LearnerError(f"invalid device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AI42LearnerError("CUDA was requested but is unavailable")
    return device


def _validate_model_args(args: argparse.Namespace) -> None:
    dimensions = ("hidden_size", "model_width", "entity_layers", "num_heads", "ff_multiplier", "timing_bins")
    if any(getattr(args, name) < 1 for name in dimensions):
        raise AI42LearnerError("model dimensions must be positive")
    if args.model_width % args.num_heads:
        raise AI42LearnerError("model-width must be divisible by num-heads")
    if args.sequence_length < 1 or args.batch_size < 1:
        raise AI42LearnerError("sequence-length and batch-size must be positive")


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AI42TrainingError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AI42TrainingError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise AI42TrainingError(f"{name} must be finite")
    return number


def _validate_training_args(args: argparse.Namespace) -> None:
    _validate_model_args(args)
    if args.max_steps < 0 or args.epochs < 1 or args.validation_batches < 1:
        raise AI42TrainingError("max-steps may be zero; epochs and validation-batches must be positive")
    if args.validation_matches is not None and args.validation_matches < 1:
        raise AI42TrainingError("validation-matches must be positive when provided")
    if args.checkpoint_interval < 1:
        raise AI42TrainingError("checkpoint-interval must be positive")
    budget = _finite_number(args.max_optimizer_seconds, "max-optimizer-seconds")
    if budget <= 0 or budget > MAX_OPTIMIZER_SECONDS:
        raise AI42TrainingError(f"max-optimizer-seconds must be in (0, {MAX_OPTIMIZER_SECONDS:g}]")
    epsilon = _finite_number(args.validation_epsilon, "validation-epsilon")
    if epsilon < 0:
        raise AI42TrainingError("validation-epsilon must be non-negative")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise AI42TrainingError("seed must be a non-negative integer")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish a complete report, preserving an existing report on failure."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def _dataset_match_ids(dataset: Any, split: str) -> tuple[str, ...]:
    if hasattr(dataset, "match_ids") and callable(dataset.match_ids):
        values = dataset.match_ids(split)
    elif hasattr(dataset, "split_match_ids") and callable(dataset.split_match_ids):
        values = dataset.split_match_ids()[split]
    else:
        raise AI42TrainingError("validated dataset does not expose match IDs")
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise AI42TrainingError(f"dataset {split} match IDs are not unique")
    return result


def _ranked_match_ids(dataset: Any, split: str, seed: int, *, limit: int | None = None) -> tuple[str, ...]:
    """Return a seeded hash-ranked, scenario-stratified match order."""

    entries = {}
    manifest = getattr(dataset, "manifest", {})
    if isinstance(manifest, Mapping):
        for entry in manifest.get("matches", ()):
            if isinstance(entry, Mapping) and entry.get("match_id") is not None:
                entries[str(entry["match_id"])] = entry
    groups: dict[str, list[tuple[str, str]]] = {}
    for match_id in sorted(_dataset_match_ids(dataset, split)):
        entry = entries.get(match_id, {})
        scenario = str(entry.get("scenario", "default")) if isinstance(entry, Mapping) else "default"
        rank = hashlib.sha256(
            f"{BATCH_PLAN_VERSION}\0{seed}\0{split}\0{scenario}\0{match_id}".encode("utf-8"),
        ).hexdigest()
        groups.setdefault(scenario, []).append((rank, match_id))
    for values in groups.values():
        values.sort()
    selected: list[str] = []
    # Round-robin over scenario groups gives every represented scenario a
    # chance in the fixed probe while retaining hash-ranked order within it.
    while groups and (limit is None or len(selected) < limit):
        for scenario in sorted(tuple(groups)):
            values = groups[scenario]
            if values:
                selected.append(values.pop(0)[1])
                if limit is not None and len(selected) >= limit:
                    break
            if not values:
                groups.pop(scenario, None)
        if not groups:
            break
    return tuple(selected)


class _OrderedDatasetView:
    """Small adapter that gives the existing batch API a deterministic order."""

    def __init__(self, dataset: Any, match_ids: Sequence[str], split: str):
        self.dataset = dataset
        self.match_ids_order = tuple(match_ids)
        self.split = split
        self.manifest = getattr(dataset, "manifest", {})

    def iter_matches(self, split: str | None = None):
        if split not in (None, self.split):
            raise AI42TrainingError(f"ordered dataset view only supports split {self.split!r}")
        if hasattr(self.dataset, "arrays_for_match") and callable(self.dataset.arrays_for_match):
            for match_id in self.match_ids_order:
                yield match_id, self.dataset.arrays_for_match(match_id)
            return
        available = {
            str(match_id): arrays for match_id, arrays in self.dataset.iter_matches(self.split)
        }
        for match_id in self.match_ids_order:
            if match_id not in available:
                raise AI42TrainingError(f"ordered dataset view cannot find match {match_id!r}")
            yield match_id, available[match_id]


def _iter_plan_batches(
    dataset: Any,
    match_ids: Sequence[str],
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    max_batches: int | None = None,
    skip_batches: int = 0,
):
    if skip_batches < 0:
        raise AI42TrainingError("skip_batches must be non-negative")
    view = _OrderedDatasetView(dataset, match_ids, split)
    for index, batch in enumerate(iter_ai42_dataset_batches(
        view, split=split, sequence_length=args.sequence_length, batch_size=args.batch_size,
    )):
        if index < skip_batches:
            continue
        if max_batches is not None and index - skip_batches >= max_batches:
            break
        yield batch.to(device)


def _batch_plan(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    train_match_ids = _ranked_match_ids(dataset, "train", args.seed)
    validation_limit = args.validation_matches if args.validation_matches is not None else args.validation_batches
    validation_match_ids = _ranked_match_ids(dataset, "validation", args.seed, limit=validation_limit)
    if not train_match_ids or not validation_match_ids:
        raise AI42TrainingError("validated dataset must contain train and validation probe matches")
    train_probe_match_ids = train_match_ids[:1]
    payload = {
        "version": BATCH_PLAN_VERSION,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "train_match_ids": list(train_match_ids),
        "validation_match_ids": list(validation_match_ids),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    validation_digest = hashlib.sha256(
        json.dumps({"version": BATCH_PLAN_VERSION, "seed": args.seed, "match_ids": list(validation_match_ids)}, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    return {
        **payload,
        "train_probe_match_ids": list(train_probe_match_ids),
        "batch_plan_hash": digest,
        "validation_probe_hash": validation_digest,
    }


def evaluate_probe(learner: AI42Learner, batches: Iterable[AI42Batch]) -> ProbeSummary:
    """Evaluate an identical caller-owned probe without changing weights."""

    was_training = learner.actor.training
    learner.actor.eval()
    # LossResult head losses are means over active rows. Weighting each head by
    # its active count gives a stable micro/count-weighted report over tails.
    head_numerators: dict[str, float] = {}
    head_denominators: dict[str, int] = {}
    accuracy_numerators: dict[str, float] = {}
    accuracy_denominators: dict[str, int] = {}
    batches_seen = 0
    supervised = action = control = 0
    try:
        with torch.no_grad():
            for batch in batches:
                result: LossResult = learner.loss(batch)
                total_count = int(result.metrics.get("supervised_count", 0))
                if total_count <= 0:
                    raise AI42TrainingError("probe contains no supervised rows")
                _finite_number(result.loss.detach().cpu().item(), "probe loss")
                for name, value in result.head_losses.items():
                    head_metrics = result.metrics.get("heads", {}).get(name, {})
                    head_count = int(head_metrics.get("count", total_count)) if isinstance(head_metrics, Mapping) else total_count
                    if head_count > 0:
                        head_numerators[name] = head_numerators.get(name, 0.0) + _finite_number(
                            value.detach().cpu().item(), f"probe {name} loss",
                        ) * head_count
                        head_denominators[name] = head_denominators.get(name, 0) + head_count
                        head_accuracy = head_metrics.get("accuracy") if isinstance(head_metrics, Mapping) else None
                        if head_accuracy is not None:
                            accuracy_numerators[name] = accuracy_numerators.get(name, 0.0) + _finite_number(
                                head_accuracy, f"probe {name} accuracy",
                            ) * head_count
                            accuracy_denominators[name] = accuracy_denominators.get(name, 0) + head_count
                supervised += total_count
                action += int(result.metrics.get("action_count", 0))
                control += int(result.metrics.get("control_count", 0))
                batches_seen += 1
    finally:
        learner.actor.train(was_training)
    if not head_denominators:
        raise AI42TrainingError("probe is empty")
    head_losses = {
        name: head_numerators[name] / head_denominators[name]
        for name in sorted(head_numerators)
    }
    # The learner's scalar loss is a weighted sum of head means.  Preserve
    # that contract at probe level: first form each head's micro loss using
    # its own denominator, then apply the configured head weight.  Weighting
    # batch-level totals by supervised_count would mix unlike head metrics.
    total_loss = sum(
        float(learner.config.head_weights.get(name, 1.0)) * value
        for name, value in head_losses.items()
    )
    return ProbeSummary(
        loss=total_loss,
        batches=batches_seen,
        supervised_count=supervised,
        action_count=action,
        control_count=control,
        head_losses=head_losses,
        head_accuracies={
            name: accuracy_numerators[name] / accuracy_denominators[name]
            for name in sorted(accuracy_numerators)
        },
        head_denominators={name: head_denominators[name] for name in sorted(head_denominators)},
    )


def _atomic_copy_file(source: Path, destination: Path) -> Path:
    """Copy one compatibility alias through a same-directory atomic replace."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False,
        ) as handle:
            temporary = handle.name
        shutil.copyfile(source, temporary)
        with open(temporary, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def _checkpoint_extra(
    path: Path,
    expected_manifest: Mapping[str, Any] | None = None,
    *,
    model: torch.nn.Module | None = None,
) -> Mapping[str, Any] | None:
    """Read promotion metadata only from a fully digest-validated artifact."""

    if expected_manifest is None:
        return None
    try:
        artifact = inspect_ai42_checkpoint(path, expected_manifest, model=model, map_location="cpu")
    except Exception:
        return None
    return artifact.extra


def _checkpoint_loss(
    path: Path,
    expected_manifest: Mapping[str, Any] | None = None,
    *,
    model: torch.nn.Module | None = None,
) -> float | None:
    extra = _checkpoint_extra(path, expected_manifest, model=model)
    if extra is None:
        return None
    post = extra.get("validation_post")
    value = post.get("loss") if isinstance(post, Mapping) else extra.get("validation_loss")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _checkpoint_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path.is_file():
            result[name] = {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return result


def _read_accepted_pointer(
    run_root: Path,
    expected_manifest: Mapping[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> tuple[dict[str, Any], float] | None:
    """Return the validated authoritative generation and its validation loss."""

    pointer_path = run_root / ACCEPTED_POINTER_FILENAME
    try:
        pointer = _read_json(pointer_path, "accepted pointer")
    except Exception:
        return None
    if not isinstance(pointer, Mapping) or pointer.get("format") != ACCEPTED_POINTER_FORMAT:
        return None
    expected_pointer_fields = {
        "format", "generation", "checkpoint", "sha256", "bytes", "validation_loss",
        "manifest_digest", "dataset_hash", "step", "epoch",
    }
    if set(pointer) != expected_pointer_fields:
        return None
    generation = pointer.get("generation")
    relative_path = pointer.get("checkpoint")
    expected_sha = pointer.get("sha256")
    expected_bytes = pointer.get("bytes")
    if (
        isinstance(generation, bool) or not isinstance(generation, int) or generation < 1
        or not isinstance(relative_path, str) or Path(relative_path).is_absolute()
        or not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != expected_sha
        or isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 1
    ):
        return None
    checkpoint_path = (run_root / relative_path).resolve()
    generation_root = (run_root / CHECKPOINT_GENERATION_DIRNAME).resolve()
    try:
        checkpoint_path.relative_to(generation_root)
    except ValueError:
        return None
    if checkpoint_path.name != f"generation-{generation:08d}.pt":
        return None
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != expected_bytes:
        return None
    try:
        if _sha256_file(checkpoint_path) != expected_sha:
            return None
        artifact = inspect_ai42_checkpoint(checkpoint_path, expected_manifest, model=model, map_location="cpu")
    except Exception:
        return None
    if pointer.get("manifest_digest") != manifest_digest(expected_manifest):
        return None
    if pointer.get("dataset_hash") != expected_manifest.get("dataset_hash"):
        return None
    if pointer.get("step") != artifact.step or pointer.get("epoch") != artifact.epoch:
        return None
    post = artifact.extra.get("validation_post")
    value = post.get("loss") if isinstance(post, Mapping) else artifact.extra.get("validation_loss")
    try:
        loss = float(value)
        pointer_loss = float(pointer.get("validation_loss"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(loss) or not math.isfinite(pointer_loss) or pointer_loss != loss:
        return None
    return dict(pointer), loss


def _next_generation(run_root: Path, prior_pointer: Mapping[str, Any] | None) -> int:
    if prior_pointer is not None:
        generation = prior_pointer.get("generation")
        if isinstance(generation, int) and not isinstance(generation, bool) and generation > 0:
            return generation + 1
    generation_root = run_root / CHECKPOINT_GENERATION_DIRNAME
    values = []
    for path in generation_root.glob("generation-*.pt"):
        try:
            values.append(int(path.stem.removeprefix("generation-")))
        except ValueError:
            continue
    return max(values, default=0) + 1


def train(args: argparse.Namespace, *, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run one deterministic BC proposal and publish its evidence report."""

    _validate_training_args(args)
    if args.dataset is None:
        raise AI42TrainingError("--dataset is required with --execute")
    if args.output is None:
        raise AI42TrainingError("--output/--run-dir is required with --execute")
    if args.dataset_hash is not None and args.dataset_hash != args.dataset_hash.lower():
        raise AI42TrainingError("--dataset-hash must use lower-case SHA-256")
    device = _device_from_arg(args.device)
    _seed_everything(args.seed)

    dataset = load_dataset(args.dataset)
    dataset_hash = dataset.manifest_hash
    if args.dataset_hash is not None and args.dataset_hash != dataset_hash:
        raise AI42TrainingError("--dataset-hash does not match the validated dataset manifest")
    split_ids = dataset.split_match_ids()
    if not split_ids["train"] or not split_ids["validation"]:
        raise AI42TrainingError("validated dataset must contain both train and validation matches")
    plan = _batch_plan(dataset, args)

    learner_config = AI42LearnerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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
    actor = AI42Actor(**learner_config.model_kwargs).to(device)
    learner = AI42Learner(actor, learner_config)
    manifest = build_learner_manifest(
        actor, learner_config, dataset_hash,
        protocol_version=13,
        dataset_schema_version=dataset.manifest.get("dataset_schema_version"),
        shard_schema_version=dataset.manifest.get("shard_schema_version"),
        seed=args.seed,
        batch_plan_version=BATCH_PLAN_VERSION,
        batch_plan_hash=plan["batch_plan_hash"],
        validation_probe_hash=plan["validation_probe_hash"],
    )

    resume_path = args.resume or args.checkpoint
    resume_state = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise AI42TrainingError(f"resume checkpoint path does not exist: {resume_path}")
        resume_state = learner.load_checkpoint(resume_path, manifest, map_location=device, restore_rng=True)

    batch_cursor = 0
    if resume_state is not None:
        if resume_state.extra.get("batch_plan_hash") != plan["batch_plan_hash"]:
            raise AI42TrainingError("resume checkpoint batch plan is not an exact match")
        raw_cursor = resume_state.extra.get("batch_cursor")
        if isinstance(raw_cursor, bool) or not isinstance(raw_cursor, int) or raw_cursor < 0:
            raise AI42TrainingError("resume checkpoint is missing an exact batch cursor")
        batch_cursor = raw_cursor

    # Both probes use immutable match selections and deterministic batch
    # generation before and after training. Evaluation streams one batch at a
    # time, so a large validation probe remains bounded in memory.
    train_probe = _iter_plan_batches(
        dataset, plan["train_probe_match_ids"], "train", args, device, max_batches=1,
    )
    validation_probe = _iter_plan_batches(
        dataset, plan["validation_match_ids"], "validation", args, device,
    )
    pre_train = evaluate_probe(learner, train_probe)
    pre_validation = evaluate_probe(learner, validation_probe)

    run_root = args.output
    run_root.mkdir(parents=True, exist_ok=True)
    accepted_path = run_root / "accepted.pt"
    best_path = run_root / "best.pt"
    pointer_path = run_root / ACCEPTED_POINTER_FILENAME
    latest_path = run_root / "latest.pt"
    report_path = args.report or (run_root / "run_report.json")
    prior_pointer_result = _read_accepted_pointer(run_root, manifest, model=actor)
    prior_accepted_exists = pointer_path.is_file()
    prior_pointer = prior_pointer_result[0] if prior_pointer_result is not None else None
    prior_accepted_loss = prior_pointer_result[1] if prior_pointer_result is not None else None
    prior_accepted_compatible = prior_accepted_loss is not None

    global_step = 0 if resume_state is None else resume_state.step
    epoch = 0 if resume_state is None else resume_state.epoch
    run_steps = 0
    gradient_norms: list[float] = []
    training_losses: list[float] = []
    deadline_reached = False
    periodic_checkpoint_seconds = 0.0
    periodic_checkpoint_count = 0

    # The budget covers only forward/backward/clip/optimizer operations. Final
    # probes, checkpoint serialization, hashing, and report publication are
    # deliberately outside this clock window.
    optimizer_started = float(clock())
    deadline = optimizer_started + float(args.max_optimizer_seconds)
    learner.actor.train(True)
    while global_step < args.max_steps and epoch < args.epochs and not deadline_reached:
        completed_epoch = True
        saw_batch = False
        for batch in _iter_plan_batches(
            dataset, plan["train_match_ids"], "train", args, device, skip_batches=batch_cursor,
        ):
            saw_batch = True
            if global_step >= args.max_steps:
                completed_epoch = False
                break
            if float(clock()) >= deadline:
                deadline_reached = True
                completed_epoch = False
                break
            result = learner.backward(batch.to(device), zero_grad=True)
            gradient_norm = learner.clip_gradients()
            # Fail closed: a backward pass that consumes the remaining budget
            # never reaches optimizer.step().
            if float(clock()) >= deadline:
                learner.optimizer.zero_grad(set_to_none=True)
                deadline_reached = True
                completed_epoch = False
                break
            learner.optimizer_step()
            run_steps += 1
            global_step += 1
            batch_cursor += 1
            gradient_norms.append(_finite_number(gradient_norm, "gradient norm"))
            training_losses.append(_finite_number(result.loss.detach().cpu().item(), "training loss"))
            if run_steps % args.checkpoint_interval == 0:
                periodic_checkpoint_count += 1
                periodic_extra = {
                    "batch_plan_hash": plan["batch_plan_hash"],
                    "validation_probe_hash": plan["validation_probe_hash"],
                    "batch_cursor": batch_cursor,
                    "accepted": False,
                    "optimizer_steps_this_run": run_steps,
                    "checkpoint_kind": "periodic_latest",
                }
                checkpoint_started = float(clock())
                learner.save_checkpoint(latest_path, manifest, step=global_step, epoch=epoch, extra=periodic_extra)
                periodic_checkpoint_seconds += max(0.0, float(clock()) - checkpoint_started)
            if float(clock()) >= deadline:
                deadline_reached = True
                completed_epoch = False
                break
            if global_step >= args.max_steps:
                # The iterator was stopped in the middle of an epoch. Keep
                # the saved epoch unchanged so an exact resume can continue
                # with the same deterministic stream contract.
                completed_epoch = False
                break
        if completed_epoch and (saw_batch or batch_cursor > 0):
            # A periodic checkpoint can be taken immediately after the final
            # batch, before the iterator returns and the epoch is normalized.
            # Treat an exhausted resumed cursor as that same completed epoch.
            epoch += 1
            batch_cursor = 0
        else:
            break
    optimizer_elapsed_raw = max(0.0, float(clock()) - optimizer_started)
    optimizer_elapsed = min(optimizer_elapsed_raw, float(args.max_optimizer_seconds))

    # Final evidence and artifact publication are outside the optimizer budget.
    post_train = evaluate_probe(learner, _iter_plan_batches(
        dataset, plan["train_probe_match_ids"], "train", args, device, max_batches=1,
    ))
    post_validation = evaluate_probe(learner, _iter_plan_batches(
        dataset, plan["validation_match_ids"], "validation", args, device,
    ))
    epsilon = float(args.validation_epsilon)
    improved_from_baseline = post_validation.loss < (pre_validation.loss - epsilon)
    improved_over_accepted = (
        (not prior_accepted_compatible)
        or (prior_accepted_loss is not None and post_validation.loss < (prior_accepted_loss - epsilon))
    )
    accepted = bool(improved_from_baseline and improved_over_accepted)
    accuracy_regressions = {
        name: {
            "pre": pre_validation.head_accuracies.get(name),
            "post": post_validation.head_accuracies.get(name),
            "delta": (
                post_validation.head_accuracies[name] - pre_validation.head_accuracies[name]
                if name in pre_validation.head_accuracies and name in post_validation.head_accuracies else None
            ),
            "regressed": (
                post_validation.head_accuracies[name] < pre_validation.head_accuracies[name]
                if name in pre_validation.head_accuracies and name in post_validation.head_accuracies else False
            ),
        }
        for name in sorted(set(pre_validation.head_accuracies) | set(post_validation.head_accuracies))
    }

    extra = {
        "train_probe_pre": pre_train.to_dict(),
        "train_probe_post": post_train.to_dict(),
        "validation_pre": pre_validation.to_dict(),
        "validation_post": post_validation.to_dict(),
        "validation_epsilon": epsilon,
        "accepted": accepted,
        "deadline_reached": deadline_reached,
        "batch_plan_hash": plan["batch_plan_hash"],
        "validation_probe_hash": plan["validation_probe_hash"],
        "batch_cursor": batch_cursor,
        "checkpoint_kind": "final_latest",
        "optimizer_steps_this_run": run_steps,
        "gradient_norm_mean": float(sum(gradient_norms) / len(gradient_norms)) if gradient_norms else 0.0,
        "training_loss_mean": float(sum(training_losses) / len(training_losses)) if training_losses else 0.0,
    }
    # latest is always a complete resumable candidate. Accepted promotion is
    # an immutable generation plus one authoritative pointer. The two legacy
    # files remain compatibility aliases and are never read for acceptance.
    learner.save_checkpoint(latest_path, manifest, step=global_step, epoch=epoch, extra=extra)

    generation_path: Path | None = None
    pointer_payload: dict[str, Any] | None = None
    alias_errors: dict[str, str] = {}
    if accepted:
        generation_root = run_root / CHECKPOINT_GENERATION_DIRNAME
        generation_root.mkdir(parents=True, exist_ok=True)
        generation = _next_generation(run_root, prior_pointer)
        generation_path = generation_root / f"generation-{generation:08d}.pt"
        while generation_path.exists():
            generation += 1
            generation_path = generation_root / f"generation-{generation:08d}.pt"
        learner.save_checkpoint(generation_path, manifest, step=global_step, epoch=epoch, extra=extra)
        pointer_payload = {
            "format": ACCEPTED_POINTER_FORMAT,
            "generation": generation,
            "checkpoint": str(generation_path.relative_to(run_root)),
            "sha256": _sha256_file(generation_path),
            "bytes": generation_path.stat().st_size,
            "validation_loss": post_validation.loss,
            "manifest_digest": manifest_digest(manifest),
            "dataset_hash": dataset_hash,
            "step": global_step,
            "epoch": epoch,
        }

    checkpoint_records = _checkpoint_hashes({"latest": latest_path})
    if generation_path is not None and pointer_payload is not None:
        generation_record = _checkpoint_hashes({"generation": generation_path})["generation"]
        checkpoint_records["accepted_generation"] = generation_record
        pointer_bytes = json.dumps(
            pointer_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        checkpoint_records["accepted_pointer"] = {
            "path": str(pointer_path),
            "sha256": hashlib.sha256(pointer_bytes).hexdigest(),
            "bytes": len(pointer_bytes),
        }
        # Both aliases are intended to identify exactly the same immutable
        # generation; their records are updated after publication below.
        for name, path in (("accepted", accepted_path), ("best", best_path)):
            checkpoint_records[name] = {
                "path": str(path),
                "sha256": generation_record["sha256"],
                "bytes": generation_record["bytes"],
            }
    else:
        checkpoint_records.update(_checkpoint_hashes({
            "accepted_pointer": pointer_path,
            "accepted": accepted_path,
            "best": best_path,
        }))

    report: dict[str, Any] = {
        "format": "AI42-bc-run-report-v1",
        "status": "accepted" if accepted else ("deadline" if deadline_reached else "rejected"),
        "accepted": accepted,
        "device": str(device),
        "seed": args.seed,
        "deterministic_order": True,
        "optimizer_budget_seconds": float(args.max_optimizer_seconds),
        "elapsed_optimizer_seconds": optimizer_elapsed,
        "optimizer_elapsed_raw_seconds": optimizer_elapsed_raw,
        "deadline_reached": deadline_reached,
        "optimizer_steps": run_steps,
        "global_step": global_step,
        "epoch": epoch,
        "max_steps": args.max_steps,
        "train_probe_pre": pre_train.to_dict(),
        "train_probe_post": post_train.to_dict(),
        "pre_validation": pre_validation.to_dict(),
        "post_validation": post_validation.to_dict(),
        "improvement": {
            "epsilon": epsilon,
            "from_baseline": improved_from_baseline,
            "over_prior_accepted": improved_over_accepted,
            "prior_accepted_present": prior_accepted_exists,
            "prior_accepted_compatible": prior_accepted_compatible,
            "prior_accepted_loss": prior_accepted_loss,
        },
        "accuracy_regressions": accuracy_regressions,
        "dataset": {
            "manifest_hash": dataset_hash,
            "dataset_schema_version": dataset.manifest.get("dataset_schema_version"),
            "shard_schema_version": dataset.manifest.get("shard_schema_version"),
            "train_matches": len(split_ids["train"]),
            "validation_matches": len(split_ids["validation"]),
            "validation_probe_matches": len(plan["validation_match_ids"]),
            "validation_sample_batches": pre_validation.batches,
        },
        "manifest_digest": manifest_digest(manifest),
        "resume": {
            "provided": resume_state is not None,
            "path": str(resume_path) if resume_path is not None else None,
            "start_step": 0 if resume_state is None else resume_state.step,
            "start_epoch": 0 if resume_state is None else resume_state.epoch,
        },
        "batch_plan": {
            "version": BATCH_PLAN_VERSION,
            "hash": plan["batch_plan_hash"],
            "validation_probe_hash": plan["validation_probe_hash"],
            "batch_cursor": batch_cursor,
            "train_match_ids": plan["train_match_ids"],
            "validation_match_ids": plan["validation_match_ids"],
        },
        "periodic_checkpoint_seconds": periodic_checkpoint_seconds,
        "periodic_checkpoint_count": periodic_checkpoint_count,
        "promotion": {
            "authoritative": ACCEPTED_POINTER_FILENAME,
            "accepted_generation": str(generation_path) if generation_path is not None else None,
            "compatibility_aliases": {
                "accepted": str(accepted_path),
                "best": str(best_path),
                "authoritative": False,
                "errors": alias_errors,
            },
        },
        "checkpoints": checkpoint_records,
    }
    report["hashes"] = {
        "dataset_manifest": dataset_hash,
        "learner_manifest": report["manifest_digest"],
        "checkpoints": {
            name: value["sha256"] for name, value in report["checkpoints"].items()
        },
    }
    # Report first: an injected report failure cannot replace the authoritative
    # pointer or either compatibility alias. The pointer itself is one atomic
    # authoritative promotion operation; aliases may be partially updated and
    # are explicitly non-authoritative.
    _atomic_write_json(report_path, report)
    if accepted and generation_path is not None and pointer_payload is not None:
        _atomic_write_json(pointer_path, pointer_payload)
        for name, alias_path in (("accepted", accepted_path), ("best", best_path)):
            try:
                _atomic_copy_file(generation_path, alias_path)
            except Exception as exc:
                alias_errors[name] = f"{type(exc).__name__}: {exc}"
        for name, alias_path in (("accepted", accepted_path), ("best", best_path)):
            if alias_path.is_file():
                checkpoint_records[name] = _checkpoint_hashes({name: alias_path})[name]
        report["promotion"]["compatibility_aliases"]["errors"] = dict(alias_errors)
        report["hashes"]["checkpoints"] = {
            name: value["sha256"] for name, value in checkpoint_records.items()
        }
        # The report was committed before aliases. Alias errors are returned to
        # the caller, while the on-disk report remains a valid authoritative
        # generation record even if a compatibility copy is unavailable.
    return report


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run the existing no-update validation path."""

    _validate_model_args(args)
    device = _device_from_arg(args.device)
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
    first_batch: AI42Batch | None = None
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
        "training_implemented": True,
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
            payload = _read_json(known.config, "BC config")
            if isinstance(payload, dict) and "training" in payload:
                parser.set_defaults(**_training_config_defaults(known.config))
            else:
                parser.set_defaults(**_config_defaults(known.config))
        except (AI42LearnerError, OSError, ValueError) as exc:
            print(f"AI-42 BC config failed: {exc}", file=sys.stderr)
            return 2
    args = parser.parse_args(raw)
    if not args.execute:
        try:
            result = preflight(args)
        except (AI42DatasetError, AI42LearnerError, OSError, RuntimeError) as exc:
            print(f"AI-42 preflight failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    try:
        result = train(args)
    except (AI42DatasetError, AI42LearnerError, OSError, RuntimeError) as exc:
        print(f"AI-42 BC training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


run_training = train
atomic_write_json = _atomic_write_json
sha256_file = _sha256_file

__all__ = [
    "AI42TrainingError", "DEFAULT_SEED", "MAX_OPTIMIZER_SECONDS", "ProbeSummary", "ValidationSummary",
    "atomic_write_json", "build_parser", "evaluate_probe", "main", "preflight", "run_training", "sha256_file", "train",
]
