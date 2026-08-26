"""Strict, no-training AI-42 Torch probe worker.

The worker is deliberately a bounded protocol adapter around the production
AI-42 learner. The Go preflight supervisor sends one canonical JSON object on
stdin and receives one canonical JSON object on stdout. The request hash is
the SHA-256 of the canonical request with ``request_sha256`` removed.  The
worker never calls ``optimizer.step`` and never writes a report or checkpoint
outside its private temporary round-trip directory.

The request shape is intentionally narrow::

    {
      "protocol": "AI42-torch-preflight-v1",
      "request_sha256": "...",
      "seed": 4242,
      "device": "cpu",
      "model": {"hidden_size": 8, "model_width": 8,
                "entity_layers": 1, "num_heads": 2,
                "ff_multiplier": 1, "timing_bins": 2},
      "learner": {"learning_rate": 0.0003, "weight_decay": 0.0001,
                   "class_balance_power": 0.5,
                   "max_gradient_norm": 1.0},
      "warm_start": {"path": "accepted.pt", "sha256": "..."},
      "batch": {"kind": "inline", "sha256": "...", "value": { ... }}
    }

``batch.kind`` is exactly ``inline`` for focused tests or ``bundle`` for the
production Go parent.  A bundle is a canonical JSON file containing the same
batch mapping.  Standalone Python dataset loading is deliberately outside the
worker protocol; bundle dimensions and bytes are bounded before model work.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    AI42_PROTOCOL_VERSION,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from .learner_ai42 import (
    AI42Batch,
    AI42Learner,
    AI42LearnerConfig,
    CheckpointError,
    _artifact_digest,
    _restore_rng_state,
    _rng_state,
    build_learner_manifest,
    inspect_ai42_checkpoint,
    load_ai42_checkpoint,
    load_ai42_model_warm_start,
    save_ai42_checkpoint,
)
from .model_ai42_actor import AI42Actor, parameter_count
from .trajectory_ai42 import canonical_json_bytes


PROTOCOL = "AI42-torch-preflight-v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 64
MAX_BATCH_ROWS = MAX_BATCH_SIZE * MAX_SEQUENCE_LENGTH
MAX_MODEL_PARAMETERS = 16_000_000
MAX_MODEL_WORKING_BYTES = 512 * 1024 * 1024
_MODEL_PARAMETER_BYTES = 4
_MODEL_WORKING_COPIES = 8

_REQUEST_FIELDS = frozenset({
    "protocol", "request_sha256", "seed", "device", "model", "learner",
    "warm_start", "batch",
})
_MODEL_FIELDS = frozenset({
    "hidden_size", "model_width", "entity_layers", "num_heads",
    "ff_multiplier", "timing_bins",
})
_LEARNER_FIELDS = frozenset({
    "learning_rate", "weight_decay", "class_balance_power",
    "max_gradient_norm", "head_weights", "class_weights", "trainable_scope",
})
_LEARNER_REQUIRED = frozenset({
    "learning_rate", "weight_decay", "class_balance_power", "max_gradient_norm",
})
_WARM_START_FIELDS = frozenset({"path", "sha256", "dataset_hash", "allow_dataset_change"})
_INLINE_FIELDS = frozenset({"kind", "sha256", "value"})
_BUNDLE_FIELDS = frozenset({"kind", "sha256", "path"})
_OBSERVATION_FIELDS = frozenset({"hero", "abilities", "entities", "global_state", "hero_ids"})
_MASK_FIELDS = frozenset({"entity_mask", "kind_mask", "target_mask", "skill_target_mask"})
_LABEL_FIELDS = frozenset({"teacher_actions", "teacher_status"})
_BATCH_ROOT_FIELDS = frozenset({
    "observations", "masks", "labels", "padding_mask", "loss_mask", "reset_mask",
    "death_mask", "sequence_ids", "time_indices",
})
_SHA256_LENGTH = 64


class TorchPreflightError(ValueError):
    """A protocol, input, checkpoint, or finite-data failure."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_decode(raw: bytes, *, name: str, limit: int) -> Any:
    if not isinstance(raw, bytes):
        raise TorchPreflightError(f"{name} must be bytes", code="invalid_json")
    if len(raw) > limit:
        raise TorchPreflightError(f"{name} exceeds the {limit}-byte limit", code="size_limit")
    try:
        text = raw.decode("utf-8")
        decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        value, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TorchPreflightError(f"{name} is not valid JSON: {exc}", code="invalid_json") from exc
    if end != len(text):
        raise TorchPreflightError(f"{name} has trailing bytes at offset {end}", code="noncanonical_json")
    try:
        expected = canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TorchPreflightError(f"{name} cannot be canonicalized: {exc}", code="noncanonical_json") from exc
    if expected != raw:
        raise TorchPreflightError(f"{name} is not canonical JSON", code="noncanonical_json")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str, *, required: frozenset[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise TorchPreflightError(f"{name} must be an object", code="schema_error")
    actual = frozenset(value)
    missing = sorted((expected if required is None else required) - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"unknown={extra}")
        raise TorchPreflightError(f"{name} field set mismatch: {', '.join(detail)}", code="schema_error")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TorchPreflightError(f"{name} must be a non-empty string without NUL", code="schema_error")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TorchPreflightError(f"{name} must be an integer", code="schema_error")
    if value < minimum or (maximum is not None and value > maximum):
        raise TorchPreflightError(f"{name} is outside its bounded range", code="bounds_error")
    return value


def _finite_number(value: Any, name: str, *, minimum: float = 0.0, strict_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TorchPreflightError(f"{name} must be a finite number", code="schema_error")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (strict_positive and result <= minimum):
        raise TorchPreflightError(f"{name} must be finite and within bounds", code="nonfinite_data")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or value.lower() != value:
        raise TorchPreflightError(f"{name} must be lower-case SHA-256 hexadecimal", code="hash_mismatch")
    try:
        if len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as exc:
        raise TorchPreflightError(f"{name} must be lower-case SHA-256 hexadecimal", code="hash_mismatch") from exc
    return value


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int


def _fingerprint(value: os.stat_result) -> _FileFingerprint:
    return _FileFingerprint(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
    )


def _opened_file_fingerprint(handle: Any, path: Path, *, limit: int) -> _FileFingerprint:
    try:
        status = os.fstat(handle.fileno())
    except OSError as exc:
        raise TorchPreflightError(f"cannot stat opened file {path}: {exc}", code="io_error") from exc
    if not stat.S_ISREG(status.st_mode):
        raise TorchPreflightError(f"{path} must be a regular file", code="io_error")
    result = _fingerprint(status)
    if result.size > limit:
        raise TorchPreflightError(f"{path} exceeds the {limit}-byte limit", code="size_limit")
    return result


def _read_bounded_file(path: Path, *, limit: int, name: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as handle:
            before = _opened_file_fingerprint(handle, path, limit=limit)
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise TorchPreflightError(f"{name} exceeds the {limit}-byte limit", code="size_limit")
                chunks.append(block)
            after = _opened_file_fingerprint(handle, path, limit=limit)
        current = _fingerprint(path.stat())
    except TorchPreflightError:
        raise
    except OSError as exc:
        raise TorchPreflightError(f"cannot read {name}: {exc}", code="io_error") from exc
    if before != after or before != current or total != before.size:
        raise TorchPreflightError(f"{name} changed while it was read", code="source_changed")
    return b"".join(chunks)


def _hash_bounded_file(path: Path, *, limit: int, name: str) -> tuple[str, _FileFingerprint]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            before = _opened_file_fingerprint(handle, path, limit=limit)
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise TorchPreflightError(f"{name} exceeds the {limit}-byte limit", code="size_limit")
                digest.update(block)
            after = _opened_file_fingerprint(handle, path, limit=limit)
        current = _fingerprint(path.stat())
    except TorchPreflightError:
        raise
    except OSError as exc:
        raise TorchPreflightError(f"cannot hash {name}: {exc}", code="io_error") from exc
    if before != after or before != current or total != before.size:
        raise TorchPreflightError(f"{name} changed while it was hashed", code="source_changed")
    return digest.hexdigest(), before


def _stage_verified_checkpoint(
    source: Path,
    expected_sha256: str,
    private_directory: Path,
) -> tuple[Path, _FileFingerprint]:
    """Copy the exact verified source bytes into a private load artifact."""

    destination = private_directory / "verified-warm-start.pt"
    digest = hashlib.sha256()
    total = 0
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            before = _opened_file_fingerprint(
                source_handle, source, limit=MAX_CHECKPOINT_BYTES,
            )
            while True:
                block = source_handle.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_CHECKPOINT_BYTES:
                    raise TorchPreflightError(
                        f"warm-start checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit",
                        code="size_limit",
                    )
                digest.update(block)
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
            after = _opened_file_fingerprint(
                source_handle, source, limit=MAX_CHECKPOINT_BYTES,
            )
        current = _fingerprint(source.stat())
    except TorchPreflightError:
        raise
    except OSError as exc:
        raise TorchPreflightError(f"cannot stage warm-start checkpoint: {exc}", code="io_error") from exc
    actual_sha256 = digest.hexdigest()
    if before != after or before != current or total != before.size:
        raise TorchPreflightError(
            "warm-start checkpoint changed while it was staged",
            code="source_changed",
        )
    if actual_sha256 != expected_sha256:
        raise TorchPreflightError(
            "warm-start file SHA-256 does not match request",
            code="hash_mismatch",
        )
    staged_sha256, _ = _hash_bounded_file(
        destination,
        limit=MAX_CHECKPOINT_BYTES,
        name="private warm-start artifact",
    )
    if staged_sha256 != expected_sha256:
        raise TorchPreflightError(
            "private warm-start artifact digest mismatch",
            code="hash_mismatch",
        )
    return destination, before


def _reverify_checkpoint_source(
    source: Path,
    expected_sha256: str,
    expected_fingerprint: _FileFingerprint,
) -> None:
    actual_sha256, current = _hash_bounded_file(
        source,
        limit=MAX_CHECKPOINT_BYTES,
        name="warm-start checkpoint source",
    )
    if current != expected_fingerprint or actual_sha256 != expected_sha256:
        raise TorchPreflightError(
            "warm-start checkpoint source changed after staging",
            code="source_changed",
        )


def _model_digest(actor: AI42Actor) -> str:
    return _artifact_digest({name: value.detach() for name, value in actor.state_dict().items()})


def _gradient_digest(actor: AI42Actor) -> str:
    return _artifact_digest({
        name: None if parameter.grad is None else parameter.grad.detach()
        for name, parameter in actor.named_parameters()
    })


def _output_digest(output: Mapping[str, torch.Tensor]) -> str:
    return _artifact_digest({name: output[name].detach() for name in sorted(output)})


def _rng_digest() -> str:
    return _artifact_digest(_rng_state())


def _estimated_model_parameters(model: Mapping[str, Any]) -> int:
    """Compute the actor's parameter count without constructing any tensors."""

    hidden = int(model["hidden_size"])
    width = int(model["model_width"])
    layers = int(model["entity_layers"])
    multiplier = int(model["ff_multiplier"])
    timing_bins = int(model["timing_bins"])

    def linear(inputs: int, outputs: int) -> int:
        return inputs * outputs + outputs

    def projection(inputs: int, outputs: int) -> int:
        return linear(inputs, outputs) + linear(outputs, outputs)

    total = sum(projection(inputs, width) for inputs in (
        HERO_FEATURES, ENTITY_FEATURES, GLOBAL_FEATURES, ABILITY_FEATURES,
    ))
    total += 128 * width
    total += ABILITY_COUNT * width
    total += 2 * linear(width, width)
    attention = (
        4 * width * width + 4 * width
        + 4 * width
        + linear(width, width * multiplier)
        + linear(width * multiplier, width)
    )
    total += layers * attention
    total += projection(width, width)
    total += projection(1, width)
    total += 16 * hidden * width + 4 * hidden * hidden + 8 * hidden
    total += projection(hidden, width)
    total += linear(hidden, 4)
    total += linear(hidden, ACTION_KINDS)
    total += ACTION_KINDS * width
    total += 2 * linear(width, width)
    total += linear(width, NAVIGATION_OFFSETS)
    total += linear(width, NAVIGATION_ANCHORS)
    total += 2 * linear(width, timing_bins)
    return total


def _validate_json_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
    *,
    scalar: str,
) -> None:
    def visit(item: Any, depth: int) -> None:
        if depth == len(shape):
            if scalar == "float":
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    raise TorchPreflightError(f"{name} contains a non-finite or non-numeric value", code="nonfinite_data")
            elif scalar == "bool":
                if not isinstance(item, bool):
                    raise TorchPreflightError(f"{name} must contain JSON booleans", code="schema_error")
            elif isinstance(item, bool) or not isinstance(item, int):
                raise TorchPreflightError(f"{name} must contain JSON integers", code="schema_error")
            return
        if not isinstance(item, list) or len(item) != shape[depth]:
            raise TorchPreflightError(
                f"{name} must have exact shape {list(shape)}",
                code="bounds_error",
            )
        for child in item:
            visit(child, depth + 1)

    visit(value, 0)


def _validate_batch_dimensions(value: Mapping[str, Any], name: str) -> None:
    observations = value["observations"]
    masks = value["masks"]
    labels = value["labels"]
    hero = observations["hero"]
    if not isinstance(hero, list):
        raise TorchPreflightError(f"{name}.observations.hero must be an array", code="schema_error")
    batch_size = len(hero)
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise TorchPreflightError(f"{name} batch size exceeds its hard bound", code="bounds_error")
    if not isinstance(hero[0], list):
        raise TorchPreflightError(f"{name}.observations.hero must contain sequences", code="schema_error")
    sequence_length = len(hero[0])
    if not 1 <= sequence_length <= MAX_SEQUENCE_LENGTH:
        raise TorchPreflightError(f"{name} sequence length exceeds its hard bound", code="bounds_error")
    if batch_size * sequence_length > MAX_BATCH_ROWS:
        raise TorchPreflightError(f"{name} recurrent row count exceeds its hard bound", code="bounds_error")
    prefix = (batch_size, sequence_length)
    for field, suffix in (
        ("hero", (HERO_FEATURES,)),
        ("abilities", (ABILITY_COUNT, ABILITY_FEATURES)),
        ("entities", (MAX_ENTITIES, ENTITY_FEATURES)),
        ("global_state", (GLOBAL_FEATURES,)),
    ):
        _validate_json_array(
            observations[field], prefix + suffix,
            f"{name}.observations.{field}", scalar="float",
        )
    if "hero_ids" in observations:
        hero_ids = observations["hero_ids"]
        hero_id_shape = prefix if isinstance(hero_ids, list) and hero_ids and isinstance(hero_ids[0], list) else (batch_size,)
        _validate_json_array(hero_ids, hero_id_shape, f"{name}.observations.hero_ids", scalar="int")
    _validate_json_array(masks["entity_mask"], prefix + (MAX_ENTITIES,), f"{name}.masks.entity_mask", scalar="bool")
    _validate_json_array(masks["kind_mask"], prefix + (ACTION_KINDS,), f"{name}.masks.kind_mask", scalar="bool")
    target_mask = masks["target_mask"]
    conditioned = (
        isinstance(target_mask, list) and target_mask
        and isinstance(target_mask[0], list) and target_mask[0]
        and isinstance(target_mask[0][0], list)
        and len(target_mask[0][0]) == ACTION_KINDS
    )
    target_shape = prefix + ((ACTION_KINDS, MAX_ENTITIES) if conditioned else (MAX_ENTITIES,))
    _validate_json_array(target_mask, target_shape, f"{name}.masks.target_mask", scalar="bool")
    _validate_json_array(
        masks["skill_target_mask"], prefix + (ABILITY_COUNT, MAX_ENTITIES),
        f"{name}.masks.skill_target_mask", scalar="bool",
    )
    _validate_json_array(labels["teacher_actions"], prefix + (4,), f"{name}.labels.teacher_actions", scalar="int")
    _validate_json_array(labels["teacher_status"], prefix, f"{name}.labels.teacher_status", scalar="int")
    for field in ("padding_mask", "loss_mask", "reset_mask", "death_mask"):
        if field in value:
            _validate_json_array(value[field], prefix, f"{name}.{field}", scalar="bool")
    if "sequence_ids" in value:
        _validate_json_array(value["sequence_ids"], (batch_size,), f"{name}.sequence_ids", scalar="int")
    if "time_indices" in value:
        _validate_json_array(value["time_indices"], prefix, f"{name}.time_indices", scalar="int")


def _validate_batch_mapping(value: Any, name: str = "batch.value") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TorchPreflightError(f"{name} must be an object", code="schema_error")
    _exact_fields(value, _BATCH_ROOT_FIELDS, name, required=frozenset({"observations", "masks", "labels"}))
    observations = value["observations"]
    masks = value["masks"]
    labels = value["labels"]
    _exact_fields(observations, _OBSERVATION_FIELDS, f"{name}.observations", required=frozenset({"hero", "abilities", "entities", "global_state"}))
    _exact_fields(masks, _MASK_FIELDS, f"{name}.masks", required=_MASK_FIELDS)
    _exact_fields(labels, _LABEL_FIELDS, f"{name}.labels", required=_LABEL_FIELDS)
    _validate_batch_dimensions(value, name)
    return dict(value)


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise TorchPreflightError("request root must be an object", code="schema_error")
    _exact_fields(request, _REQUEST_FIELDS, "request", required=_REQUEST_FIELDS)
    if request["protocol"] != PROTOCOL:
        raise TorchPreflightError("request protocol is incompatible", code="protocol_mismatch")
    request_hash = _sha256(request["request_sha256"], "request.request_sha256")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    expected_hash = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if request_hash != expected_hash:
        raise TorchPreflightError("request_sha256 does not match canonical request bytes", code="hash_mismatch")
    _integer(request["seed"], "request.seed", maximum=(1 << 32) - 1)
    _string(request["device"], "request.device")

    model = request["model"]
    _exact_fields(model, _MODEL_FIELDS, "request.model", required=_MODEL_FIELDS)
    for field in _MODEL_FIELDS:
        _integer(model[field], f"request.model.{field}", minimum=1, maximum=4096)
    if model["model_width"] % model["num_heads"]:
        raise TorchPreflightError("request.model.model_width must divide evenly by num_heads", code="schema_error")
    estimated_parameters = _estimated_model_parameters(model)
    estimated_working_bytes = estimated_parameters * _MODEL_PARAMETER_BYTES * _MODEL_WORKING_COPIES
    if estimated_parameters > MAX_MODEL_PARAMETERS or estimated_working_bytes > MAX_MODEL_WORKING_BYTES:
        raise TorchPreflightError(
            "request.model exceeds the preflight parameter or working-memory cap",
            code="bounds_error",
        )

    learner = request["learner"]
    _exact_fields(learner, _LEARNER_FIELDS, "request.learner", required=_LEARNER_REQUIRED)
    for field in _LEARNER_REQUIRED:
        _finite_number(learner[field], f"request.learner.{field}", strict_positive=field == "max_gradient_norm")
    if "head_weights" in learner:
        if not isinstance(learner["head_weights"], Mapping):
            raise TorchPreflightError("request.learner.head_weights must be an object", code="schema_error")
        for name, item in learner["head_weights"].items():
            _string(name, "request.learner.head_weights key")
            _finite_number(item, f"request.learner.head_weights[{name!r}]")
    if "trainable_scope" in learner:
        scope = _string(learner["trainable_scope"], "request.learner.trainable_scope")
        if scope not in {"all", "supervised_heads"}:
            raise TorchPreflightError("request.learner.trainable_scope is invalid", code="schema_error")
    if "class_weights" in learner:
        if not isinstance(learner["class_weights"], Mapping):
            raise TorchPreflightError("request.learner.class_weights must be an object", code="schema_error")
        for name, values in learner["class_weights"].items():
            _string(name, "request.learner.class_weights key")
            if not isinstance(values, list):
                raise TorchPreflightError(f"request.learner.class_weights[{name!r}] must be an array", code="schema_error")
            for index, item in enumerate(values):
                _finite_number(item, f"request.learner.class_weights[{name!r}][{index}]")

    warm_start = request["warm_start"]
    _exact_fields(warm_start, _WARM_START_FIELDS, "request.warm_start", required=_WARM_START_FIELDS)
    _string(warm_start["path"], "request.warm_start.path")
    _sha256(warm_start["sha256"], "request.warm_start.sha256")
    _sha256(warm_start["dataset_hash"], "request.warm_start.dataset_hash")
    if not isinstance(warm_start["allow_dataset_change"], bool):
        raise TorchPreflightError("request.warm_start.allow_dataset_change must be boolean", code="schema_error")

    batch = request["batch"]
    if not isinstance(batch, Mapping) or "kind" not in batch:
        raise TorchPreflightError("request.batch must be an object with kind", code="schema_error")
    kind = batch["kind"]
    if kind == "inline":
        _exact_fields(batch, _INLINE_FIELDS, "request.batch", required=_INLINE_FIELDS)
        _sha256(batch["sha256"], "request.batch.sha256")
        raw_value = canonical_json_bytes(_validate_batch_mapping(batch["value"]))
        if hashlib.sha256(raw_value).hexdigest() != batch["sha256"]:
            raise TorchPreflightError("inline batch sha256 does not match canonical batch bytes", code="hash_mismatch")
    elif kind == "bundle":
        _exact_fields(batch, _BUNDLE_FIELDS, "request.batch", required=_BUNDLE_FIELDS)
        _string(batch["path"], "request.batch.path")
        _sha256(batch["sha256"], "request.batch.sha256")
    else:
        raise TorchPreflightError("request.batch.kind is unsupported", code="schema_error")
    return dict(request)


def _load_batch(spec: Mapping[str, Any]) -> tuple[AI42Batch, str, str]:
    kind = spec["kind"]
    if kind == "inline":
        value = _validate_batch_mapping(spec["value"])
        raw = canonical_json_bytes(value)
        batch = AI42Batch.from_mapping(value)
        source = "inline"
        source_hash = hashlib.sha256(raw).hexdigest()
    elif kind == "bundle":
        path = Path(spec["path"])
        raw = _read_bounded_file(
            path,
            limit=MAX_BUNDLE_BYTES,
            name=f"batch bundle {path}",
        )
        value = _canonical_decode(raw, name=f"batch bundle {path}", limit=MAX_BUNDLE_BYTES)
        value = _validate_batch_mapping(value, name=f"batch bundle {path}")
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != spec["sha256"]:
            raise TorchPreflightError(
                "batch bundle SHA-256 does not match request",
                code="hash_mismatch",
            )
        batch = AI42Batch.from_mapping(value)
        source = "bundle"
    else:  # _validate_request rejects every other kind before worker I/O.
        raise TorchPreflightError("request.batch.kind is unsupported", code="schema_error")
    if batch.batch_size < 1 or batch.sequence_length < 1:
        raise TorchPreflightError("batch must contain at least one recurrent row", code="bounds_error")
    if batch.batch_size > MAX_BATCH_SIZE or batch.sequence_length > MAX_SEQUENCE_LENGTH:
        raise TorchPreflightError("batch exceeds the bounded preflight dimensions", code="bounds_error")
    if batch.batch_size * batch.sequence_length > MAX_BATCH_ROWS:
        raise TorchPreflightError("batch exceeds the bounded recurrent row limit", code="bounds_error")
    if not bool(batch.supervision_mask.any()):
        raise TorchPreflightError("batch contains no supervised rows", code="data_error")
    return batch, source, source_hash


def _device(value: str) -> torch.device:
    requested = value
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise TorchPreflightError(f"invalid device {value!r}: {exc}", code="device_error") from exc
    if device.type not in {"cpu", "cuda"}:
        raise TorchPreflightError("only CPU and CUDA devices are supported", code="device_error")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TorchPreflightError("CUDA was requested but is unavailable", code="device_error")
    return device


def _result_finite(value: Any, name: str = "response") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TorchPreflightError(f"{name} contains a non-finite number", code="nonfinite_data")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _result_finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _result_finite(item, f"{name}[{index}]")


def run_preflight(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one validated request and return an in-memory response payload."""

    validated = _validate_request(request)
    outer_rng = _rng_state()
    deterministic_debug_before = torch.get_deterministic_debug_mode()
    private_directory: tempfile.TemporaryDirectory[str] | None = None
    started = time.perf_counter_ns()
    try:
        seed = int(validated["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_deterministic_debug_mode(2)
        device = _device(validated["device"])
        model_kwargs = {name: int(validated["model"][name]) for name in _MODEL_FIELDS}
        estimated_parameters = _estimated_model_parameters(model_kwargs)
        learner_kwargs: dict[str, Any] = {
            name: float(validated["learner"][name]) for name in _LEARNER_REQUIRED
        }
        for name in ("head_weights", "class_weights"):
            if name in validated["learner"]:
                learner_kwargs[name] = validated["learner"][name]
        if "trainable_scope" in validated["learner"]:
            learner_kwargs["trainable_scope"] = validated["learner"]["trainable_scope"]
        config = AI42LearnerConfig(model_kwargs=model_kwargs, **learner_kwargs)

        warm_spec = validated["warm_start"]
        warm_source_path = Path(warm_spec["path"])
        private_directory = tempfile.TemporaryDirectory(prefix="ai42-torch-preflight-private-")
        warm_path, warm_source_fingerprint = _stage_verified_checkpoint(
            warm_source_path,
            warm_spec["sha256"],
            Path(private_directory.name),
        )
        actual_warm_hash = warm_spec["sha256"]

        batch, batch_source, batch_hash = _load_batch(validated["batch"])
        batch = batch.to(device)
        actor = AI42Actor(**model_kwargs).to(device)
        if parameter_count(actor) != estimated_parameters:
            raise TorchPreflightError(
                "model parameter estimate does not match constructed actor",
                code="invariant_error",
            )
        learner = AI42Learner(actor, config)
        actor.eval()

        # Inspect first so the complete serialized artifact (including its
        # optimizer/RNG/cursor payload) is validated without restoring any of
        # those values into the active learner.
        source_artifact = inspect_ai42_checkpoint(
            warm_path, None, model=actor, map_location=device,
        )
        source_manifest = dict(source_artifact.manifest)
        source_dataset_hash = str(source_manifest["dataset_hash"])
        target_dataset_hash = str(warm_spec["dataset_hash"])
        allow_dataset_change = bool(warm_spec["allow_dataset_change"])
        compatibility_fields = {
            field: source_manifest[field]
            for field in ("protocol_version", "dataset_schema_version", "shard_schema_version")
            if field in source_manifest
        }
        if "protocol_version" in compatibility_fields and compatibility_fields["protocol_version"] != AI42_PROTOCOL_VERSION:
            raise TorchPreflightError("warm-start protocol version is incompatible", code="protocol_mismatch")
        expected_manifest = build_learner_manifest(
            actor, config, target_dataset_hash, **compatibility_fields,
        )

        optimizer_before = _artifact_digest(learner.optimizer.state_dict())
        rng_before = _rng_digest()
        model_before_warm = _model_digest(actor)
        warm_started = load_ai42_model_warm_start(
            warm_path, actor, expected_manifest, map_location=device,
            allow_dataset_change=allow_dataset_change,
        )
        optimizer_after_warm = _artifact_digest(learner.optimizer.state_dict())
        rng_after_warm = _rng_digest()
        model_after_warm = _model_digest(actor)

        actor.train(True)
        forward_started = time.perf_counter_ns()
        with torch.no_grad():
            first_forward = learner.forward(batch)
            second_forward = learner.forward(batch)
        forward_elapsed = time.perf_counter_ns() - forward_started
        first_forward_hash = _output_digest(first_forward)
        second_forward_hash = _output_digest(second_forward)
        deterministic_forward = first_forward_hash == second_forward_hash
        if not deterministic_forward:
            raise TorchPreflightError("recurrent forward is not byte-deterministic", code="determinism_error")
        finite_outputs = all(bool(torch.isfinite(item).all().item()) for item in first_forward.values())
        if not finite_outputs:
            raise TorchPreflightError("recurrent forward produced non-finite output", code="nonfinite_data")

        backward_started = time.perf_counter_ns()
        first_loss = learner.backward(batch, zero_grad=True)
        first_loss_value = float(first_loss.loss.detach().cpu().item())
        if not math.isfinite(first_loss_value):
            raise TorchPreflightError("masked BC loss is non-finite", code="nonfinite_data")
        first_gradient_digest_before_clip = _gradient_digest(actor)
        first_gradient_norm = learner.clip_gradients()
        first_gradient_digest = _gradient_digest(actor)
        backward_elapsed = time.perf_counter_ns() - backward_started
        if not math.isfinite(first_gradient_norm):
            raise TorchPreflightError("gradient norm is non-finite", code="nonfinite_data")
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in actor.parameters()
        )
        if not finite_gradients:
            raise TorchPreflightError("backward produced non-finite gradient", code="nonfinite_data")

        # A second pass proves the bounded recurrent loss/backward/clip path is
        # deterministic, while still deliberately omitting optimizer.step().
        learner.optimizer.zero_grad(set_to_none=True)
        second_loss = learner.backward(batch, zero_grad=True)
        second_loss_value = float(second_loss.loss.detach().cpu().item())
        second_gradient_norm = learner.clip_gradients()
        second_gradient_digest = _gradient_digest(actor)
        deterministic_backward = (
            first_loss_value == second_loss_value
            and first_gradient_norm == second_gradient_norm
            and first_gradient_digest == second_gradient_digest
        )
        if not deterministic_backward:
            raise TorchPreflightError("masked BC backward is not deterministic", code="determinism_error")

        model_before_compute = _model_digest(actor)
        optimizer_before_roundtrip = _artifact_digest(learner.optimizer.state_dict())
        rng_before_roundtrip = _rng_digest()
        roundtrip_started = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="ai42-torch-preflight-") as directory:
            roundtrip_path = Path(directory) / "roundtrip.pt"
            save_ai42_checkpoint(
                roundtrip_path, actor, learner.optimizer, expected_manifest,
                step=0, epoch=0,
                extra={"batch_cursor": 0, "preflight_only": True},
            )
            roundtrip_artifact = inspect_ai42_checkpoint(
                roundtrip_path, expected_manifest, model=actor, map_location=device,
            )
            clone = copy.deepcopy(actor)
            clone_learner = AI42Learner(clone, config)
            clone_optimizer = clone_learner.optimizer
            resumed = load_ai42_checkpoint(
                roundtrip_path, clone, clone_optimizer, expected_manifest,
                map_location=device, restore_rng=False,
            )
            checkpoint_roundtrip = (
                resumed.step == 0
                and resumed.epoch == 0
                and dict(resumed.manifest) == expected_manifest
                and _model_digest(clone) == model_before_compute
                and _artifact_digest(clone_optimizer.state_dict()) == optimizer_before_roundtrip
                and bool(roundtrip_artifact.payload_digest)
            )
            roundtrip_payload_digest = roundtrip_artifact.payload_digest
        roundtrip_elapsed = time.perf_counter_ns() - roundtrip_started
        if not checkpoint_roundtrip:
            raise TorchPreflightError("checkpoint roundtrip invariant failed", code="checkpoint_error")

        learner.optimizer.zero_grad(set_to_none=True)
        model_after = _model_digest(actor)
        optimizer_after = _artifact_digest(learner.optimizer.state_dict())
        rng_after = _rng_digest()
        parameters_unchanged = model_before_compute == model_after
        optimizer_unchanged = optimizer_before_roundtrip == optimizer_after
        rng_unchanged = rng_before_roundtrip == rng_after
        cursor_unchanged = True  # the worker has no resume cursor by contract
        staged_sha256, _ = _hash_bounded_file(
            warm_path,
            limit=MAX_CHECKPOINT_BYTES,
            name="private warm-start artifact",
        )
        if staged_sha256 != actual_warm_hash:
            raise TorchPreflightError(
                "private warm-start artifact changed after loading",
                code="source_changed",
            )
        _reverify_checkpoint_source(
            warm_source_path,
            actual_warm_hash,
            warm_source_fingerprint,
        )
        invariants = {
            "finite_outputs": finite_outputs,
            "finite_loss": math.isfinite(first_loss_value),
            "finite_gradients": finite_gradients,
            "gradient_clip_checked": True,
            "deterministic_recurrent_forward": deterministic_forward,
            "deterministic_masked_backward": deterministic_backward,
            "parameters_unchanged": parameters_unchanged,
            "optimizer_unchanged": optimizer_unchanged,
            "rng_unchanged": rng_unchanged,
            "optimizer_not_restored": optimizer_before == optimizer_after_warm,
            "rng_not_restored": rng_before == rng_after_warm,
            "cursor_not_restored": cursor_unchanged,
            "optimizer_step_called": False,
            "optimizer_authorized": False,
            "final_report_published": False,
            "checkpoint_roundtrip": checkpoint_roundtrip,
            "exact_checkpoint_bytes_loaded": staged_sha256 == actual_warm_hash,
            "checkpoint_source_unchanged": True,
        }
        expected_false = {"optimizer_step_called", "optimizer_authorized", "final_report_published"}
        failed = sorted(
            name for name, value in invariants.items()
            if (name in expected_false and value) or (name not in expected_false and not value)
        )
        if failed:
            raise TorchPreflightError(
                f"one or more preflight invariants failed: {failed}",
                code="invariant_error",
            )
        total_elapsed = time.perf_counter_ns() - started
        result = {
            "protocol": PROTOCOL,
            "ai42_protocol_version": AI42_PROTOCOL_VERSION,
            "device": str(device),
            "parameter_count": parameter_count(actor),
            "estimated_parameter_count": estimated_parameters,
            "batch": {
                "kind": batch_source,
                "sha256": batch_hash,
                "batch_size": batch.batch_size,
                "sequence_length": batch.sequence_length,
                "supervised_count": int(batch.supervision_mask.sum().detach().cpu().item()),
            },
            "warm_start": {
                "file_sha256": warm_started.source_file_sha256,
                "manifest_digest": warm_started.source_manifest_digest,
                "payload_digest": warm_started.source_payload_digest,
                "model_hash": warm_started.source_model_hash,
                "model_artifact_hash": warm_started.source_model_artifact_hash,
                "optimizer_artifact_hash": source_artifact.artifact_hashes["optimizer"],
                "rng_artifact_hash": source_artifact.artifact_hashes["rng"],
                "dataset_hash": warm_started.source_dataset_hash,
                "dataset_changed": warm_started.source_dataset_hash != target_dataset_hash,
                "dataset_change_allowed": allow_dataset_change,
                "source_step": warm_started.source_step,
                "source_epoch": warm_started.source_epoch,
                "source_cursor": source_artifact.extra.get("batch_cursor"),
            },
            "loss": {
                "value": first_loss_value,
                "gradient_norm": first_gradient_norm,
                "gradient_norm_after_repeat": second_gradient_norm,
                "gradient_digest_before_clip": first_gradient_digest_before_clip,
                "gradient_digest": first_gradient_digest,
                "repeat_gradient_digest": second_gradient_digest,
                "summary": first_loss.to_dict(),
            },
            "hashes": {
                "request_sha256": validated["request_sha256"],
                "warm_start_sha256": actual_warm_hash,
                "model_before_warm_start": model_before_warm,
                "model_after_warm_start": model_after_warm,
                "model_after": model_after,
                "optimizer_before": optimizer_before,
                "optimizer_after": optimizer_after,
                "rng_before_warm_start": rng_before,
                "rng_after_warm_start": rng_after_warm,
                "rng_after": rng_after,
                "forward_first": first_forward_hash,
                "forward_second": second_forward_hash,
                "roundtrip_payload": roundtrip_payload_digest,
            },
            "timings_ms": {
                "forward": forward_elapsed / 1_000_000.0,
                "backward_and_clip": backward_elapsed / 1_000_000.0,
                "checkpoint_roundtrip": roundtrip_elapsed / 1_000_000.0,
                "total": total_elapsed / 1_000_000.0,
            },
            "invariants": invariants,
        }
        _result_finite(result)
        return result
    except (TorchPreflightError, CheckpointError, RuntimeError, ValueError, OSError) as exc:
        if isinstance(exc, TorchPreflightError):
            raise
        raise TorchPreflightError(str(exc), code="preflight_error") from exc
    finally:
        try:
            if private_directory is not None:
                private_directory.cleanup()
        finally:
            try:
                _restore_rng_state(outer_rng)
            finally:
                if torch.get_deterministic_debug_mode() != deterministic_debug_before:
                    torch.set_deterministic_debug_mode(deterministic_debug_before)


def _response(*, request_sha256: str | None, result: dict[str, Any] | None = None, error: TorchPreflightError | None = None) -> dict[str, Any]:
    response = {
        "protocol": PROTOCOL,
        "request_sha256": request_sha256,
        "ok": error is None,
        "error": None if error is None else {"code": error.code, "message": str(error)},
        "result": result,
    }
    _result_finite(response)
    return response


def process_request(raw: bytes) -> bytes:
    """Decode, execute, and encode one request without emitting diagnostics."""

    request_sha256: str | None = None
    try:
        request = _canonical_decode(raw, name="request", limit=MAX_REQUEST_BYTES)
        if isinstance(request, Mapping) and isinstance(request.get("request_sha256"), str):
            request_sha256 = request["request_sha256"]
        result = run_preflight(request)
        response = _response(request_sha256=request_sha256, result=result)
    except TorchPreflightError as exc:
        response = _response(request_sha256=request_sha256, error=exc)
    except Exception as exc:  # the supervisor must receive a fail-closed response
        response = _response(
            request_sha256=request_sha256,
            error=TorchPreflightError(f"unexpected worker failure: {exc}", code="worker_error"),
        )
    return canonical_json_bytes(response)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one-request stdin/stdout worker."""

    if argv:
        error = TorchPreflightError("this worker accepts its request on stdin and no CLI arguments", code="schema_error")
        sys.stdout.buffer.write(canonical_json_bytes(_response(request_sha256=None, error=error)))
        return 2
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    output = process_request(raw)
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    try:
        response = json.loads(output)
    except json.JSONDecodeError:
        return 2
    return 0 if response.get("ok") is True else 1


__all__ = [
    "MAX_BATCH_ROWS", "MAX_BATCH_SIZE", "MAX_BUNDLE_BYTES", "MAX_CHECKPOINT_BYTES",
    "MAX_MODEL_PARAMETERS", "MAX_MODEL_WORKING_BYTES", "MAX_REQUEST_BYTES",
    "MAX_SEQUENCE_LENGTH", "PROTOCOL", "TorchPreflightError", "main", "process_request",
    "run_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
