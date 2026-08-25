"""Training-free AI-42 runtime boundaries.

This module is deliberately small at the boundary: it knows about the actor
observation contract, recurrent state, manifests, and action selection.  It
does not import a learner or any training-only model and it has no connection
to a live profile or server process.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable
import hashlib
import json
import re
import time

import torch

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from .model_ai42_actor import (
    AI42Actor,
    CONTROL_CLASSES,
    CONTROL_ISSUE,
    CONTROL_NAMES,
)


MANIFEST_HASH_FIELDS = (
    "model_hash",
    "config_hash",
    "checkpoint_hash",
    "observation_hash",
    "action_hash",
    "trajectory_hash",
)
MANIFEST_VERSION_FIELDS = (
    "model_version",
    "config_version",
    "checkpoint_version",
    "observation_version",
    "action_version",
    "trajectory_version",
)
MANIFEST_FIELDS = ("manifest_version",) + MANIFEST_VERSION_FIELDS + MANIFEST_HASH_FIELDS
ACTOR_INPUT_NAMES = (
    "hero",
    "abilities",
    "entities",
    "global_state",
    "entity_mask",
    "h",
    "c",
)
ACTOR_OUTPUT_NAMES = (
    "kind",
    "target",
    "offset",
    "anchor",
    "timing",
    "timing_aux",
    "next_h",
    "next_c",
)
ACTION_MASK_NAMES = ("kind", "target", "offset", "anchor", "timing", "timing_aux")
LIVE_PROFILE_WIRING = False


def hash_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return the canonical SHA-256 digest for a byte payload."""

    return hashlib.sha256(bytes(value)).hexdigest()


def hash_json(value: Any) -> str:
    """Hash JSON data using the same canonical encoding as manifests."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hash_bytes(encoded)


def _normalise_hash(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("manifest hashes must be lowercase SHA-256 hex strings")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("manifest hashes must be exactly 64 lowercase SHA-256 hex characters")
    return value


def _normalise_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class AI42Manifest:
    """Immutable, independently versioned AI-42 artifact manifest.

    Hash values are intentionally not combined into one opaque schema hash.
    A change to one contract therefore identifies the exact incompatible
    component in a mismatch report.
    """

    manifest_version: str = "AI42-manifest-v1"
    model_version: str = "AI42-model-v1"
    config_version: str = "AI42-config-v1"
    checkpoint_version: str = "AI42-checkpoint-v1"
    observation_version: str = "AI42-observation-v1"
    action_version: str = "AI42-action-v1"
    trajectory_version: str = "AI42-trajectory-v1"
    model_hash: str = hash_bytes(b"AI42-model-v1")
    config_hash: str = hash_bytes(b"AI42-config-v1")
    checkpoint_hash: str = hash_bytes(b"AI42-checkpoint-v1")
    observation_hash: str = hash_bytes(b"AI42-observation-v1")
    action_hash: str = hash_bytes(b"AI42-action-v1")
    trajectory_hash: str = hash_bytes(b"AI42-trajectory-v1")

    def __post_init__(self) -> None:
        for field in ("manifest_version",) + MANIFEST_VERSION_FIELDS:
            _normalise_text(getattr(self, field), field)
        for field in MANIFEST_HASH_FIELDS:
            object.__setattr__(self, field, _normalise_hash(getattr(self, field)))

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in MANIFEST_FIELDS}

    def canonical_json(self) -> str:
        """Return deterministic compact JSON with lexicographically sorted keys."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    @property
    def digest(self) -> str:
        return hash_bytes(self.canonical_bytes())

    @property
    def version(self) -> str:
        """Short alias for callers that treat the manifest version generically."""

        return self.manifest_version

    def mismatches(self, actual: "AI42Manifest | Mapping[str, Any]") -> dict[str, tuple[str, str]]:
        return manifest_mismatches(self, actual)

    def assert_compatible(self, actual: "AI42Manifest | Mapping[str, Any]") -> None:
        assert_manifest_compatible(self, actual)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AI42Manifest":
        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        unknown = set(value) - set(MANIFEST_FIELDS)
        missing = set(MANIFEST_FIELDS) - set(value)
        if unknown or missing:
            raise ValueError(
                "manifest field set mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**{field: value[field] for field in MANIFEST_FIELDS})


def build_manifest(
    *,
    model: Any,
    config: Any,
    checkpoint: Any,
    observation: Any,
    action: Any,
    trajectory: Any,
    **versions: str,
) -> AI42Manifest:
    """Build a manifest by hashing six independent artifact payloads.

    ``bytes`` and filesystem paths are hashed as bytes. Other values are
    canonicalized as JSON, which makes config/schema fixtures reproducible.
    """

    def digest(value: Any) -> str:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return hash_bytes(value)
        if isinstance(value, Path):
            return hash_bytes(value.read_bytes())
        if isinstance(value, str):
            return hash_bytes(value.encode("utf-8"))
        return hash_json(value)

    fields = {
        "model_hash": digest(model),
        "config_hash": digest(config),
        "checkpoint_hash": digest(checkpoint),
        "observation_hash": digest(observation),
        "action_hash": digest(action),
        "trajectory_hash": digest(trajectory),
    }
    fields.update(versions)
    return AI42Manifest(**fields)


def _coerce_manifest(value: AI42Manifest | Mapping[str, Any] | str) -> AI42Manifest:
    if isinstance(value, AI42Manifest):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return AI42Manifest.from_dict(value)


def manifest_mismatches(
    expected: AI42Manifest | Mapping[str, Any],
    actual: AI42Manifest | Mapping[str, Any],
) -> dict[str, tuple[str, str]]:
    """Return deterministic ``field -> (expected, actual)`` differences."""

    expected_manifest = _coerce_manifest(expected)
    actual_manifest = _coerce_manifest(actual)
    return {
        field: (getattr(expected_manifest, field), getattr(actual_manifest, field))
        for field in MANIFEST_FIELDS
        if getattr(expected_manifest, field) != getattr(actual_manifest, field)
    }


class ManifestMismatchError(ValueError):
    def __init__(self, mismatches: Mapping[str, tuple[str, str]]):
        self.mismatches = dict(mismatches)
        fields = ", ".join(self.mismatches) or "unknown"
        super().__init__(f"AI-42 manifest mismatch: {fields}")


def assert_manifest_compatible(
    expected: AI42Manifest | Mapping[str, Any],
    actual: AI42Manifest | Mapping[str, Any],
) -> None:
    mismatches = manifest_mismatches(expected, actual)
    if mismatches:
        raise ManifestMismatchError(mismatches)


@dataclass(frozen=True, slots=True)
class CheckpointCompatibilityReport:
    """Auditable result of a complete checkpoint compatibility check."""

    compatible: bool
    loaded: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    mismatch_details: tuple[tuple[str, str], ...] = ()
    migrated: bool = False
    reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.compatible and not self.migrated

    @property
    def rejected(self) -> bool:
        return not self.compatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "loaded": list(self.loaded),
            "skipped": list(self.skipped),
            "mismatched": list(self.mismatched),
            "mismatch_details": [list(item) for item in self.mismatch_details],
            "migrated": self.migrated,
            "reason": self.reason,
        }


class CheckpointCompatibilityError(ValueError):
    def __init__(self, report: CheckpointCompatibilityReport):
        self.report = report
        super().__init__(report.reason or "AI-42 checkpoint is incompatible")


def _read_checkpoint(checkpoint: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        loaded = checkpoint
    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint must be a mapping or a path to one")
    return loaded


def _checkpoint_parts(
    checkpoint: Mapping[str, Any],
) -> tuple[AI42Manifest | None, Mapping[str, Any] | None, list[str]]:
    raw_manifest = checkpoint.get("manifest")
    raw_state = checkpoint.get("state_dict", checkpoint.get("model"))
    errors: list[str] = []
    manifest: AI42Manifest | None = None
    if raw_manifest is not None:
        try:
            manifest = _coerce_manifest(raw_manifest)
        except (TypeError, ValueError) as exc:
            errors.append(f"manifest:{exc}")
    else:
        errors.append("manifest:missing")
    if not isinstance(raw_state, Mapping):
        errors.append("state_dict:missing")
        raw_state = None
    return manifest, raw_state, errors


def _state_report(
    actor: torch.nn.Module,
    state: Mapping[str, Any] | None,
    *,
    prefix_errors: Sequence[str] = (),
    audit_mismatches: Sequence[str] = (),
    migrated: bool = False,
) -> CheckpointCompatibilityReport:
    current_mismatches: list[str] = list(prefix_errors)
    mismatched: list[str] = [*current_mismatches, *audit_mismatches]
    details: list[tuple[str, str]] = []
    skipped: list[str] = []
    expected = actor.state_dict()
    if state is None:
        current_mismatches.append("state_dict:missing")
        mismatched.append("state_dict:missing")
        return CheckpointCompatibilityReport(
            False, skipped=tuple(expected), mismatched=tuple(mismatched),
            migrated=migrated, reason="; ".join(mismatched),
        )
    expected_keys = set(expected)
    actual_keys = set(state)
    for key in sorted(expected_keys - actual_keys):
        issue = f"state_dict:{key}:missing"
        current_mismatches.append(issue)
        mismatched.append(issue)
    for key in sorted(actual_keys - expected_keys):
        skipped.append(key)
        issue = f"state_dict:{key}:unexpected"
        current_mismatches.append(issue)
        mismatched.append(issue)
    for key in sorted(expected_keys & actual_keys):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            issue = f"state_dict:{key}:not_tensor"
            current_mismatches.append(issue)
            mismatched.append(issue)
            details.append((key, "not_tensor"))
            continue
        if tuple(value.shape) != tuple(expected[key].shape):
            issue = f"state_dict:{key}:shape"
            current_mismatches.append(issue)
            mismatched.append(issue)
            details.append((key, f"shape {tuple(value.shape)} != {tuple(expected[key].shape)}"))
        if value.dtype != expected[key].dtype:
            issue = f"state_dict:{key}:dtype"
            current_mismatches.append(issue)
            mismatched.append(issue)
            details.append((key, f"dtype {value.dtype} != {expected[key].dtype}"))
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            issue = f"state_dict:{key}:non_finite"
            current_mismatches.append(issue)
            mismatched.append(issue)
            details.append((key, "contains NaN or Inf"))
    compatible = not current_mismatches
    return CheckpointCompatibilityReport(
        compatible,
        loaded=tuple(sorted(expected_keys)) if compatible else (),
        skipped=tuple(sorted(skipped)),
        mismatched=tuple(mismatched),
        mismatch_details=tuple(details),
        migrated=migrated,
        reason=None if compatible else "; ".join(mismatched),
    )


def checkpoint_compatibility_report(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any] | str | Path,
    expected_manifest: AI42Manifest | Mapping[str, Any],
) -> CheckpointCompatibilityReport:
    """Inspect a checkpoint without mutating ``actor`` or loading any tensor."""

    saved = _read_checkpoint(checkpoint)
    actual_manifest, state, errors = _checkpoint_parts(saved)
    if actual_manifest is not None:
        errors.extend(
            f"manifest:{field}" for field in manifest_mismatches(expected_manifest, actual_manifest)
        )
    return _state_report(actor, state, prefix_errors=errors)


check_checkpoint_compatibility = checkpoint_compatibility_report


def load_ai42_checkpoint(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any] | str | Path,
    expected_manifest: AI42Manifest | Mapping[str, Any],
    *,
    migration: Callable[[Mapping[str, Any], AI42Manifest | None, AI42Manifest], Mapping[str, Any]] | None = None,
    strict: bool = False,
) -> CheckpointCompatibilityReport:
    """Load only a complete compatible state dict.

    A migration callback is an explicit opt-in.  Its returned state is checked
    against the complete target state before ``load_state_dict`` is called, so
    a migration can never silently load a subset.
    """

    saved = _read_checkpoint(checkpoint)
    expected = _coerce_manifest(expected_manifest)
    source_manifest, state, errors = _checkpoint_parts(saved)
    manifest_diff = (
        manifest_mismatches(expected, source_manifest)
        if source_manifest is not None else {}
    )
    if manifest_diff:
        errors.extend(f"manifest:{field}" for field in manifest_diff)
    migrated = False
    audit_mismatches: list[str] = []
    preflight = _state_report(actor, state, prefix_errors=errors)
    if not preflight.compatible and migration is not None and state is not None:
        audit_mismatches = [*preflight.mismatched]
        state = migration(state, source_manifest, expected)
        migrated = True
        # A migration is allowed to repair a manifest mismatch, but not to
        # bypass a malformed checkpoint or produce an implicit partial load.
        errors = []
    report = _state_report(
        actor,
        state,
        prefix_errors=errors,
        audit_mismatches=audit_mismatches,
        migrated=migrated,
    )
    if report.compatible:
        # This is deliberately after every key/shape/dtype check above.
        before_load = {
            name: tensor.detach().clone()
            for name, tensor in actor.state_dict().items()
        }
        try:
            actor.load_state_dict(state, strict=True)  # type: ignore[arg-type]
        except Exception:
            # Keep the no-partial-load guarantee even if a backend rejects a
            # tensor during the final assignment.
            actor.load_state_dict(before_load, strict=True)
            raise
    if strict and not report.compatible:
        raise CheckpointCompatibilityError(report)
    return report


def load_compatible_checkpoint(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any] | str | Path,
    expected_manifest: AI42Manifest | Mapping[str, Any],
) -> CheckpointCompatibilityReport:
    """Strict exact-load convenience wrapper; migration is not implicit."""

    report = load_ai42_checkpoint(actor, checkpoint, expected_manifest, strict=True)
    return report


@dataclass(frozen=True, slots=True)
class RecurrentSnapshot:
    h: torch.Tensor
    c: torch.Tensor
    hidden_size: int
    slot_count: int = HERO_COUNT
    device: str = "cpu"
    dtype: str = "torch.float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "h": self.h,
            "c": self.c,
            "hidden_size": self.hidden_size,
            "slot_count": self.slot_count,
            "device": self.device,
            "dtype": self.dtype,
        }


class HeroLifecycle:
    MATCH_RESET = "match_reset"
    DEATH = "death"
    RESPAWN = "respawn"


class RecurrentStateStore:
    """Own exactly one recurrent state pair for each of ten stable hero slots."""

    slot_count = HERO_COUNT

    def __init__(
        self,
        hidden_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = dtype
        shape = (self.slot_count, self.hidden_size)
        self._h = torch.zeros(shape, device=self.device, dtype=dtype)
        self._c = torch.zeros(shape, device=self.device, dtype=dtype)

    def _check_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < self.slot_count:
            raise IndexError(f"hero slot must be an integer in [0, {self.slot_count}), got {slot!r}")
        return slot

    def _check_tensor(self, value: torch.Tensor, name: str) -> None:
        expected = (self.hidden_size,)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(value.shape) not in (expected, (1, self.hidden_size)):
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if value.device != self.device:
            raise ValueError(f"{name} device {value.device} does not match store device {self.device}")
        if value.dtype != self.dtype:
            raise ValueError(f"{name} dtype {value.dtype} does not match store dtype {self.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")

    def get(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self._check_slot(slot)
        return self._h[slot].clone(), self._c[slot].clone()

    state_for = get

    def set(self, slot: int, h: torch.Tensor, c: torch.Tensor) -> None:
        slot = self._check_slot(slot)
        self._check_tensor(h, "h")
        self._check_tensor(c, "c")
        h_value = h.reshape(self.hidden_size).detach()
        c_value = c.reshape(self.hidden_size).detach()
        self._h[slot].copy_(h_value)
        self._c[slot].copy_(c_value)

    def reset_slot(self, slot: int) -> None:
        slot = self._check_slot(slot)
        self._h[slot].zero_()
        self._c[slot].zero_()

    def reset_match(self) -> None:
        self._h.zero_()
        self._c.zero_()

    def apply_event(self, slot: int | None, event: str) -> None:
        if event == HeroLifecycle.MATCH_RESET:
            self.reset_match()
        elif event in (HeroLifecycle.DEATH, HeroLifecycle.RESPAWN):
            if slot is None:
                raise ValueError(f"{event} requires a hero slot")
            self.reset_slot(slot)
        else:
            raise ValueError(f"unknown hero lifecycle event: {event!r}")

    def on_death(self, slot: int) -> None:
        self.apply_event(slot, HeroLifecycle.DEATH)

    def on_respawn(self, slot: int) -> None:
        self.apply_event(slot, HeroLifecycle.RESPAWN)

    def snapshot(self) -> RecurrentSnapshot:
        return RecurrentSnapshot(
            h=self._h.detach().clone(),
            c=self._c.detach().clone(),
            hidden_size=self.hidden_size,
            slot_count=self.slot_count,
            device=str(self.device),
            dtype=str(self.dtype),
        )

    def restore(self, snapshot: RecurrentSnapshot | Mapping[str, Any]) -> None:
        if isinstance(snapshot, Mapping):
            snapshot = RecurrentSnapshot(
                h=snapshot["h"],
                c=snapshot["c"],
                hidden_size=int(snapshot["hidden_size"]),
                slot_count=int(snapshot.get("slot_count", HERO_COUNT)),
                device=str(snapshot.get("device", self.device)),
                dtype=str(snapshot.get("dtype", self.dtype)),
            )
        if not isinstance(snapshot, RecurrentSnapshot):
            raise TypeError("snapshot must be a RecurrentSnapshot or mapping")
        if snapshot.hidden_size != self.hidden_size or snapshot.slot_count != self.slot_count:
            raise ValueError("snapshot recurrent shape contract does not match this store")
        if snapshot.device != str(self.device):
            raise ValueError(f"snapshot device {snapshot.device} does not match store device {self.device}")
        if snapshot.dtype != str(self.dtype):
            raise ValueError(f"snapshot dtype {snapshot.dtype} does not match store dtype {self.dtype}")
        if not isinstance(snapshot.h, torch.Tensor) or not isinstance(snapshot.c, torch.Tensor):
            raise TypeError("snapshot h and c must be tensors")
        expected = (self.slot_count, self.hidden_size)
        if tuple(snapshot.h.shape) != expected or tuple(snapshot.c.shape) != expected:
            raise ValueError(f"snapshot state must have shape {expected}")
        if snapshot.h.device != self.device or snapshot.c.device != self.device:
            raise ValueError("snapshot tensor device does not match store")
        if snapshot.h.dtype != self.dtype or snapshot.c.dtype != self.dtype:
            raise ValueError("snapshot tensor dtype does not match store")
        if not bool(torch.isfinite(snapshot.h).all()) or not bool(torch.isfinite(snapshot.c).all()):
            raise ValueError("snapshot contains non-finite values")
        self._h.copy_(snapshot.h)
        self._c.copy_(snapshot.c)


class InferenceSchemaError(ValueError):
    pass


class ActionMaskError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceResult(Mapping[str, Any]):
    action: Mapping[str, Any]
    raw_output: Mapping[str, Any] | None
    used_fallback: bool
    reason: str | None
    slot: int

    def __getitem__(self, key: str) -> Any:
        return self.action[key]

    def __iter__(self):
        return iter(self.action)

    def __len__(self) -> int:
        return len(self.action)

    @property
    def fallback(self) -> bool:
        return self.used_fallback


class InferenceTelemetry:
    """Counters plus a deterministic bounded window of recent latencies."""

    def __init__(
        self,
        *,
        latency_sample_size: int = 256,
        latency_threshold_seconds: float | None = None,
    ) -> None:
        if latency_sample_size < 1:
            raise ValueError("latency_sample_size must be positive")
        if latency_threshold_seconds is not None and latency_threshold_seconds < 0:
            raise ValueError("latency_threshold_seconds must be non-negative")
        self.calls = 0
        self.successes = 0
        self.fallbacks = 0
        self.reasons: Counter[str] = Counter()
        self.last_latency_seconds = 0.0
        self.total_latency_seconds = 0.0
        self.max_latency_seconds = 0.0
        self.latency_threshold_seconds = latency_threshold_seconds
        self.latency_threshold_exceeded = 0
        self._latency_samples: deque[float] = deque(maxlen=latency_sample_size)

    def _record_latency(self, latency_seconds: float) -> None:
        latency = max(0.0, float(latency_seconds))
        self.last_latency_seconds = latency
        self.total_latency_seconds += latency
        self.max_latency_seconds = max(self.max_latency_seconds, latency)
        self._latency_samples.append(latency)
        if self.latency_threshold_seconds is not None and latency > self.latency_threshold_seconds:
            self.latency_threshold_exceeded += 1

    def record_success(self, latency_seconds: float = 0.0) -> None:
        self.calls += 1
        self.successes += 1
        self._record_latency(latency_seconds)

    def record_fallback(self, reason: str, latency_seconds: float = 0.0) -> None:
        self.calls += 1
        self.fallbacks += 1
        self.reasons[reason] += 1
        self._record_latency(latency_seconds)

    def _percentile(self, fraction: float) -> float:
        if not self._latency_samples:
            return 0.0
        ordered = sorted(self._latency_samples)
        index = int((len(ordered) - 1) * fraction)
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        latency = {
            "last": self.last_latency_seconds,
            "total": self.total_latency_seconds,
            "max": self.max_latency_seconds,
            "p50": self._percentile(0.50),
            "p95": self._percentile(0.95),
            "p99": self._percentile(0.99),
            "sample_count": len(self._latency_samples),
            "sample_capacity": self._latency_samples.maxlen,
        }
        return {
            "calls": self.calls,
            "successes": self.successes,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(sorted(self.reasons.items())),
            "last_latency_seconds": self.last_latency_seconds,
            "total_latency_seconds": self.total_latency_seconds,
            "max_latency_seconds": self.max_latency_seconds,
            "latency_p50_seconds": self._percentile(0.50),
            "latency_p95_seconds": self._percentile(0.95),
            "latency_p99_seconds": self._percentile(0.99),
            "latency_sample_count": len(self._latency_samples),
            "latency_sample_capacity": self._latency_samples.maxlen,
            "latency_threshold_seconds": self.latency_threshold_seconds,
            "latency_threshold_exceeded": self.latency_threshold_exceeded,
            "latency_seconds": latency,
        }


def _actor_device_dtype(actor: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(actor.parameters())
    except (AttributeError, StopIteration):
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device)


def _batched_observation(observation: Mapping[str, Any], actor: Any) -> dict[str, torch.Tensor]:
    required = {"hero", "abilities", "entities", "global_state", "entity_mask"}
    missing = required - set(observation)
    allowed = required | {"hero_ids"}
    unknown = set(observation) - allowed
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing={sorted(missing)}")
        if unknown:
            detail.append(f"unknown={sorted(unknown)}")
        raise InferenceSchemaError("actor observation schema mismatch: " + ", ".join(detail))
    device, dtype = _actor_device_dtype(actor)
    values = {
        "hero": _as_tensor(observation["hero"], device=device, dtype=dtype),
        "abilities": _as_tensor(observation["abilities"], device=device, dtype=dtype),
        "entities": _as_tensor(observation["entities"], device=device, dtype=dtype),
        "global_state": _as_tensor(observation["global_state"], device=device, dtype=dtype),
        "entity_mask": _as_tensor(observation["entity_mask"], device=device, dtype=torch.bool),
    }
    if "hero_ids" in observation:
        values["hero_ids"] = _as_tensor(observation["hero_ids"], device=device, dtype=torch.long)
    if values["hero"].ndim == 1:
        values["hero"] = values["hero"].unsqueeze(0)
    if values["abilities"].ndim == 2:
        values["abilities"] = values["abilities"].unsqueeze(0)
    if values["entities"].ndim == 2:
        values["entities"] = values["entities"].unsqueeze(0)
    if values["global_state"].ndim == 1:
        values["global_state"] = values["global_state"].unsqueeze(0)
    if values["entity_mask"].ndim == 1:
        values["entity_mask"] = values["entity_mask"].unsqueeze(0)
    if "hero_ids" in values and values["hero_ids"].ndim == 0:
        values["hero_ids"] = values["hero_ids"].reshape(1)
    if values["hero"].shape[0] != 1:
        raise InferenceSchemaError("runtime inference accepts exactly one hero row")
    if not all(torch.isfinite(value).all() for key, value in values.items() if key != "entity_mask" and key != "hero_ids"):
        raise InferenceSchemaError("observation contains non-finite values")
    return values


def _output_tensor(output: Mapping[str, Any], name: str, *aliases: str) -> torch.Tensor:
    source = name if name in output else next((alias for alias in aliases if alias in output), None)
    if source is None:
        raise InferenceSchemaError(f"actor output is missing {name}")
    value = output[source]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if not bool(torch.isfinite(value).all()):
        raise InferenceSchemaError(f"actor output {name} contains non-finite values")
    return value


def _mask_for(
    masks: Mapping[str, Any], name: str, kind: int, size: int,
) -> torch.Tensor:
    if not isinstance(masks, Mapping):
        raise ActionMaskError("action masks must be a mapping")
    present = [key for key in (name, f"{name}_mask") if key in masks]
    if len(present) != 1:
        raise ActionMaskError(f"exactly one {name} mask is required")
    value = masks[present[0]]
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.bool).detach().cpu()
    if tensor.ndim == 0:
        raise ActionMaskError(f"{name} mask must be one- or two-dimensional")
    if tensor.ndim == 1 and tensor.shape[0] == size:
        return tensor
    if tensor.ndim == 2 and tensor.shape[-1] == size:
        if tensor.shape[0] == 1:
            return tensor[0]
        if tensor.shape[0] == ACTION_KINDS:
            return tensor[kind]
    if tensor.ndim == 3 and tuple(tensor.shape[-2:]) == (ACTION_KINDS, size):
        return tensor[0, kind]
    raise ActionMaskError(f"{name} mask has invalid shape {tuple(tensor.shape)}, expected [{size}] or [{ACTION_KINDS}, {size}]")


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor, name: str) -> int:
    vector = logits.detach().reshape(-1).cpu()
    if vector.numel() != mask.numel():
        raise InferenceSchemaError(f"{name} logits and mask sizes differ")
    if not bool(mask.any()):
        raise ActionMaskError(f"{name} mask has no valid action")
    safe = torch.where(mask, vector, torch.full_like(vector, -torch.inf))
    if not bool(torch.isfinite(safe).any()):
        raise InferenceSchemaError(f"{name} logits are invalid under mask")
    return int(torch.argmax(safe).item())


class AI42InferenceAdapter:
    """Deterministic actor-only inference with safe fallback telemetry."""

    def __init__(
        self,
        actor: AI42Actor,
        *,
        expected_manifest: AI42Manifest | Mapping[str, Any] | None = None,
        model_manifest: AI42Manifest | Mapping[str, Any] | None = None,
        state_store: RecurrentStateStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        latency_sample_size: int = 256,
        latency_threshold_seconds: float | None = None,
    ) -> None:
        if getattr(actor, "has_value_head", False):
            raise TypeError("AI-42 runtime accepts actor-only modules")
        self.actor = actor.eval()
        hidden_size = int(getattr(actor, "hidden_size"))
        actor_device, actor_dtype = _actor_device_dtype(actor)
        if state_store is None:
            state_store = RecurrentStateStore(hidden_size, device=actor_device, dtype=actor_dtype)
        if state_store.hidden_size != hidden_size:
            raise ValueError("state store hidden size does not match actor")
        if state_store.slot_count != HERO_COUNT:
            raise ValueError(f"state store must contain exactly {HERO_COUNT} stable hero slots")
        if state_store.device != actor_device:
            raise ValueError("state store device does not match actor")
        if state_store.dtype != actor_dtype:
            raise ValueError("state store dtype does not match actor")
        state_shape = (HERO_COUNT, hidden_size)
        if tuple(state_store._h.shape) != state_shape or tuple(state_store._c.shape) != state_shape:
            raise ValueError(f"state store tensors must have shape {state_shape}")
        if state_store._h.device != actor_device or state_store._c.device != actor_device:
            raise ValueError("state store tensor device does not match actor")
        if state_store._h.dtype != actor_dtype or state_store._c.dtype != actor_dtype:
            raise ValueError("state store tensor dtype does not match actor")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.state_store = state_store
        self._clock = clock
        self.telemetry = InferenceTelemetry(
            latency_sample_size=latency_sample_size,
            latency_threshold_seconds=latency_threshold_seconds,
        )
        self.expected_manifest = _coerce_manifest(expected_manifest) if expected_manifest is not None else None
        self.model_manifest = None
        self._manifest_error: ManifestMismatchError | None = None
        if model_manifest is not None:
            try:
                self.model_manifest = _coerce_manifest(model_manifest)
            except (TypeError, ValueError) as exc:
                self._manifest_error = ManifestMismatchError({"manifest": ("valid", str(exc))})
        if self.expected_manifest is not None and self.model_manifest is not None:
            differences = manifest_mismatches(self.expected_manifest, self.model_manifest)
            if differences:
                self._manifest_error = ManifestMismatchError(differences)

    def reset_match(self) -> None:
        self.state_store.reset_match()

    def on_death(self, slot: int) -> None:
        self.state_store.on_death(slot)

    def on_respawn(self, slot: int) -> None:
        self.state_store.on_respawn(slot)

    def apply_lifecycle(self, slot: int | None, event: str) -> None:
        self.state_store.apply_event(slot, event)

    def snapshot_state(self) -> RecurrentSnapshot:
        return self.state_store.snapshot()

    def restore_state(self, snapshot: RecurrentSnapshot | Mapping[str, Any]) -> None:
        self.state_store.restore(snapshot)

    def _fallback(self, slot: int, reason: str, masks: Mapping[str, Any], entity_mask: torch.Tensor | None) -> InferenceResult:
        def mask_or_none(name: str, kind: int, size: int) -> torch.Tensor | None:
            try:
                return _mask_for(masks, name, kind, size)
            except (ActionMaskError, TypeError, RuntimeError, ValueError, AttributeError):
                return None

        kind_mask = mask_or_none("kind", 0, ACTION_KINDS)
        timing_bins = int(getattr(self.actor, "timing_bins", 0))
        selected = [-1] * 6
        if (
            kind_mask is not None
            and bool(kind_mask.any())
            and entity_mask is not None
            and timing_bins > 0
        ):
            kind = int(torch.where(kind_mask)[0][0].item())
            entity_valid = entity_mask.reshape(-1).detach().cpu().bool()
            masks_by_head = (
                mask_or_none("target", kind, entity_valid.numel()),
                mask_or_none("offset", kind, NAVIGATION_OFFSETS),
                mask_or_none("anchor", kind, NAVIGATION_ANCHORS),
                mask_or_none("timing", kind, timing_bins),
                mask_or_none("timing_aux", kind, timing_bins),
            )
            target_mask = masks_by_head[0]
            if target_mask is not None:
                masks_by_head = (target_mask & entity_valid, *masks_by_head[1:])
            if all(mask is not None and bool(mask.any()) for mask in masks_by_head):
                selected = [
                    kind,
                    *(
                        int(torch.where(mask)[0][0].item())
                        for mask in masks_by_head
                        if mask is not None
                    ),
                ]
        kind, target, offset, anchor, timing, timing_aux = selected
        issued = kind >= 0
        action = {
            "control": CONTROL_ISSUE if issued else -1,
            "control_name": CONTROL_NAMES[CONTROL_ISSUE] if issued else None,
            "kind": kind,
            "target": target,
            "offset": offset,
            "anchor": anchor,
            "timing": timing,
            "timing_aux": timing_aux,
            "valid": kind >= 0,
            "issued": issued,
            "fallback": True,
            "fallback_reason": reason,
        }
        return InferenceResult(MappingProxyType(action), None, True, reason, slot)

    def _infer_once(
        self,
        slot: int,
        observation: Mapping[str, Any],
        action_masks: Mapping[str, Any] | None = None,
    ) -> InferenceResult:
        self.state_store._check_slot(slot)
        masks = action_masks if action_masks is not None else {}
        try:
            values = _batched_observation(observation, self.actor)
            entity_mask = values["entity_mask"]
            if self._manifest_error is not None:
                return self._fallback(slot, "manifest_mismatch", masks, entity_mask)
            entity_slots = int(values["entities"].shape[1])
            timing_bins = int(getattr(self.actor, "timing_bins", 0))
            if timing_bins < 1:
                raise InferenceSchemaError("actor timing_bins must be positive")
            # Validate the complete mask schema before invoking the actor.
            _mask_for(masks, "kind", 0, ACTION_KINDS)
            _mask_for(masks, "target", 0, entity_slots)
            _mask_for(masks, "offset", 0, NAVIGATION_OFFSETS)
            _mask_for(masks, "anchor", 0, NAVIGATION_ANCHORS)
            _mask_for(masks, "timing", 0, timing_bins)
            _mask_for(masks, "timing_aux", 0, timing_bins)
            old_h, old_c = self.state_store.get(slot)
            kwargs = {name: values[name] for name in ("hero", "abilities", "entities", "global_state", "entity_mask")}
            if "hero_ids" in values:
                kwargs["hero_ids"] = values["hero_ids"]
            with torch.no_grad():
                output = self.actor(
                    **kwargs,
                    h=old_h.unsqueeze(0),
                    c=old_c.unsqueeze(0),
                )
            if not isinstance(output, Mapping):
                raise InferenceSchemaError("actor output must be a mapping")
            allowed_output = {
                "control", "kind", "target", "offset", "anchor", "direction", "distance",
                "timing", "timing_aux", "h", "c", "next_h", "next_c",
            }
            unexpected_output = set(output) - allowed_output
            if unexpected_output:
                raise InferenceSchemaError(
                    f"actor output schema contains unexpected fields: {sorted(unexpected_output)}"
                )
            control_logits = _output_tensor(output, "control")
            if tuple(control_logits.shape) != (1, CONTROL_CLASSES):
                raise InferenceSchemaError(
                    f"control logits shape {tuple(control_logits.shape)} is invalid"
                )
            control = int(torch.argmax(control_logits[0]).item())
            next_h = _output_tensor(output, "h", "next_h")
            next_c = _output_tensor(output, "c", "next_c")
            state_shapes = {
                "h": (1, self.state_store.hidden_size),
                "c": (1, self.state_store.hidden_size),
            }
            for name, value in (("h", next_h), ("c", next_c)):
                if tuple(value.shape) != state_shapes[name]:
                    raise InferenceSchemaError(
                        f"{name} logits/state shape {tuple(value.shape)} is invalid"
                    )
            control_name = CONTROL_NAMES[control]
            if control != CONTROL_ISSUE:
                # WAIT/HOLD/CANCEL are valid control decisions, but they do
                # not issue a v13 action and therefore carry no fabricated
                # kind or parameter values.  Recurrent state still advances.
                self.state_store.set(slot, next_h, next_c)
                action = {
                    "control": control,
                    "control_name": control_name,
                    "kind": -1,
                    "target": -1,
                    "offset": -1,
                    "anchor": -1,
                    "timing": -1,
                    "timing_aux": -1,
                    "valid": True,
                    "issued": False,
                    "fallback": False,
                }
                return InferenceResult(MappingProxyType(action), output, False, None, slot)
            kind_logits = _output_tensor(output, "kind")
            target_logits = _output_tensor(output, "target")
            offset_logits = _output_tensor(output, "offset", "direction")
            anchor_logits = _output_tensor(output, "anchor", "distance")
            timing_logits = _output_tensor(output, "timing")
            timing_aux_logits = _output_tensor(output, "timing_aux")
            if tuple(kind_logits.shape) != (1, ACTION_KINDS):
                raise InferenceSchemaError(f"kind logits shape {tuple(kind_logits.shape)} is invalid")
            expected_shapes = {
                "target": (1, ACTION_KINDS, entity_slots),
                "offset": (1, ACTION_KINDS, NAVIGATION_OFFSETS),
                "anchor": (1, ACTION_KINDS, NAVIGATION_ANCHORS),
                "timing": (1, ACTION_KINDS, int(getattr(self.actor, "timing_bins", timing_logits.shape[-1]))),
                "timing_aux": (1, ACTION_KINDS, int(getattr(self.actor, "timing_bins", timing_aux_logits.shape[-1]))),
            }
            for name, value in (("target", target_logits), ("offset", offset_logits), ("anchor", anchor_logits), ("timing", timing_logits), ("timing_aux", timing_aux_logits)):
                if tuple(value.shape) != expected_shapes[name]:
                    raise InferenceSchemaError(f"{name} logits/state shape {tuple(value.shape)} is invalid")
            kind_mask = _mask_for(masks, "kind", 0, ACTION_KINDS)
            entity_valid = entity_mask[0].detach().cpu().bool()
            kind = _masked_argmax(kind_logits[0], kind_mask, "kind")
            target_mask = _mask_for(masks, "target", kind, entity_slots) & entity_valid
            target = _masked_argmax(target_logits[0, kind], target_mask, "target")
            offset = _masked_argmax(offset_logits[0, kind], _mask_for(masks, "offset", kind, NAVIGATION_OFFSETS), "offset")
            anchor = _masked_argmax(anchor_logits[0, kind], _mask_for(masks, "anchor", kind, NAVIGATION_ANCHORS), "anchor")
            timing_size = int(timing_logits.shape[-1])
            timing = _masked_argmax(timing_logits[0, kind], _mask_for(masks, "timing", kind, timing_size), "timing")
            timing_aux_size = int(timing_aux_logits.shape[-1])
            timing_aux = _masked_argmax(timing_aux_logits[0, kind], _mask_for(masks, "timing_aux", kind, timing_aux_size), "timing_aux")
            self.state_store.set(slot, next_h, next_c)
            action = {
                "control": control,
                "control_name": control_name,
                "kind": kind,
                "target": target,
                "offset": offset,
                "anchor": anchor,
                "timing": timing,
                "timing_aux": timing_aux,
                "valid": True,
                "issued": True,
                "fallback": False,
            }
            return InferenceResult(MappingProxyType(action), output, False, None, slot)
        except ManifestMismatchError:
            return self._fallback(slot, "manifest_mismatch", masks, locals().get("entity_mask"))
        except InferenceSchemaError:
            return self._fallback(slot, "schema_mismatch", masks, locals().get("entity_mask"))
        except ActionMaskError:
            return self._fallback(slot, "mask_invalid", masks, locals().get("entity_mask"))
        except Exception:
            return self._fallback(slot, "exception", masks, locals().get("entity_mask"))

    def infer(
        self,
        slot: int,
        observation: Mapping[str, Any],
        action_masks: Mapping[str, Any] | None = None,
    ) -> InferenceResult:
        started = self._clock()
        result = self._infer_once(slot, observation, action_masks)
        elapsed = max(0.0, float(self._clock() - started))
        if result.used_fallback:
            self.telemetry.record_fallback(result.reason or "unknown", elapsed)
        else:
            self.telemetry.record_success(elapsed)
        return result

    predict = infer
    step = infer

    def telemetry_snapshot(self) -> dict[str, Any]:
        return self.telemetry.snapshot()


# Compatibility-oriented names make the boundary discoverable without adding
# another runtime module or changing package exports.
AI42Runtime = AI42InferenceAdapter
AI42InferenceRuntime = AI42InferenceAdapter
HeroRecurrentState = RecurrentStateStore
RecurrentStateManager = RecurrentStateStore
Manifest = AI42Manifest
AI42InferenceManifest = AI42Manifest
AI42RecurrentState = RecurrentStateStore


__all__ = [
    "ACTOR_INPUT_NAMES",
    "ACTOR_OUTPUT_NAMES",
    "AI42InferenceAdapter",
    "AI42InferenceRuntime",
    "AI42Manifest",
    "AI42InferenceManifest",
    "AI42RecurrentState",
    "AI42Runtime",
    "ActionMaskError",
    "CheckpointCompatibilityError",
    "CheckpointCompatibilityReport",
    "HeroLifecycle",
    "InferenceResult",
    "InferenceSchemaError",
    "InferenceTelemetry",
    "LIVE_PROFILE_WIRING",
    "ManifestMismatchError",
    "RecurrentSnapshot",
    "RecurrentStateStore",
    "assert_manifest_compatible",
    "build_manifest",
    "checkpoint_compatibility_report",
    "check_checkpoint_compatibility",
    "hash_bytes",
    "hash_json",
    "load_ai42_checkpoint",
    "load_compatible_checkpoint",
    "manifest_mismatches",
    "RecurrentStateManager",
]
