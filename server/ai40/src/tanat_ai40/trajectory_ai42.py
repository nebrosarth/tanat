"""Immutable AI-42 teacher trajectories, replay checks, and dataset tooling.

The module is intentionally standard-library-only. It does not import the
environment, torch, numpy, or any learner. Observation and mask payloads are
kept external to the durable record; deterministic replay must supply those
payloads (directly or through ``replay_step``), and their hashes are recomputed.
Canonical executed actions are stored and rehashed locally. An integration
that hashes another authoritative action payload declares ``external`` action
hashing and must supply that payload during replay.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import math
import random
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


TRAJECTORY_SCHEMA_VERSION = 2
TRAJECTORY_SCHEMA = (
    "tanat.ai42.teacher_trajectory.v2|"
    "envelope(schema_version,schema_hash,trajectory_id,match_id,hero_id,"
    "expected_ticks,records)|"
    "record(tick,sequence,hero_id,recurrent_parent_id,recurrent_boundary_id,"
    "observation_hash,action_hash,action_hash_source,mask_hash,"
    "original_ai30_intent,projected_neural_action,executed_action,valid,"
    "rejection_reason,outcome,integrity_hash)|"
    "action(kind,target,point,anchor,timing,skill,lineage_id)|"
    "outcome(reward,terminal,winner,hero_alive,team_reward,damage,kills,"
    "deaths,event)|canonical-json-utf8"
)
TRAJECTORY_SCHEMA_HASH = hashlib.sha256(TRAJECTORY_SCHEMA.encode("utf-8")).hexdigest()
TRAJECTORY_SCHEMA_HASH_BYTES = bytes.fromhex(TRAJECTORY_SCHEMA_HASH)
AI42_TRAJECTORY_SCHEMA_VERSION = TRAJECTORY_SCHEMA_VERSION
AI42_TRAJECTORY_SCHEMA_HASH = TRAJECTORY_SCHEMA_HASH
AI42_TRAJECTORY_SCHEMA_HASH_BYTES = TRAJECTORY_SCHEMA_HASH_BYTES
ACTION_HEADS = ("kind", "target", "point", "anchor", "timing", "skill")


class TrajectoryError(ValueError):
    """Base error for malformed trajectory data or invariant violations."""


class TrajectoryDecodeError(TrajectoryError):
    """A serialized trajectory is malformed, incompatible, or non-canonical."""


class ReplayValidationError(TrajectoryError):
    """A trajectory cannot reproduce the expected deterministic replay."""


class MatchValidationError(TrajectoryError):
    """A collection does not contain one complete trajectory per hero slot."""


class ActionKind(str, Enum):
    WAIT = "wait"
    HOLD = "hold"
    CANCEL = "cancel"
    MOVE = "move"
    ATTACK = "attack"
    SKILL = "skill"
    ABILITY = "ability"
    TELEPORT = "teleport"
    ITEM = "item"
    RETREAT = "retreat"
    RECOVER = "recover"
    NOOP = "wait"


class ActionHashSource(str, Enum):
    CANONICAL_EXECUTED = "canonical_executed"
    EXTERNAL = "external"


class RejectionReason(str, Enum):
    NONE = "none"
    MASKED = "masked"
    INVALID = "invalid"
    SERVER_REJECTED = "server_rejected"
    SAFETY = "safety"
    TIMEOUT = "timeout"
    POLICY_ERROR = "policy_error"
    UNKNOWN = "unknown"


_ACTION_FIELDS = frozenset({
    "kind", "target", "point", "anchor", "timing", "skill", "lineage_id",
})
_OUTCOME_FIELDS = frozenset({
    "reward", "terminal", "winner", "hero_alive", "team_reward", "damage",
    "kills", "deaths", "event",
})
_RECORD_FIELDS = frozenset({
    "tick", "sequence", "hero_id", "recurrent_parent_id", "recurrent_boundary_id",
    "observation_hash", "action_hash", "action_hash_source", "mask_hash",
    "original_ai30_intent", "projected_neural_action", "executed_action", "valid",
    "rejection_reason", "outcome", "integrity_hash",
})
_TRAJECTORY_FIELDS = frozenset({
    "schema_version", "schema_hash", "trajectory_id", "match_id", "hero_id",
    "expected_ticks", "records",
})
_NO_LINEAGE_ACTIONS = frozenset({ActionKind.WAIT.value, ActionKind.CANCEL.value})
_MISSING = object()


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise TrajectoryDecodeError(f"{path}: missing field(s): {', '.join(missing)}")
    if extra:
        raise TrajectoryDecodeError(f"{path}: unexpected field(s): {', '.join(extra)}")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrajectoryError(f"{path} must be finite")
    return result


def _scalar(value: Any, path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TrajectoryError(f"{path} must be a string, integer, float, or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise TrajectoryError(f"{path} must be finite")
    return value


def _identifier(value: Any, path: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        suffix = " or null" if allow_none else ""
        raise TrajectoryError(f"{path} must be a non-empty string{suffix}")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrajectoryError(f"{path} must be a non-negative integer")
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot represent non-finite floats")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_hex__": bytes(value).hex()}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _canonical_value({field.name: getattr(value, field.name) for field in fields(value)})
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted-key, UTF-8 canonical JSON bytes."""

    return json.dumps(
        _canonical_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def hash_payload(value: Any) -> str:
    """Hash bytes directly or hash a value's canonical JSON form."""

    payload = bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _hash_string(value: Any, path: str) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).hex()
    if not isinstance(value, str) or len(value) != 64:
        raise TrajectoryError(f"{path} must be a 64-character SHA-256 hex string")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise TrajectoryError(f"{path} must be hexadecimal") from exc
    if len(decoded) != 32 or value.lower() != value:
        raise TrajectoryError(f"{path} must use lower-case SHA-256 hexadecimal")
    return value


def _point(value: Any, path: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise TrajectoryError(f"{path} must be a 2D or 3D point")
    return tuple(_finite(item, f"{path}[{index}]") for index, item in enumerate(value))


def _kind(value: ActionKind | str, path: str) -> str:
    if isinstance(value, ActionKind):
        return value.value
    if not isinstance(value, str) or not value:
        raise TrajectoryError(f"{path} must be a non-empty string")
    return value


def _strict_contiguous(values: Sequence[int], path: str, error_type: type[TrajectoryError]) -> None:
    if len(set(values)) != len(values):
        raise error_type(f"{path} contains duplicate values")
    for index in range(1, len(values)):
        if values[index] != values[index - 1] + 1:
            raise error_type(
                f"{path} is not strictly contiguous at index {index}: "
                f"{values[index - 1]} -> {values[index]}"
            )


@dataclass(frozen=True, slots=True)
class Action:
    """Immutable factorized action with explicit hold/cancel lineage."""

    kind: str | ActionKind
    target: str | int | None = None
    point: tuple[float, ...] | list[float] | None = None
    anchor: str | int | None = None
    timing: str | int | None = None
    skill: str | int | None = None
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _kind(self.kind, "action.kind"))
        object.__setattr__(self, "target", _scalar(self.target, "action.target"))
        object.__setattr__(self, "point", _point(self.point, "action.point"))
        object.__setattr__(self, "anchor", _scalar(self.anchor, "action.anchor"))
        object.__setattr__(self, "timing", _scalar(self.timing, "action.timing"))
        object.__setattr__(self, "skill", _scalar(self.skill, "action.skill"))
        object.__setattr__(self, "lineage_id", _identifier(self.lineage_id, "action.lineage_id", allow_none=True))
        self._validate_control_semantics()

    def _validate_control_semantics(self) -> None:
        irrelevant = {
            "target": self.target, "point": self.point,
            "anchor": self.anchor, "skill": self.skill,
        }
        if self.kind in {ActionKind.WAIT.value, ActionKind.HOLD.value, ActionKind.CANCEL.value}:
            populated = [name for name, value in irrelevant.items() if value is not None]
            if populated:
                raise TrajectoryError(f"{self.kind} action cannot carry {', '.join(populated)}")
        if self.kind == ActionKind.WAIT.value:
            if self.timing is not None or self.lineage_id is not None:
                raise TrajectoryError("wait action cannot carry timing or lineage_id")
        elif self.kind == ActionKind.HOLD.value:
            if self.lineage_id is None:
                raise TrajectoryError("hold action must reference a prior boundary in lineage_id")
        elif self.kind == ActionKind.CANCEL.value:
            if self.timing is not None:
                raise TrajectoryError("cancel action cannot carry timing")
            if self.lineage_id is None:
                raise TrajectoryError("cancel action must reference a prior boundary in lineage_id")
        elif self.lineage_id is not None:
            raise TrajectoryError("lineage_id is reserved for hold/cancel actions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "target": self.target, "point": self.point,
            "anchor": self.anchor, "timing": self.timing, "skill": self.skill,
            "lineage_id": self.lineage_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "action") -> "Action":
        if not isinstance(value, Mapping):
            raise TrajectoryDecodeError(f"{path} must be an object")
        _require_exact_fields(value, _ACTION_FIELDS, path)
        try:
            return cls(**dict(value))
        except TrajectoryError as exc:
            raise TrajectoryDecodeError(f"{path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Outcome:
    reward: float = 0.0
    terminal: bool = False
    winner: str | int | None = None
    hero_alive: bool | None = None
    team_reward: float = 0.0
    damage: float = 0.0
    kills: int = 0
    deaths: int = 0
    event: str | None = None

    def __post_init__(self) -> None:
        for name in ("reward", "team_reward", "damage"):
            object.__setattr__(self, name, _finite(getattr(self, name), f"outcome.{name}"))
        if not isinstance(self.terminal, bool):
            raise TrajectoryError("outcome.terminal must be boolean")
        if self.hero_alive is not None and not isinstance(self.hero_alive, bool):
            raise TrajectoryError("outcome.hero_alive must be boolean or null")
        _scalar(self.winner, "outcome.winner")
        if self.event is not None and not isinstance(self.event, str):
            raise TrajectoryError("outcome.event must be a string or null")
        for name in ("kills", "deaths"):
            _nonnegative_int(getattr(self, name), f"outcome.{name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward, "terminal": self.terminal, "winner": self.winner,
            "hero_alive": self.hero_alive, "team_reward": self.team_reward,
            "damage": self.damage, "kills": self.kills, "deaths": self.deaths,
            "event": self.event,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "outcome") -> "Outcome":
        if not isinstance(value, Mapping):
            raise TrajectoryDecodeError(f"{path} must be an object")
        _require_exact_fields(value, _OUTCOME_FIELDS, path)
        try:
            return cls(**dict(value))
        except TrajectoryError as exc:
            raise TrajectoryDecodeError(f"{path}: {exc}") from exc


def _coerce_action(value: Action | Mapping[str, Any], path: str) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, Mapping):
        return Action.from_dict(value, path=path)
    raise TrajectoryError(f"{path} must be an Action or action mapping")


@dataclass(frozen=True, slots=True)
class TeacherTrajectoryRecord:
    """Exactly one immutable hero decision at one policy tick."""

    tick: int
    sequence: int
    hero_id: str
    recurrent_parent_id: str
    recurrent_boundary_id: str
    observation_hash: str
    action_hash: str
    mask_hash: str
    original_ai30_intent: Action
    projected_neural_action: Action
    executed_action: Action
    valid: bool
    action_hash_source: ActionHashSource | str = ActionHashSource.CANONICAL_EXECUTED
    rejection_reason: RejectionReason | str = RejectionReason.NONE
    outcome: Outcome = Outcome()
    integrity_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick", _nonnegative_int(self.tick, "record.tick"))
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "record.sequence"))
        object.__setattr__(self, "hero_id", _identifier(self.hero_id, "record.hero_id"))
        object.__setattr__(self, "recurrent_parent_id", _identifier(self.recurrent_parent_id, "record.recurrent_parent_id"))
        object.__setattr__(self, "recurrent_boundary_id", _identifier(self.recurrent_boundary_id, "record.recurrent_boundary_id"))
        if self.recurrent_parent_id == self.recurrent_boundary_id:
            raise TrajectoryError("record recurrent parent and boundary identities must differ")
        for name in ("observation_hash", "action_hash", "mask_hash"):
            object.__setattr__(self, name, _hash_string(getattr(self, name), f"record.{name}"))
        for name in ("original_ai30_intent", "projected_neural_action", "executed_action"):
            value = getattr(self, name)
            if not isinstance(value, Action):
                if not isinstance(value, Mapping):
                    raise TrajectoryError(f"record.{name} must be an Action")
                object.__setattr__(self, name, Action.from_dict(value, path=f"record.{name}"))
        source = self.action_hash_source.value if isinstance(self.action_hash_source, ActionHashSource) else self.action_hash_source
        if source not in {item.value for item in ActionHashSource}:
            raise TrajectoryError("record.action_hash_source is invalid")
        object.__setattr__(self, "action_hash_source", source)
        if source == ActionHashSource.CANONICAL_EXECUTED.value:
            if self.action_hash != hash_payload(self.executed_action.to_dict()):
                raise TrajectoryError("record.action_hash does not match canonical executed action")
        if not isinstance(self.valid, bool):
            raise TrajectoryError("record.valid must be boolean")
        reason = self.rejection_reason.value if isinstance(self.rejection_reason, RejectionReason) else self.rejection_reason
        if not isinstance(reason, str) or not reason:
            raise TrajectoryError("record.rejection_reason must be a non-empty string")
        object.__setattr__(self, "rejection_reason", reason)
        if self.valid and reason != RejectionReason.NONE.value:
            raise TrajectoryError("valid record must have rejection_reason='none'")
        if not self.valid and reason == RejectionReason.NONE.value:
            raise TrajectoryError("rejected record must have a non-'none' rejection reason")
        if not isinstance(self.outcome, Outcome):
            if not isinstance(self.outcome, Mapping):
                raise TrajectoryError("record.outcome must be an Outcome")
            object.__setattr__(self, "outcome", Outcome.from_dict(self.outcome, path="record.outcome"))
        computed = self.compute_integrity_hash()
        if self.integrity_hash is None:
            object.__setattr__(self, "integrity_hash", computed)
        else:
            supplied = _hash_string(self.integrity_hash, "record.integrity_hash")
            if supplied != computed:
                raise TrajectoryError("record.integrity_hash does not match record contents")
            object.__setattr__(self, "integrity_hash", supplied)

    @property
    def ai30_intent(self) -> Action:
        return self.original_ai30_intent

    @property
    def projected_action(self) -> Action:
        return self.projected_neural_action

    @property
    def recurrent_state_id(self) -> str:
        return self.recurrent_boundary_id

    def integrity_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick, "sequence": self.sequence, "hero_id": self.hero_id,
            "recurrent_parent_id": self.recurrent_parent_id,
            "recurrent_boundary_id": self.recurrent_boundary_id,
            "observation_hash": self.observation_hash, "action_hash": self.action_hash,
            "action_hash_source": self.action_hash_source, "mask_hash": self.mask_hash,
            "original_ai30_intent": self.original_ai30_intent.to_dict(),
            "projected_neural_action": self.projected_neural_action.to_dict(),
            "executed_action": self.executed_action.to_dict(), "valid": self.valid,
            "rejection_reason": self.rejection_reason, "outcome": self.outcome.to_dict(),
        }

    def compute_integrity_hash(self) -> str:
        return hash_payload(self.integrity_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.integrity_dict()
        result["integrity_hash"] = self.integrity_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "record") -> "TeacherTrajectoryRecord":
        if not isinstance(value, Mapping):
            raise TrajectoryDecodeError(f"{path} must be an object")
        _require_exact_fields(value, _RECORD_FIELDS, path)
        try:
            return cls(
                tick=value["tick"], sequence=value["sequence"], hero_id=value["hero_id"],
                recurrent_parent_id=value["recurrent_parent_id"],
                recurrent_boundary_id=value["recurrent_boundary_id"],
                observation_hash=value["observation_hash"], action_hash=value["action_hash"],
                action_hash_source=value["action_hash_source"], mask_hash=value["mask_hash"],
                original_ai30_intent=Action.from_dict(value["original_ai30_intent"], path=f"{path}.original_ai30_intent"),
                projected_neural_action=Action.from_dict(value["projected_neural_action"], path=f"{path}.projected_neural_action"),
                executed_action=Action.from_dict(value["executed_action"], path=f"{path}.executed_action"),
                valid=value["valid"], rejection_reason=value["rejection_reason"],
                outcome=Outcome.from_dict(value["outcome"], path=f"{path}.outcome"),
                integrity_hash=value["integrity_hash"],
            )
        except TrajectoryError as exc:
            if isinstance(exc, TrajectoryDecodeError):
                raise
            raise TrajectoryDecodeError(f"{path}: {exc}") from exc
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            raise TrajectoryDecodeError(f"{path} cannot be canonicalized: {exc}") from exc

    @classmethod
    def from_payload(
        cls, *, tick: int, sequence: int, hero_id: str,
        recurrent_parent_id: str, recurrent_boundary_id: str,
        observation: Any, mask: Any,
        original_ai30_intent: Action | Mapping[str, Any],
        projected_neural_action: Action | Mapping[str, Any],
        executed_action: Action | Mapping[str, Any], valid: bool,
        rejection_reason: RejectionReason | str = RejectionReason.NONE,
        outcome: Outcome | Mapping[str, Any] = Outcome(), action: Any = _MISSING,
    ) -> "TeacherTrajectoryRecord":
        """Create hashes; passing ``action`` declares an external action payload."""

        ai30 = _coerce_action(original_ai30_intent, "original_ai30_intent")
        projected = _coerce_action(projected_neural_action, "projected_neural_action")
        executed = _coerce_action(executed_action, "executed_action")
        result = outcome if isinstance(outcome, Outcome) else Outcome.from_dict(outcome)
        external = action is not _MISSING
        return cls(
            tick=tick, sequence=sequence, hero_id=hero_id,
            recurrent_parent_id=recurrent_parent_id,
            recurrent_boundary_id=recurrent_boundary_id,
            observation_hash=hash_payload(observation),
            action_hash=hash_payload(action if external else executed.to_dict()),
            action_hash_source=ActionHashSource.EXTERNAL if external else ActionHashSource.CANONICAL_EXECUTED,
            mask_hash=hash_payload(mask), original_ai30_intent=ai30,
            projected_neural_action=projected, executed_action=executed, valid=valid,
            rejection_reason=rejection_reason, outcome=result,
        )


def _validate_record_structure(
    records: Sequence[TeacherTrajectoryRecord], *, hero_id: str | None,
    expected_ticks: Sequence[int] | None, expected_sequence_start: int | None,
    error_type: type[TrajectoryError],
) -> tuple[int, ...]:
    for index, record in enumerate(records):
        if not isinstance(record, TeacherTrajectoryRecord):
            raise error_type(f"records[{index}] is not a TeacherTrajectoryRecord")
        if hero_id is not None and record.hero_id != hero_id:
            raise error_type(f"records[{index}].hero_id={record.hero_id!r} does not match {hero_id!r}")
        for action_name in (
            "original_ai30_intent", "projected_neural_action", "executed_action",
        ):
            try:
                getattr(record, action_name)._validate_control_semantics()
            except TrajectoryError as exc:
                raise error_type(f"records[{index}].{action_name}: {exc}") from exc
    ticks = tuple(record.tick for record in records)
    sequences = tuple(record.sequence for record in records)
    _strict_contiguous(ticks, "policy ticks", error_type)
    _strict_contiguous(sequences, "record sequence", error_type)
    if expected_sequence_start is not None and sequences and sequences[0] != expected_sequence_start:
        raise error_type(f"record sequence starts at {sequences[0]}, expected {expected_sequence_start}")
    if expected_ticks is not None:
        expected = tuple(expected_ticks)
        _strict_contiguous(expected, "expected_ticks", error_type)
        if ticks != expected:
            raise error_type(f"record ticks {ticks!r} do not equal expected_ticks {expected!r}")
    boundaries: dict[str, int] = {}
    lineage_roots: dict[str, str] = {}
    cancelled_roots: set[str] = set()
    for index, record in enumerate(records):
        if record.recurrent_boundary_id in boundaries:
            raise error_type(f"duplicate recurrent boundary {record.recurrent_boundary_id!r} at records[{index}]")
        boundaries[record.recurrent_boundary_id] = index
        if index and record.recurrent_parent_id != records[index - 1].recurrent_boundary_id:
            raise error_type(f"records[{index}].recurrent_parent_id does not match prior boundary")
        # HOLD/CANCEL describe the AI-30 teacher order state. Projected and
        # executed external commands use the v13 wire vocabulary and cannot
        # carry those control transitions.
        action = record.original_ai30_intent
        if action.kind == ActionKind.HOLD.value:
            if index == 0 or action.lineage_id != records[index - 1].recurrent_boundary_id:
                raise error_type("hold action must reference the immediately prior executed boundary")
            if records[index - 1].original_ai30_intent.kind in _NO_LINEAGE_ACTIONS:
                raise error_type("hold action references a non-holdable prior action")
            lineage_roots[record.recurrent_boundary_id] = lineage_roots[action.lineage_id]
        elif action.kind == ActionKind.CANCEL.value:
            lineage_id = action.lineage_id or ""
            referenced = boundaries.get(lineage_id)
            if referenced is None or referenced >= index:
                raise error_type("cancel action must reference a prior executed boundary")
            if records[referenced].original_ai30_intent.kind in _NO_LINEAGE_ACTIONS:
                raise error_type("cancel action references a non-cancellable prior action")
            root = lineage_roots[lineage_id]
            if root in cancelled_roots:
                raise error_type("cancel action references an already cancelled prior action")
            cancelled_roots.add(root)
            lineage_roots[record.recurrent_boundary_id] = record.recurrent_boundary_id
        else:
            lineage_roots[record.recurrent_boundary_id] = record.recurrent_boundary_id
    return ticks


@dataclass(frozen=True, slots=True)
class TeacherTrajectory:
    """A complete, contiguous trajectory for one hero in one match."""

    trajectory_id: str
    match_id: str
    hero_id: str
    records: tuple[TeacherTrajectoryRecord, ...] | Sequence[TeacherTrajectoryRecord]
    expected_ticks: tuple[int, ...] | Sequence[int] | None = None
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    schema_hash: str = TRAJECTORY_SCHEMA_HASH

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_id", _identifier(self.trajectory_id, "trajectory_id"))
        object.__setattr__(self, "match_id", _identifier(self.match_id, "match_id"))
        object.__setattr__(self, "hero_id", _identifier(self.hero_id, "hero_id"))
        if isinstance(self.schema_version, bool) or self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise TrajectoryError(f"schema_version mismatch: expected {TRAJECTORY_SCHEMA_VERSION}, got {self.schema_version!r}")
        if self.schema_hash != TRAJECTORY_SCHEMA_HASH:
            raise TrajectoryError("schema_hash mismatch")
        if not isinstance(self.records, (list, tuple)):
            raise TrajectoryError("trajectory.records must be a sequence")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        ticks: tuple[int, ...] | None = None
        if self.expected_ticks is not None:
            ticks = tuple(_nonnegative_int(tick, "trajectory.expected_ticks[]") for tick in self.expected_ticks)
            object.__setattr__(self, "expected_ticks", ticks)
        _validate_record_structure(
            records, hero_id=self.hero_id, expected_ticks=ticks,
            expected_sequence_start=None, error_type=TrajectoryError,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "schema_hash": self.schema_hash,
            "trajectory_id": self.trajectory_id, "match_id": self.match_id,
            "hero_id": self.hero_id, "expected_ticks": self.expected_ticks,
            "records": [record.to_dict() for record in self.records],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherTrajectory":
        if not isinstance(value, Mapping):
            raise TrajectoryDecodeError("trajectory must be an object")
        _require_exact_fields(value, _TRAJECTORY_FIELDS, "trajectory")
        if value["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
            raise TrajectoryDecodeError(f"schema_version mismatch: expected {TRAJECTORY_SCHEMA_VERSION}, got {value['schema_version']!r}")
        if value["schema_hash"] != TRAJECTORY_SCHEMA_HASH:
            raise TrajectoryDecodeError("schema_hash mismatch")
        expected_ticks = value["expected_ticks"]
        if expected_ticks is not None and not isinstance(expected_ticks, list):
            raise TrajectoryDecodeError("trajectory.expected_ticks must be an array or null")
        if not isinstance(value["records"], list):
            raise TrajectoryDecodeError("trajectory.records must be an array")
        try:
            return cls(
                trajectory_id=value["trajectory_id"], match_id=value["match_id"],
                hero_id=value["hero_id"],
                records=tuple(
                    TeacherTrajectoryRecord.from_dict(record, path=f"trajectory.records[{index}]")
                    for index, record in enumerate(value["records"])
                ),
                expected_ticks=None if expected_ticks is None else tuple(expected_ticks),
                schema_version=value["schema_version"], schema_hash=value["schema_hash"],
            )
        except TrajectoryError as exc:
            if isinstance(exc, TrajectoryDecodeError):
                raise
            raise TrajectoryDecodeError(str(exc)) from exc
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            raise TrajectoryDecodeError(f"trajectory cannot be canonicalized: {exc}") from exc

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview | str) -> "TeacherTrajectory":
        try:
            raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        except (TypeError, UnicodeEncodeError) as exc:
            raise TrajectoryDecodeError("trajectory payload must be UTF-8 bytes or string") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrajectoryDecodeError("trajectory payload is not UTF-8") from exc
        try:
            decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
            value, end = decoder.raw_decode(text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise TrajectoryDecodeError(f"invalid trajectory JSON: {exc}") from exc
        if end != len(text):
            raise TrajectoryDecodeError(f"trailing bytes at offset {end}")
        try:
            trajectory = cls.from_dict(value)
        except TrajectoryDecodeError:
            raise
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            raise TrajectoryDecodeError(f"trajectory cannot be canonicalized: {exc}") from exc
        try:
            canonical = trajectory.to_bytes()
        except (UnicodeEncodeError, TypeError, ValueError) as exc:
            raise TrajectoryDecodeError(f"trajectory cannot be canonicalized: {exc}") from exc
        if canonical != raw:
            raise TrajectoryDecodeError("trajectory encoding is not canonical JSON")
        return trajectory


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryDecodeError(f"duplicate field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise TrajectoryDecodeError(f"non-finite JSON constant is not allowed: {value}")


def encode_trajectory(trajectory: TeacherTrajectory) -> bytes:
    if not isinstance(trajectory, TeacherTrajectory):
        raise TypeError("encode_trajectory expects a TeacherTrajectory")
    return trajectory.to_bytes()


def decode_trajectory(payload: bytes | bytearray | memoryview | str) -> TeacherTrajectory:
    return TeacherTrajectory.from_bytes(payload)


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """External deterministic step data used to verify omitted payloads."""

    observation: Any
    mask: Any
    action_payload: Any = _MISSING
    executed_action: Action | Mapping[str, Any] | None = None
    outcome: Outcome | Mapping[str, Any] | None = None
    recurrent_parent_id: str | None = None
    recurrent_boundary_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayValidationResult:
    ok: bool
    hero_id: str
    record_count: int
    first_tick: int | None
    last_tick: int | None

    def __bool__(self) -> bool:
        return self.ok


def _payload_lookup(values: Mapping[int, Any] | Sequence[Any] | None, ticks: tuple[int, ...], name: str) -> dict[int, Any] | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        if set(values) != set(ticks):
            raise ReplayValidationError(f"{name} must contain exactly the expected ticks")
        return dict(values)
    if isinstance(values, (str, bytes, bytearray, memoryview)) or len(values) != len(ticks):
        raise ReplayValidationError(f"{name} must contain one payload per expected tick")
    return dict(zip(ticks, values))


def _hash_lookup(values: Mapping[int, str] | Sequence[str] | None, ticks: tuple[int, ...], name: str) -> dict[int, str] | None:
    result = _payload_lookup(values, ticks, name)
    if result is None:
        return None
    try:
        return {tick: _hash_string(value, f"{name}[{tick}]") for tick, value in result.items()}
    except TrajectoryError as exc:
        raise ReplayValidationError(str(exc)) from exc


def _step_from_value(value: ReplayStep | Mapping[str, Any], tick: int) -> ReplayStep:
    if isinstance(value, ReplayStep):
        return value
    if not isinstance(value, Mapping):
        raise ReplayValidationError(f"replay_step at tick {tick} must return ReplayStep or mapping")
    missing = {"observation", "mask"} - set(value)
    if missing:
        raise ReplayValidationError(f"replay_step at tick {tick} is missing {', '.join(sorted(missing))}")
    allowed = {
        "observation", "mask", "action_payload", "executed_action", "outcome",
        "recurrent_parent_id", "recurrent_boundary_id",
    }
    extra = set(value) - allowed
    if extra:
        raise ReplayValidationError(f"replay_step at tick {tick} has unexpected fields {', '.join(sorted(extra))}")
    return ReplayStep(**dict(value))


def validate_integrity(
    trajectory_or_records: TeacherTrajectory | Sequence[TeacherTrajectoryRecord], *,
    expected_ticks: Iterable[int] | None = None, hero_id: str | None = None,
    expected_sequence_start: int | None = None,
) -> ReplayValidationResult:
    """Validate only self-contained structure, lineage, and stored hashes.

    This is not deterministic replay: a coherently modified record can be
    rehashed and remain internally valid. ``validate_replay`` additionally
    requires independent authoritative outputs and is the strict gate.
    """

    if isinstance(trajectory_or_records, TeacherTrajectory):
        records = trajectory_or_records.records
        expected_ticks = trajectory_or_records.expected_ticks if expected_ticks is None else expected_ticks
        hero_id = trajectory_or_records.hero_id if hero_id is None else hero_id
    else:
        records = tuple(trajectory_or_records)
    if hero_id is None:
        hero_id = records[0].hero_id if records else ""
    if records and (not isinstance(hero_id, str) or not hero_id):
        raise ReplayValidationError("hero_id must be a non-empty string")
    expected: tuple[int, ...] | None = None
    if expected_ticks is not None:
        try:
            expected = tuple(_nonnegative_int(tick, "expected_ticks[]") for tick in expected_ticks)
        except TrajectoryError as exc:
            raise ReplayValidationError(str(exc)) from exc
    ticks = _validate_record_structure(
        records, hero_id=hero_id, expected_ticks=expected,
        expected_sequence_start=expected_sequence_start,
        error_type=ReplayValidationError,
    )
    for record in records:
        if record.compute_integrity_hash() != record.integrity_hash:
            raise ReplayValidationError(f"record integrity hash mismatch at tick {record.tick}")
        if record.action_hash_source == ActionHashSource.CANONICAL_EXECUTED.value:
            if hash_payload(record.executed_action.to_dict()) != record.action_hash:
                raise ReplayValidationError(f"canonical action hash mismatch at tick {record.tick}")
    return ReplayValidationResult(
        True, hero_id, len(records),
        ticks[0] if ticks else None, ticks[-1] if ticks else None,
    )


def validate_replay(
    trajectory_or_records: TeacherTrajectory | Sequence[TeacherTrajectoryRecord], *,
    expected_ticks: Iterable[int] | None = None, hero_id: str | None = None,
    expected_sequence_start: int | None = None,
    observation_payloads: Mapping[int, Any] | Sequence[Any] | None = None,
    mask_payloads: Mapping[int, Any] | Sequence[Any] | None = None,
    action_payloads: Mapping[int, Any] | Sequence[Any] | None = None,
    executed_actions: Mapping[int, Action | Mapping[str, Any]] | Sequence[Action | Mapping[str, Any]] | None = None,
    outcomes: Mapping[int, Outcome | Mapping[str, Any]] | Sequence[Outcome | Mapping[str, Any]] | None = None,
    recurrent_parent_ids: Mapping[int, str] | Sequence[str] | None = None,
    recurrent_boundary_ids: Mapping[int, str] | Sequence[str] | None = None,
    replay_step: Callable[[TeacherTrajectoryRecord], ReplayStep | Mapping[str, Any]] | None = None,
    observation_hashes: Mapping[int, str] | Sequence[str] | None = None,
    action_hashes: Mapping[int, str] | Sequence[str] | None = None,
    mask_hashes: Mapping[int, str] | Sequence[str] | None = None,
) -> ReplayValidationResult:
    """Verify structure, lineage, integrity, and all replay payload hashes.

    Observation, mask, executed action, outcome, and recurrent transition
    outputs must all come from an independent source. Supply indexed
    collections or a callback that advances an external deterministic
    environment. Hash-only collections are optional additional expectations.
    """

    integrity = validate_integrity(
        trajectory_or_records, expected_ticks=expected_ticks, hero_id=hero_id,
        expected_sequence_start=expected_sequence_start,
    )
    records = (
        trajectory_or_records.records
        if isinstance(trajectory_or_records, TeacherTrajectory)
        else tuple(trajectory_or_records)
    )
    ticks = tuple(record.tick for record in records)
    hero_id = integrity.hero_id
    if replay_step is None and any(value is None for value in (
        observation_payloads, mask_payloads, executed_actions, outcomes,
        recurrent_parent_ids, recurrent_boundary_ids,
    )):
        raise ReplayValidationError(
            "strict replay requires observation_payloads, mask_payloads, "
            "executed_actions, outcomes, recurrent_parent_ids, and "
            "recurrent_boundary_ids when replay_step is absent"
        )
    observations = _payload_lookup(observation_payloads, ticks, "observation_payloads")
    masks = _payload_lookup(mask_payloads, ticks, "mask_payloads")
    actions = _payload_lookup(action_payloads, ticks, "action_payloads")
    expected_executed = _payload_lookup(executed_actions, ticks, "executed_actions")
    expected_outcomes = _payload_lookup(outcomes, ticks, "outcomes")
    expected_parents = _payload_lookup(recurrent_parent_ids, ticks, "recurrent_parent_ids")
    expected_boundaries = _payload_lookup(recurrent_boundary_ids, ticks, "recurrent_boundary_ids")
    obs_hashes = _hash_lookup(observation_hashes, ticks, "observation_hashes")
    act_hashes = _hash_lookup(action_hashes, ticks, "action_hashes")
    msk_hashes = _hash_lookup(mask_hashes, ticks, "mask_hashes")
    for record in records:
        tick = record.tick
        step = _step_from_value(replay_step(record), tick) if replay_step is not None else None
        observation = step.observation if step is not None else observations[tick]
        mask = step.mask if step is not None else masks[tick]
        if hash_payload(observation) != record.observation_hash:
            raise ReplayValidationError(f"observation hash mismatch at tick {tick}")
        if hash_payload(mask) != record.mask_hash:
            raise ReplayValidationError(f"mask hash mismatch at tick {tick}")
        external_action = step.action_payload if step is not None else (actions[tick] if actions is not None else _MISSING)
        if record.action_hash_source == ActionHashSource.EXTERNAL.value:
            if external_action is _MISSING:
                raise ReplayValidationError(f"external action payload is required at tick {tick}")
            if hash_payload(external_action) != record.action_hash:
                raise ReplayValidationError(f"external action hash mismatch at tick {tick}")
        if obs_hashes is not None and obs_hashes[tick] != record.observation_hash:
            raise ReplayValidationError(f"expected observation hash mismatch at tick {tick}")
        if act_hashes is not None and act_hashes[tick] != record.action_hash:
            raise ReplayValidationError(f"expected action hash mismatch at tick {tick}")
        if msk_hashes is not None and msk_hashes[tick] != record.mask_hash:
            raise ReplayValidationError(f"expected mask hash mismatch at tick {tick}")
        supplied_action = step.executed_action if step is not None else expected_executed[tick]
        supplied_outcome = step.outcome if step is not None else expected_outcomes[tick]
        supplied_parent = step.recurrent_parent_id if step is not None else expected_parents[tick]
        supplied_boundary = step.recurrent_boundary_id if step is not None else expected_boundaries[tick]
        if any(value is None for value in (
            supplied_action, supplied_outcome, supplied_parent, supplied_boundary,
        )):
            raise ReplayValidationError(
                f"replay_step must supply executed_action, outcome, recurrent_parent_id, "
                f"and recurrent_boundary_id at tick {tick}"
            )
        try:
            replayed_action = _coerce_action(supplied_action, f"executed_actions[{tick}]")
        except TrajectoryError as exc:
            raise ReplayValidationError(str(exc)) from exc
        if replayed_action != record.executed_action:
            raise ReplayValidationError(f"executed action mismatch at tick {tick}")
        try:
            actual_outcome = supplied_outcome if isinstance(supplied_outcome, Outcome) else Outcome.from_dict(supplied_outcome)
        except TrajectoryError as exc:
            raise ReplayValidationError(str(exc)) from exc
        if actual_outcome != record.outcome:
            raise ReplayValidationError(f"outcome mismatch at tick {tick}")
        if supplied_parent != record.recurrent_parent_id:
            raise ReplayValidationError(f"recurrent parent mismatch at tick {tick}")
        if supplied_boundary != record.recurrent_boundary_id:
            raise ReplayValidationError(f"recurrent boundary mismatch at tick {tick}")
    return ReplayValidationResult(True, hero_id, len(records), ticks[0] if ticks else None, ticks[-1] if ticks else None)


validate_trajectory_replay = validate_replay


@dataclass(frozen=True, slots=True)
class MatchValidationResult:
    ok: bool
    match_id: str
    hero_ids: tuple[str, ...]
    tick_count: int
    first_tick: int | None
    last_tick: int | None

    def __bool__(self) -> bool:
        return self.ok


def validate_match_trajectories(
    trajectories: Sequence[TeacherTrajectory], *,
    expected_hero_ids: Iterable[str] | None = None, heroes_per_team: int = 5,
    expected_match_id: str | None = None, expected_ticks: Iterable[int] | None = None,
) -> MatchValidationResult:
    """Require one equally covered trajectory for every expected hero slot."""

    if isinstance(heroes_per_team, bool) or not isinstance(heroes_per_team, int) or heroes_per_team < 1:
        raise MatchValidationError("heroes_per_team must be a positive integer")
    rows = tuple(trajectories)
    for index, trajectory in enumerate(rows):
        if not isinstance(trajectory, TeacherTrajectory):
            raise MatchValidationError(f"trajectories[{index}] is not a TeacherTrajectory")
    expected_count = heroes_per_team * 2
    if expected_hero_ids is None:
        expected_ids = None
    else:
        expected_ids = tuple(expected_hero_ids)
        if any(not isinstance(hero, str) or not hero for hero in expected_ids):
            raise MatchValidationError("expected_hero_ids must contain non-empty strings")
        if len(set(expected_ids)) != len(expected_ids):
            raise MatchValidationError("expected_hero_ids contains duplicates")
        if len(expected_ids) != expected_count:
            raise MatchValidationError(
                f"expected_hero_ids must contain exactly {expected_count} hero slots"
            )
    if len(rows) != expected_count:
        raise MatchValidationError(f"expected {expected_count} hero trajectories, got {len(rows)}")
    match_ids = {trajectory.match_id for trajectory in rows}
    if len(match_ids) != 1:
        raise MatchValidationError("all hero trajectories must have the same match_id")
    match_id = next(iter(match_ids)) if match_ids else (expected_match_id or "")
    if expected_match_id is not None and match_id != expected_match_id:
        raise MatchValidationError(f"match_id {match_id!r} does not match {expected_match_id!r}")
    hero_ids = tuple(trajectory.hero_id for trajectory in rows)
    if len(set(hero_ids)) != len(hero_ids):
        raise MatchValidationError("hero trajectories contain duplicate hero_id values")
    if expected_ids is not None and set(hero_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(hero_ids))
        extra = sorted(set(hero_ids) - set(expected_ids))
        raise MatchValidationError(f"hero slot mismatch: missing={missing}, extra={extra}")
    trajectory_ids = [trajectory.trajectory_id for trajectory in rows]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise MatchValidationError("trajectory_id values must be unique within a match")
    if expected_ticks is None:
        explicit_ticks = None
    else:
        try:
            explicit_ticks = tuple(
                _nonnegative_int(tick, "expected_ticks[]") for tick in expected_ticks
            )
        except TrajectoryError as exc:
            raise MatchValidationError(str(exc)) from exc
        _strict_contiguous(explicit_ticks, "expected_ticks", MatchValidationError)
    coverage: tuple[int, ...] = ()
    for index, trajectory in enumerate(rows):
        ticks = _validate_record_structure(
            trajectory.records, hero_id=trajectory.hero_id,
            expected_ticks=explicit_ticks, expected_sequence_start=None,
            error_type=MatchValidationError,
        )
        if index == 0:
            coverage = ticks
        elif ticks != coverage:
            raise MatchValidationError("all hero trajectories must have identical tick coverage")
    return MatchValidationResult(
        True, match_id, tuple(sorted(hero_ids)), len(coverage),
        coverage[0] if coverage else None, coverage[-1] if coverage else None,
    )


@dataclass(frozen=True, slots=True)
class TeacherMatch:
    """Immutable match-level collection validated at construction."""

    match_id: str
    trajectories: tuple[TeacherTrajectory, ...] | Sequence[TeacherTrajectory]
    expected_hero_ids: tuple[str, ...] | Sequence[str] | None = None
    heroes_per_team: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _identifier(self.match_id, "match_id"))
        rows = tuple(self.trajectories)
        object.__setattr__(self, "trajectories", rows)
        expected = None if self.expected_hero_ids is None else tuple(self.expected_hero_ids)
        object.__setattr__(self, "expected_hero_ids", expected)
        validate_match_trajectories(
            rows, expected_hero_ids=expected, heroes_per_team=self.heroes_per_team,
            expected_match_id=self.match_id,
        )


def _trajectory_groups(dataset: Any) -> tuple[tuple[TeacherTrajectoryRecord, ...], ...]:
    if isinstance(dataset, TeacherTrajectory):
        return (dataset.records,)
    if isinstance(dataset, TeacherTrajectoryRecord):
        return ((dataset,),)
    groups: list[tuple[TeacherTrajectoryRecord, ...]] = []
    pending: list[TeacherTrajectoryRecord] = []
    for index, item in enumerate(dataset):
        if isinstance(item, TeacherTrajectory):
            if pending:
                groups.append(tuple(pending))
                pending.clear()
            groups.append(item.records)
        elif isinstance(item, TeacherTrajectoryRecord):
            pending.append(item)
        else:
            raise TypeError(f"dataset[{index}] is not a trajectory or trajectory record")
    if pending:
        groups.append(tuple(pending))
    return tuple(groups)


def _label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return canonical_json_bytes(value).decode("utf-8")


def _freeze_counts(counter: Counter[str]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(counter.items())))


def _balance(counts: Mapping[str, int]) -> Mapping[str, float]:
    total = sum(counts.values())
    return MappingProxyType({} if not total else {key: counts[key] / total for key in counts})


@dataclass(frozen=True, slots=True)
class HeadMetric:
    count: int
    correct: int
    accuracy: float | None
    loss_total: float | None
    loss_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count, "correct": self.correct, "accuracy": self.accuracy,
            "loss_total": self.loss_total, "loss_mean": self.loss_mean,
        }


@dataclass(frozen=True, slots=True)
class OutcomeTotals:
    reward_total: float
    team_reward_total: float
    damage_total: float
    kills_total: int
    deaths_total: int
    terminal_count: int
    winner_counts: Mapping[str, int]

    @property
    def return_total(self) -> float:
        return self.reward_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_total": self.reward_total, "return_total": self.return_total,
            "team_reward_total": self.team_reward_total, "damage_total": self.damage_total,
            "kills_total": self.kills_total, "deaths_total": self.deaths_total,
            "terminal_count": self.terminal_count, "winner_counts": dict(self.winner_counts),
        }


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    total_records: int
    kind_counts: Mapping[str, int]
    target_counts: Mapping[str, int]
    point_counts: Mapping[str, int]
    anchor_counts: Mapping[str, int]
    timing_counts: Mapping[str, int]
    skill_counts: Mapping[str, int]
    wait_hold_cancel_counts: Mapping[str, int]
    validity_counts: Mapping[str, int]
    rejection_reason_counts: Mapping[str, int]
    balances: Mapping[str, Mapping[str, float]]
    outcome_totals: OutcomeTotals
    recurrent_boundary_count: int
    recurrent_boundary_unique: bool
    duplicate_recurrent_boundaries: int
    recurrent_lineage_failures: int
    hash_integrity_failures: Mapping[str, int]
    head_metrics: Mapping[str, HeadMetric]
    per_skill_metrics: Mapping[str, HeadMetric]

    @property
    def by_kind(self) -> Mapping[str, int]:
        return self.kind_counts

    @property
    def by_target(self) -> Mapping[str, int]:
        return self.target_counts

    @property
    def by_point(self) -> Mapping[str, int]:
        return self.point_counts

    @property
    def by_anchor(self) -> Mapping[str, int]:
        return self.anchor_counts

    @property
    def by_timing(self) -> Mapping[str, int]:
        return self.timing_counts

    @property
    def by_skill(self) -> Mapping[str, int]:
        return self.skill_counts

    @property
    def by_validity(self) -> Mapping[str, int]:
        return self.validity_counts

    def to_dict(self) -> dict[str, Any]:
        count_names = (
            "kind_counts", "target_counts", "point_counts", "anchor_counts",
            "timing_counts", "skill_counts", "wait_hold_cancel_counts",
            "validity_counts", "rejection_reason_counts",
        )
        result: dict[str, Any] = {"total_records": self.total_records}
        result.update({name: dict(getattr(self, name)) for name in count_names})
        result["balances"] = {name: dict(value) for name, value in self.balances.items()}
        result["outcome_totals"] = self.outcome_totals.to_dict()
        result.update({
            "recurrent_boundary_count": self.recurrent_boundary_count,
            "recurrent_boundary_unique": self.recurrent_boundary_unique,
            "duplicate_recurrent_boundaries": self.duplicate_recurrent_boundaries,
            "recurrent_lineage_failures": self.recurrent_lineage_failures,
            "hash_integrity_failures": dict(self.hash_integrity_failures),
            "head_metrics": {name: metric.to_dict() for name, metric in self.head_metrics.items()},
            "per_skill_metrics": {
                name: metric.to_dict() for name, metric in self.per_skill_metrics.items()
            },
        })
        return result


def _action_source(record: TeacherTrajectoryRecord, source: str) -> Action:
    choices = {
        "original": record.original_ai30_intent,
        "projected": record.projected_neural_action,
        "executed": record.executed_action,
    }
    if source not in choices:
        raise ValueError("action source must be original, projected, or executed")
    return choices[source]


def _audit_actions(
    records: tuple[TeacherTrajectoryRecord, ...],
    supplied: Sequence[Action | Mapping[str, Any]] | None,
    source: str, name: str,
) -> tuple[Action, ...]:
    if supplied is None:
        return tuple(_action_source(record, source) for record in records)
    if len(supplied) != len(records):
        raise ValueError(f"{name} must contain one action per record")
    return tuple(_coerce_action(value, f"{name}[{index}]") for index, value in enumerate(supplied))


def _head_value(action: Action, head: str) -> Any:
    value = getattr(action, head)
    if head == "skill" and value is None and action.kind.startswith("skill_"):
        return action.kind
    return value


def _head_applicable(action: Action, head: str) -> bool:
    return head == "kind" or _head_value(action, head) is not None


def _loss_metric(
    value: float | Sequence[float] | None,
    record_count: int,
    head: str,
    applicable_indices: Sequence[int],
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if not applicable_indices:
        return None, None
    if isinstance(value, bool):
        raise ValueError(f"head_losses[{head!r}] must be numeric or a numeric sequence")
    if isinstance(value, (int, float)):
        loss = _finite(value, f"head_losses[{head!r}]")
        return loss, loss
    if len(value) != record_count:
        raise ValueError(f"head_losses[{head!r}] must contain one value per record")
    losses = [
        _finite(value[index], f"head_losses[{head!r}][{index}]")
        for index in applicable_indices
    ]
    total = sum(losses)
    return total, (total / len(losses) if losses else None)


def audit_dataset(
    dataset: Any, *, action: str = "executed",
    prediction_source: str = "projected", label_source: str = "original",
    prediction_actions: Sequence[Action | Mapping[str, Any]] | None = None,
    label_actions: Sequence[Action | Mapping[str, Any]] | None = None,
    head_losses: Mapping[
        str, float | Sequence[float] | Mapping[str, float]
    ] | None = None,
    payload_provider: Callable[[TeacherTrajectoryRecord], ReplayStep | Mapping[str, Any]] | None = None,
) -> DatasetAudit:
    """Audit class balance, outcomes, integrity, lineage, and head metrics."""

    groups = _trajectory_groups(dataset)
    records = tuple(record for group in groups for record in group)
    counters = {name: Counter() for name in (
        "kind", "target", "point", "anchor", "timing", "skill",
        "wait_hold_cancel", "validity", "rejection_reason",
    )}
    winners: Counter[str] = Counter()
    reward_total = team_reward_total = damage_total = 0.0
    kills_total = deaths_total = terminal_count = 0
    hash_failures: Counter[str] = Counter()
    for record in records:
        selected = _action_source(record, action)
        counters["kind"][selected.kind] += 1
        for head in ("target", "point", "anchor", "timing"):
            counters[head][_label(getattr(selected, head))] += 1
        skill = selected.skill
        if skill is None and selected.kind.startswith("skill_"):
            skill = selected.kind
        counters["skill"][_label(skill)] += 1
        if selected.kind in {ActionKind.WAIT.value, ActionKind.HOLD.value, ActionKind.CANCEL.value}:
            counters["wait_hold_cancel"][selected.kind] += 1
        counters["validity"]["valid" if record.valid else "rejected"] += 1
        counters["rejection_reason"][record.rejection_reason] += 1
        reward_total += record.outcome.reward
        team_reward_total += record.outcome.team_reward
        damage_total += record.outcome.damage
        kills_total += record.outcome.kills
        deaths_total += record.outcome.deaths
        terminal_count += int(record.outcome.terminal)
        winners[_label(record.outcome.winner)] += 1
        if record.compute_integrity_hash() != record.integrity_hash:
            hash_failures["record"] += 1
        if record.action_hash_source == ActionHashSource.CANONICAL_EXECUTED.value:
            if hash_payload(record.executed_action.to_dict()) != record.action_hash:
                hash_failures["action"] += 1
        else:
            hash_failures["action_unchecked"] += 1
        if payload_provider is None:
            hash_failures["observation_unchecked"] += 1
            hash_failures["mask_unchecked"] += 1
        else:
            try:
                step = _step_from_value(payload_provider(record), record.tick)
                if hash_payload(step.observation) != record.observation_hash:
                    hash_failures["observation"] += 1
                if hash_payload(step.mask) != record.mask_hash:
                    hash_failures["mask"] += 1
                if record.action_hash_source == ActionHashSource.EXTERNAL.value:
                    hash_failures["action_unchecked"] -= 1
                    if step.action_payload is _MISSING or hash_payload(step.action_payload) != record.action_hash:
                        hash_failures["action"] += 1
            except (TrajectoryError, TypeError, ValueError):
                hash_failures["payload_provider"] += 1
    for kind in (ActionKind.WAIT.value, ActionKind.HOLD.value, ActionKind.CANCEL.value):
        counters["wait_hold_cancel"].setdefault(kind, 0)
    for status in ("valid", "rejected"):
        counters["validity"].setdefault(status, 0)

    duplicate_boundaries = lineage_failures = boundary_count = 0
    for group in groups:
        seen: set[str] = set()
        for index, record in enumerate(group):
            boundary_count += 1
            if record.recurrent_boundary_id in seen:
                duplicate_boundaries += 1
            seen.add(record.recurrent_boundary_id)
            if index and record.recurrent_parent_id != group[index - 1].recurrent_boundary_id:
                lineage_failures += 1

    predictions = _audit_actions(records, prediction_actions, prediction_source, "prediction_actions")
    labels = _audit_actions(records, label_actions, label_source, "label_actions")
    losses = head_losses or {}
    unknown_losses = set(losses) - set(ACTION_HEADS)
    if unknown_losses:
        raise ValueError(f"head_losses has unknown heads: {sorted(unknown_losses)}")
    skill_groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if _head_applicable(label, "skill"):
            skill_groups.setdefault(_label(_head_value(label, "skill")), []).append(index)
    raw_skill_loss = losses.get("skill")
    skill_loss_totals: dict[str, float] | None = None
    if isinstance(raw_skill_loss, Mapping):
        skill_loss_totals = {}
        for raw_skill, raw_loss in raw_skill_loss.items():
            skill = _label(raw_skill)
            if skill in skill_loss_totals:
                raise ValueError(f"head_losses['skill'] has duplicate normalized key {skill!r}")
            skill_loss_totals[skill] = _finite(
                raw_loss, f"head_losses['skill'][{skill!r}]",
            )
        unknown_skills = set(skill_loss_totals) - set(skill_groups)
        if unknown_skills:
            raise ValueError(
                f"head_losses['skill'] has inactive skills: {sorted(unknown_skills)}"
            )
    metrics: dict[str, HeadMetric] = {}
    for head in ACTION_HEADS:
        applicable = [
            index for index, label in enumerate(labels)
            if _head_applicable(label, head)
        ]
        correct = sum(
            _head_value(predictions[index], head) == _head_value(labels[index], head)
            for index in applicable
        )
        if head == "skill" and skill_loss_totals is not None:
            if set(skill_loss_totals) == set(skill_groups):
                loss_total = sum(skill_loss_totals.values())
                loss_mean = loss_total / len(applicable) if applicable else None
            else:
                loss_total = loss_mean = None
        else:
            loss_total, loss_mean = _loss_metric(
                losses.get(head), len(records), head, applicable,
            )
        metrics[head] = HeadMetric(
            len(applicable), correct,
            (correct / len(applicable) if applicable else None),
            loss_total, loss_mean,
        )

    per_skill: dict[str, HeadMetric] = {}
    for skill, applicable in sorted(skill_groups.items()):
        correct = sum(
            _head_value(predictions[index], "skill") == _head_value(labels[index], "skill")
            for index in applicable
        )
        if skill_loss_totals is not None:
            loss_total = skill_loss_totals.get(skill)
            loss_mean = (
                loss_total / len(applicable) if loss_total is not None else None
            )
        elif isinstance(raw_skill_loss, (int, float)):
            # A scalar is already an aggregate skill-head loss. Copying it to
            # every skill would multiply the reported total.
            loss_total = loss_mean = None
        else:
            loss_total, loss_mean = _loss_metric(
                raw_skill_loss, len(records), "skill", applicable,
            )
        per_skill[skill] = HeadMetric(
            len(applicable), correct, correct / len(applicable), loss_total, loss_mean,
        )

    named_counts = {
        "kind_counts": _freeze_counts(counters["kind"]),
        "target_counts": _freeze_counts(counters["target"]),
        "point_counts": _freeze_counts(counters["point"]),
        "anchor_counts": _freeze_counts(counters["anchor"]),
        "timing_counts": _freeze_counts(counters["timing"]),
        "skill_counts": _freeze_counts(counters["skill"]),
        "wait_hold_cancel_counts": _freeze_counts(counters["wait_hold_cancel"]),
        "validity_counts": _freeze_counts(counters["validity"]),
        "rejection_reason_counts": _freeze_counts(counters["rejection_reason"]),
    }
    balances = MappingProxyType({name.removesuffix("_counts"): _balance(counts) for name, counts in named_counts.items()})
    outcomes = OutcomeTotals(
        reward_total, team_reward_total, damage_total, kills_total, deaths_total,
        terminal_count, _freeze_counts(winners),
    )
    return DatasetAudit(
        total_records=len(records), balances=balances, outcome_totals=outcomes,
        recurrent_boundary_count=boundary_count,
        recurrent_boundary_unique=duplicate_boundaries == 0,
        duplicate_recurrent_boundaries=duplicate_boundaries,
        recurrent_lineage_failures=lineage_failures,
        hash_integrity_failures=_freeze_counts(hash_failures),
        head_metrics=MappingProxyType(metrics),
        per_skill_metrics=MappingProxyType(per_skill),
        **named_counts,
    )


audit_trajectory_dataset = audit_dataset


@dataclass(frozen=True, slots=True)
class StratifiedSample:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_group_ids: tuple[str, ...] = ()
    validation_group_ids: tuple[str, ...] = ()

    @property
    def training_indices(self) -> tuple[int, ...]:
        return self.train_indices

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return self.validation_indices


def _item_stratum(item: Any, stratify_by: str | Callable[[Any], Any] | None) -> str:
    if stratify_by is not None:
        if callable(stratify_by):
            return _label(stratify_by(item))
        if isinstance(item, Mapping):
            if stratify_by not in item:
                raise ValueError(f"stratify field {stratify_by!r} is missing")
            return _label(item[stratify_by])
        if not hasattr(item, stratify_by):
            raise ValueError(f"stratify field {stratify_by!r} is missing")
        return _label(getattr(item, stratify_by))
    if isinstance(item, TeacherTrajectoryRecord):
        return item.executed_action.kind
    if isinstance(item, Mapping) and "kind" in item:
        return _label(item["kind"])
    return _label(item)


def _automatic_validation_groups(
    groups: Mapping[str, list[int]], strata: Mapping[int, str],
    fraction: float, rng: random.Random,
) -> set[str]:
    if not fraction or len(groups) < 2:
        return set()
    dominant: dict[str, str] = {}
    for group, indices in groups.items():
        counts = Counter(strata[index] for index in indices)
        maximum = max(counts.values())
        dominant[group] = min(label for label, count in counts.items() if count == maximum)
    by_class: dict[str, list[str]] = {}
    for group, label in dominant.items():
        by_class.setdefault(label, []).append(group)
    selected: set[str] = set()
    for label in sorted(by_class):
        candidates = sorted(by_class[label])
        rng.shuffle(candidates)
        if len(candidates) > 1:
            count = min(len(candidates) - 1, max(1, int(len(candidates) * fraction)))
            selected.update(candidates[:count])
    desired = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    if len(selected) < desired:
        remaining = sorted(set(groups) - selected)
        rng.shuffle(remaining)
        selected.update(remaining[:desired - len(selected)])
    return selected


def generate_stratified_sample_indices(
    items: Sequence[Any], *, validation_indices: Iterable[int] | None = None,
    validation_fraction: float = 0.2, samples_per_class: int | None = None,
    seed: int = 0, stratify_by: str | Callable[[Any], Any] | None = None,
    group_ids: Sequence[Any] | None = None,
) -> StratifiedSample:
    """Generate deterministic class-aware indices with optional group isolation."""

    if not isinstance(items, Sequence):
        raise TypeError("items must be a sequence with stable integer indices")
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if samples_per_class is not None and (
        isinstance(samples_per_class, bool) or not isinstance(samples_per_class, int) or samples_per_class < 1
    ):
        raise ValueError("samples_per_class must be a positive integer or null")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    size = len(items)
    strata = {index: _item_stratum(item, stratify_by) for index, item in enumerate(items)}
    class_groups: dict[str, list[int]] = {}
    for index, label in strata.items():
        class_groups.setdefault(label, []).append(index)
    normalized_group_ids: tuple[str, ...] | None = None
    grouped: dict[str, list[int]] = {}
    if group_ids is not None:
        if len(group_ids) != size:
            raise ValueError("group_ids must contain one group per item")
        normalized_group_ids = tuple(_label(group) for group in group_ids)
        for index, group in enumerate(normalized_group_ids):
            grouped.setdefault(group, []).append(index)
    rng = random.Random(seed)
    if validation_indices is None:
        if normalized_group_ids is None:
            validation_set: set[int] = set()
            for label in sorted(class_groups):
                candidates = class_groups[label][:]
                rng.shuffle(candidates)
                if len(candidates) > 1 and validation_fraction:
                    count = min(len(candidates) - 1, max(1, int(len(candidates) * validation_fraction)))
                    validation_set.update(candidates[:count])
        else:
            selected_groups = _automatic_validation_groups(grouped, strata, validation_fraction, rng)
            validation_set = {index for group in selected_groups for index in grouped[group]}
    else:
        values = list(validation_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in values):
            raise ValueError("validation_indices must contain integers")
        if len(set(values)) != len(values):
            raise ValueError("validation_indices must not contain duplicates")
        if any(index < 0 or index >= size for index in values):
            raise ValueError("validation_indices contains an out-of-range index")
        validation_set = set(values)
        if normalized_group_ids is not None:
            for group, indices in grouped.items():
                selected = validation_set.intersection(indices)
                if selected and len(selected) != len(indices):
                    raise ValueError(f"validation_indices splits group {group!r}")
    selected: list[int] = []
    for label in sorted(class_groups):
        available = [index for index in class_groups[label] if index not in validation_set]
        rng.shuffle(available)
        if samples_per_class is None:
            selected.extend(available)
        elif available:
            selected.extend(available[offset % len(available)] for offset in range(samples_per_class))
    rng.shuffle(selected)
    if set(selected) & validation_set:
        raise RuntimeError("stratified sampler leaked a validation index")
    train_groups = validation_groups = ()
    if normalized_group_ids is not None:
        train_groups = tuple(sorted({normalized_group_ids[index] for index in selected}))
        validation_groups = tuple(sorted({normalized_group_ids[index] for index in validation_set}))
        if set(train_groups) & set(validation_groups):
            raise RuntimeError("stratified sampler split a group")
    return StratifiedSample(tuple(selected), tuple(sorted(validation_set)), train_groups, validation_groups)


def generate_group_stratified_sample_indices(
    trajectories: Sequence[TeacherTrajectory], *, group_by: str = "match_id",
    validation_fraction: float = 0.2, samples_per_class: int | None = None,
    seed: int = 0,
) -> StratifiedSample:
    """Split flattened records while keeping each match or trajectory intact."""

    if group_by not in {"match_id", "trajectory_id"}:
        raise ValueError("group_by must be match_id or trajectory_id")
    records: list[TeacherTrajectoryRecord] = []
    groups: list[str] = []
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, TeacherTrajectory):
            raise TypeError(f"trajectories[{index}] is not a TeacherTrajectory")
        group = getattr(trajectory, group_by)
        records.extend(trajectory.records)
        groups.extend([group] * len(trajectory.records))
    return generate_stratified_sample_indices(
        records, validation_fraction=validation_fraction,
        samples_per_class=samples_per_class, seed=seed, group_ids=groups,
    )


stratified_sampler_indices = generate_stratified_sample_indices
make_stratified_sample_indices = generate_stratified_sample_indices


__all__ = [
    "ACTION_HEADS", "AI42_TRAJECTORY_SCHEMA_HASH",
    "AI42_TRAJECTORY_SCHEMA_HASH_BYTES", "AI42_TRAJECTORY_SCHEMA_VERSION",
    "Action", "ActionHashSource", "ActionKind", "DatasetAudit", "HeadMetric",
    "MatchValidationError", "MatchValidationResult", "Outcome", "OutcomeTotals",
    "ReplayStep", "ReplayValidationError", "ReplayValidationResult",
    "RejectionReason", "StratifiedSample", "TeacherMatch", "TeacherTrajectory",
    "TeacherTrajectoryRecord", "TRAJECTORY_SCHEMA", "TRAJECTORY_SCHEMA_HASH",
    "TRAJECTORY_SCHEMA_HASH_BYTES", "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryDecodeError", "TrajectoryError", "audit_dataset",
    "audit_trajectory_dataset", "canonical_json_bytes", "decode_trajectory",
    "encode_trajectory", "generate_group_stratified_sample_indices",
    "generate_stratified_sample_indices", "hash_payload",
    "make_stratified_sample_indices", "stratified_sampler_indices",
    "validate_integrity", "validate_match_trajectories", "validate_replay",
    "validate_trajectory_replay",
]
