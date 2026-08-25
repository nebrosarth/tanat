"""Durable, torch-free AI-42 behavior-cloning dataset artifacts.

The v13 bridge owns the meaning of a trajectory.  This module owns the
durable representation: complete ten-slot policy ticks in deterministic NPZ
shards and a canonical, hashed JSON index.  It deliberately does not import
the runtime or a learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np

from .bridge_ai42 import (
    AI42BridgeOutput,
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
from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_DTYPE,
    ACTION_KINDS,
    AI42_PROTOCOL_VERSION,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
    MAX_ENTITIES,
    StepResult,
)
from .trajectory_ai42 import (
    AI42_TRAJECTORY_SCHEMA_HASH,
    Action,
    ActionKind,
    canonical_json_bytes,
    hash_payload,
)


DATASET_SCHEMA_VERSION = "AI42-dataset-v1"
SHARD_SCHEMA_VERSION = "AI42-npz-shard-v2"
MANIFEST_FILENAME = "manifest.json"
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".npz"

_FLOAT_DTYPE = np.dtype("<f4")
_MASK_DTYPE = np.dtype("u1")
_ACTION_DTYPE = np.dtype(ACTION_DTYPE)
_INT_DTYPE = np.dtype("<i4")
_STEP_DTYPE = np.dtype("<u4")
_ARRAY_DTYPES: dict[str, np.dtype] = {
    "hero": _FLOAT_DTYPE,
    "abilities": _FLOAT_DTYPE,
    "entities": _FLOAT_DTYPE,
    "global": _FLOAT_DTYPE,
    "entity_mask": _MASK_DTYPE,
    "kind_mask": _MASK_DTYPE,
    "target_mask": _MASK_DTYPE,
    "skill_target_mask": _MASK_DTYPE,
    "teacher_status": _MASK_DTYPE,
    "teacher_action": _ACTION_DTYPE,
    "projected_action": _ACTION_DTYPE,
    "executed_action": _ACTION_DTYPE,
    "executed_valid": _MASK_DTYPE,
    "rejection_reason": _MASK_DTYPE,
    "rewards": _FLOAT_DTYPE,
    "done": _MASK_DTYPE,
    "winner": _INT_DTYPE,
    "step": _STEP_DTYPE,
    "elapsed": _FLOAT_DTYPE,
    "invalid": _MASK_DTYPE,
}
_ARRAY_NAMES = tuple(_ARRAY_DTYPES)
_FLOAT_ARRAY_NAMES = frozenset({"hero", "abilities", "entities", "global", "rewards", "elapsed"})
_MASK_ARRAY_NAMES = frozenset(
    {"entity_mask", "kind_mask", "target_mask", "skill_target_mask", "done", "invalid"}
)
_ACTION_ARRAY_NAMES = frozenset({"teacher_action", "projected_action", "executed_action"})
_REJECTION_CODES = {
    "none": REJECTION_REASON_NONE,
    "masked": REJECTION_REASON_MASKED,
    "invalid": REJECTION_REASON_INVALID,
    "server_rejected": REJECTION_REASON_SERVER_REJECTED,
    "safety": REJECTION_REASON_SAFETY,
    "timeout": REJECTION_REASON_TIMEOUT,
    "policy_error": REJECTION_REASON_POLICY_ERROR,
    "unknown": REJECTION_REASON_UNKNOWN,
}
_REJECTION_NAMES = {value: key for key, value in _REJECTION_CODES.items()}
_TEACHER_STATUSES = frozenset(
    {
        TEACHER_STATUS_NONE,
        TEACHER_STATUS_ACTION,
        TEACHER_STATUS_WAIT,
        TEACHER_STATUS_HOLD,
        TEACHER_STATUS_CANCEL,
        TEACHER_STATUS_UNAVAILABLE,
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "dataset_schema_version",
        "shard_schema_version",
        "protocol_version",
        "schema_hash",
        "reward_hash",
        "trajectory_schema_hash",
        "runtime_manifest_hash",
        "runtime_manifest",
        "split_seed",
        "validation_fraction",
        "matches",
        "shards",
        "manifest_hash",
    }
)
_MATCH_FIELDS = frozenset(
    {
        "match_id",
        "split",
        "shard",
        "row_offset",
        "tick_count",
        "ticks",
        "hero_ids",
        "trajectory_ids",
        "trajectory_hashes",
        "recurrent_parent_ids",
        "recurrent_boundary_ids",
        "seed",
        "scenario",
        "controller_by_slot",
        "roster_ids",
        "side_by_slot",
    }
)
_SHARD_FIELDS = frozenset(
    {"name", "sha256", "match_ids", "row_count", "raw_bytes", "stored_bytes", "compression"}
)


class AI42DatasetError(ValueError):
    """Raised when a dataset capture or durable artifact is invalid."""


DatasetError = AI42DatasetError


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_hex(value: bytes | bytearray | memoryview | str, name: str) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) != 32:
            raise AI42DatasetError(f"{name} must contain exactly 32 bytes")
        return raw.hex()
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise AI42DatasetError(f"{name} must be lower-case SHA-256 hexadecimal")
    try:
        if len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as exc:
        raise AI42DatasetError(f"{name} must be lower-case SHA-256 hexadecimal") from exc
    return value


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping) or not value:
        raise AI42DatasetError(f"{name} must be a non-empty mapping")
    if any(not isinstance(key, str) for key in value):
        raise AI42DatasetError(f"{name} keys must be strings")
    return value


def _runtime_manifest_hash(value: Mapping[str, Any]) -> str:
    try:
        return hash_payload(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AI42DatasetError(f"runtime_manifest is not canonicalizable: {exc}") from exc


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AI42DatasetError(f"{path} field set mismatch: missing={missing}, extra={extra}")


def _strict_json(payload: bytes, path: str) -> Any:
    try:
        text = payload.decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
        value, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AI42DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if end != len(text):
        raise AI42DatasetError(f"{path} has trailing bytes at offset {end}")
    try:
        if canonical_json_bytes(value) != payload:
            raise AI42DatasetError(f"{path} is not canonical JSON")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AI42DatasetError(f"{path} cannot be canonicalized: {exc}") from exc
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AI42DatasetError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise AI42DatasetError(f"non-finite JSON constant is not allowed: {value}")


def _copy_exact(value: Any, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise AI42DatasetError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.shape != shape:
        raise AI42DatasetError(f"{name} must have shape {shape}, got {array.shape}")
    if name in _FLOAT_ARRAY_NAMES and not np.isfinite(array).all():
        raise AI42DatasetError(f"{name} contains non-finite values")
    return np.array(array, dtype=dtype, copy=True)


def _validate_mask(array: np.ndarray, name: str) -> None:
    if name in _MASK_ARRAY_NAMES or name in {"teacher_status", "executed_valid", "rejection_reason"}:
        if np.any((array != 0) & (array != 1)) and name in _MASK_ARRAY_NAMES | {"executed_valid"}:
            raise AI42DatasetError(f"{name} must contain only zero/one values")


def _validate_action_array(array: np.ndarray, name: str) -> None:
    if array.dtype != _ACTION_DTYPE:
        raise AI42DatasetError(f"{name} must have dtype {_ACTION_DTYPE}")
    if np.any(array["kind"] >= ACTION_KINDS):
        raise AI42DatasetError(f"{name}.kind contains an out-of-range value")
    # The structured dtype already bounds the wire integer fields.  Keep the
    # explicit range checks for readability and for non-native NumPy views.
    if np.any(array["direction"] >= 81) or np.any(array["distance"] >= 15):
        raise AI42DatasetError(f"{name} navigation fields are outside v13 ranges")
    skill = (array["kind"] >= 3) & (array["kind"] <= 6)
    if np.any(skill & (array["distance"] != 0)):
        raise AI42DatasetError(f"{name} skill actions cannot use navigation anchors in v13")


def _action_to_wire(action: Action, name: str = "action") -> tuple[int, int, int, int]:
    if not isinstance(action, Action):
        raise AI42DatasetError(f"{name} must be an Action")
    kind = action.kind
    if kind in {ActionKind.WAIT.value, ActionKind.HOLD.value, ActionKind.CANCEL.value, "unavailable"}:
        return 0, 0, 0, 0
    target = action.target if isinstance(action.target, int) and not isinstance(action.target, bool) else 0
    direction = 0
    distance = 0
    if kind == ActionKind.MOVE.value:
        wire_kind = 1
    elif kind == ActionKind.ATTACK.value:
        wire_kind = 2
    elif kind == ActionKind.SKILL.value:
        if not isinstance(action.skill, int) or not 1 <= action.skill <= 4:
            raise AI42DatasetError(f"{name}.skill cannot be represented by v13")
        wire_kind = action.skill + 2
    elif kind == ActionKind.TELEPORT.value:
        wire_kind = 7
    else:
        raise AI42DatasetError(f"{name}.kind={kind!r} cannot be represented by v13")
    if action.point is not None:
        if len(action.point) != 2 or any(not math.isfinite(float(item)) for item in action.point):
            raise AI42DatasetError(f"{name}.point cannot be represented by v13")
        coords = tuple(float(item) for item in action.point)
        if any(item != int(item) or not -4 <= int(item) <= 4 for item in coords):
            raise AI42DatasetError(f"{name}.point is outside v13 offset81")
        direction = (int(coords[1]) + 4) * 9 + int(coords[0]) + 4
    elif action.anchor is not None:
        if not isinstance(action.anchor, int) or not 0 <= action.anchor < 15:
            raise AI42DatasetError(f"{name}.anchor is outside v13 anchor15")
        distance = action.anchor
    if not 0 <= int(target) <= 0xFFFF:
        raise AI42DatasetError(f"{name}.target is outside v13 uint16")
    return int(wire_kind), int(target), int(direction), int(distance)


def _wire_row(value: Any, name: str) -> np.void:
    if isinstance(value, np.void) and value.dtype == _ACTION_DTYPE:
        return value
    if isinstance(value, Mapping):
        fields = ("kind", "target", "direction", "distance")
        if set(value) != set(fields):
            raise AI42DatasetError(f"{name} must contain exactly {fields}")
        raw = tuple(value[field] for field in fields)
    elif all(hasattr(value, field) for field in ("kind", "target", "direction", "distance")):
        raw = (value.kind, value.target, value.direction, value.distance)
    else:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise AI42DatasetError(f"{name} is not a v13 wire action") from exc
    if len(raw) != 4 or any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise AI42DatasetError(f"{name} must contain four integer fields")
    try:
        ints = tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AI42DatasetError(f"{name} must contain four integer fields") from exc
    if any(ints[index] != raw[index] for index in range(4)):
        raise AI42DatasetError(f"{name} must contain integral fields")
    row = np.zeros((), dtype=_ACTION_DTYPE)
    row["kind"], row["target"], row["direction"], row["distance"] = ints
    return row[()]


def _wire_actions(values: Any, name: str, ticks: int) -> np.ndarray:
    array = np.zeros((ticks, HERO_COUNT), dtype=_ACTION_DTYPE)
    rows = tuple(values)
    if len(rows) != ticks:
        raise AI42DatasetError(f"{name} must contain exactly {ticks} ticks")
    for tick, row_values in enumerate(rows):
        row_values = tuple(row_values)
        if len(row_values) != HERO_COUNT:
            raise AI42DatasetError(f"{name}[{tick}] must contain exactly {HERO_COUNT} slots")
        for hero, value in enumerate(row_values):
            array[tick, hero] = _wire_row(value, f"{name}[{tick}][{hero}]")
    _validate_action_array(array, name)
    return array


def _wire_actions_fast(values: Any, name: str, ticks: int) -> np.ndarray:
    """Vectorized v13 action normalization for the production collector."""

    array = np.asarray(values)
    if array.dtype == _ACTION_DTYPE and array.shape == (ticks, HERO_COUNT):
        result = np.array(array, copy=True)
    elif array.ndim == 3 and array.shape == (ticks, HERO_COUNT, 4):
        if array.dtype.kind not in "iu" or np.any(array < 0):
            raise AI42DatasetError(f"{name} must contain four non-negative integer fields")
        result = np.zeros((ticks, HERO_COUNT), dtype=_ACTION_DTYPE)
        result["kind"] = array[:, :, 0]
        result["target"] = array[:, :, 1]
        result["direction"] = array[:, :, 2]
        result["distance"] = array[:, :, 3]
    else:
        return _wire_actions(values, name, ticks)
    _validate_action_array(result, name)
    return result


def _array_from_actions(actions: Sequence[Sequence[Action]], name: str) -> np.ndarray:
    array = np.zeros((len(actions), HERO_COUNT), dtype=_ACTION_DTYPE)
    for tick, row in enumerate(actions):
        if len(row) != HERO_COUNT:
            raise AI42DatasetError(f"{name}[{tick}] must contain exactly {HERO_COUNT} slots")
        for hero, action in enumerate(row):
            array[tick, hero] = _action_to_wire(action, f"{name}[{tick}][{hero}]")
    return array


def _derived_status(action: Action) -> int:
    return {
        "unavailable": TEACHER_STATUS_UNAVAILABLE,
        ActionKind.WAIT.value: TEACHER_STATUS_WAIT,
        ActionKind.HOLD.value: TEACHER_STATUS_HOLD,
        ActionKind.CANCEL.value: TEACHER_STATUS_CANCEL,
    }.get(action.kind, TEACHER_STATUS_ACTION)


def _metadata_seed(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AI42DatasetError("match seed must be an integer")
    return int(value)


def _metadata_roster(value: Any, hero_ids: Sequence[str]) -> tuple[Any, ...]:
    if value is None:
        return tuple(hero_ids)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise AI42DatasetError("match roster_ids must contain ten slots") from exc
    if len(values) != HERO_COUNT or len(set(values)) != HERO_COUNT:
        raise AI42DatasetError("match roster_ids must contain ten unique slots")
    normalized: list[Any] = []
    for index, item in enumerate(values):
        if isinstance(item, (bool, np.bool_)):
            raise AI42DatasetError(f"match roster_ids[{index}] must be a string or integer")
        if isinstance(item, np.integer):
            item = int(item)
        if not isinstance(item, (str, int)) or (isinstance(item, str) and not item):
            raise AI42DatasetError(f"match roster_ids[{index}] must be a string or integer")
        normalized.append(item)
    return tuple(normalized)


def _metadata_scenario(value: Any) -> str:
    if value is None:
        return "unspecified"
    if not isinstance(value, str) or not value:
        raise AI42DatasetError("match scenario must be a non-empty string")
    return value


def _metadata_controllers(value: Any) -> tuple[int, ...]:
    if value is None:
        return (2,) * HERO_COUNT
    try:
        values = tuple(value)
    except TypeError as exc:
        raise AI42DatasetError("match controller_by_slot must contain ten slots") from exc
    try:
        valid = len(values) == HERO_COUNT and all(
            not isinstance(item, (bool, np.bool_)) and int(item) == item and 0 <= int(item) <= 3
            for item in values
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise AI42DatasetError("match controller_by_slot must contain ten values in [0, 3]")
    return tuple(int(item) for item in values)


def _metadata_sides(value: Any) -> tuple[int, ...]:
    if value is None:
        return (0,) * (HERO_COUNT // 2) + (1,) * (HERO_COUNT // 2)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise AI42DatasetError("match side_by_slot must contain ten slots") from exc
    try:
        valid = len(values) == HERO_COUNT and all(
            not isinstance(item, (bool, np.bool_)) and int(item) == item and int(item) in (0, 1)
            for item in values
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise AI42DatasetError("match side_by_slot must contain ten binary slots")
    sides = tuple(int(item) for item in values)
    if sides.count(0) != HERO_COUNT // 2 or sides.count(1) != HERO_COUNT // 2:
        raise AI42DatasetError("match side_by_slot must contain five slots per side")
    return sides


def _prepared_from_match(match: "AI42DatasetMatch") -> "_PreparedMatch":
    if match.raw_capture is not None:
        return _prepared_from_raw_match(match, match.raw_capture)
    output = match.output
    if not isinstance(output, AI42BridgeOutput):
        raise AI42DatasetError("match.output must be an AI42BridgeOutput")
    try:
        # Publishing a training artifact is the trust boundary. Strict replay
        # is mandatory and also checks each trajectory's integrity before
        # comparing it with the immutable authoritative evidence captured from
        # the v13 StepResult.
        output.validate_replay(
            observation_payloads=output.observation_payloads,
            mask_payloads=output.mask_payloads,
            action_payloads=output.action_payloads,
            executed_actions=output.executed_actions,
            outcomes=output.outcomes,
            recurrent_parent_ids=output.recurrent_parent_ids,
            recurrent_boundary_ids=output.recurrent_boundary_ids,
        )
    except Exception as exc:
        raise AI42DatasetError(f"trajectory integrity/replay validation failed: {exc}") from exc
    if output.protocol_version != AI42_PROTOCOL_VERSION:
        raise AI42DatasetError("match protocol version is not v13")
    expected_hashes = {
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
    }
    for field, expected in expected_hashes.items():
        if getattr(output, field) != expected:
            raise AI42DatasetError(f"match {field} does not match the active AI42 contract")
    ticks = len(output.observation_payloads)
    if ticks < 1:
        raise AI42DatasetError("match must contain at least one policy tick")
    if len(output.mask_payloads) != ticks or len(output.action_payloads) != ticks:
        raise AI42DatasetError("match has missing or extra tick payloads")
    trajectories = output.trajectories
    if len(trajectories) != HERO_COUNT:
        raise AI42DatasetError("match must contain exactly ten hero trajectories")
    hero_ids = tuple(trajectory.hero_id for trajectory in trajectories)
    if len(set(hero_ids)) != HERO_COUNT:
        raise AI42DatasetError("match hero IDs must be unique")
    steps = tuple(record.tick for record in trajectories[0].records)
    if len(steps) != ticks or any(steps[index] != steps[0] + index for index in range(ticks)):
        raise AI42DatasetError("match ticks must be contiguous")
    for trajectory in trajectories:
        if tuple(record.tick for record in trajectory.records) != steps:
            raise AI42DatasetError("all ten hero slots must cover the same contiguous ticks")

    def payload(name: str, shape: tuple[int, ...], dtype: np.dtype, values: Any) -> np.ndarray:
        # Bridge observations are canonical JSON values, so their NumPy dtype
        # is intentionally restored at this durable boundary. Captures from
        # StepResult use the strict path below and cannot be silently cast.
        array = np.asarray(values)
        if array.shape != shape:
            raise AI42DatasetError(f"{output.match.match_id}.{name} must have shape {shape}, got {array.shape}")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise AI42DatasetError(f"{output.match.match_id}.{name} contains non-finite values")
        return np.asarray(array, dtype=dtype).copy()

    arrays: dict[str, np.ndarray] = {}
    arrays["hero"] = payload("hero", (ticks, HERO_COUNT, HERO_FEATURES), _FLOAT_DTYPE, [
        [output.observation_payloads[tick][hero]["hero"] for hero in range(HERO_COUNT)]
        for tick in range(ticks)
    ])
    arrays["abilities"] = payload("abilities", (ticks, HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES), _FLOAT_DTYPE, [
        [output.observation_payloads[tick][hero]["abilities"] for hero in range(HERO_COUNT)]
        for tick in range(ticks)
    ])
    arrays["entities"] = payload("entities", (ticks, HERO_COUNT, MAX_ENTITIES, ENTITY_FEATURES), _FLOAT_DTYPE, [
        [output.observation_payloads[tick][hero]["entities"] for hero in range(HERO_COUNT)]
        for tick in range(ticks)
    ])
    arrays["global"] = payload("global", (ticks, HERO_COUNT, GLOBAL_FEATURES), _FLOAT_DTYPE, [
        [output.observation_payloads[tick][hero]["global"] for hero in range(HERO_COUNT)]
        for tick in range(ticks)
    ])
    for name, shape in (
        ("entity_mask", (ticks, HERO_COUNT, MAX_ENTITIES)),
        ("kind_mask", (ticks, HERO_COUNT, ACTION_KINDS)),
        ("target_mask", (ticks, HERO_COUNT, MAX_ENTITIES)),
        ("skill_target_mask", (ticks, HERO_COUNT, ABILITY_COUNT, MAX_ENTITIES)),
    ):
        arrays[name] = payload(name, shape, _MASK_DTYPE, [
            [output.mask_payloads[tick][hero][name] for hero in range(HERO_COUNT)]
            for tick in range(ticks)
        ])
        _validate_mask(arrays[name], name)

    records = [[trajectory.records[tick] for hero, trajectory in enumerate(trajectories)] for tick in range(ticks)]
    derived_status = np.asarray(
        [[_derived_status(record.original_ai30_intent) for record in row] for row in records], dtype=_MASK_DTYPE,
    )
    derived_teacher = _array_from_actions(
        [[record.original_ai30_intent for record in row] for row in records], "teacher_action",
    )
    projected_rows = []
    for tick, row in enumerate(output.action_payloads):
        actions = []
        if len(row) != HERO_COUNT:
            raise AI42DatasetError(f"projected_action[{tick}] must contain exactly {HERO_COUNT} slots")
        for hero, value in enumerate(row):
            try:
                action = Action.from_dict(value, path=f"projected_action[{tick}][{hero}]")
            except Exception as exc:
                raise AI42DatasetError(f"projected action cannot be decoded at {tick}/{hero}: {exc}") from exc
            actions.append(action)
        projected_rows.append(actions)
    derived_projected = _array_from_actions(projected_rows, "projected_action")
    derived_executed = _array_from_actions(
        [[record.executed_action for record in row] for row in records], "executed_action",
    )
    derived_valid = np.asarray([[record.valid for record in row] for row in records], dtype=_MASK_DTYPE)
    derived_rejection = np.asarray(
        [[_REJECTION_CODES.get(record.rejection_reason, REJECTION_REASON_UNKNOWN) for record in row] for row in records],
        dtype=_MASK_DTYPE,
    )
    derived_rewards = np.asarray(
        [[record.outcome.reward for record in row] for row in records], dtype=_FLOAT_DTYPE,
    )
    derived_done = np.asarray(
        [int(all(record.outcome.terminal for record in row)) for row in records], dtype=_MASK_DTYPE,
    )
    derived_winner = np.asarray(
        [int(next((record.outcome.winner for record in row if isinstance(record.outcome.winner, int)), -1)) for row in records],
        dtype=_INT_DTYPE,
    )
    optional = {
        "teacher_status": (match.teacher_status, derived_status, (ticks, HERO_COUNT), _MASK_DTYPE),
        "teacher_action": (match.teacher_action, derived_teacher, (ticks, HERO_COUNT), _ACTION_DTYPE),
        "projected_action": (match.projected_action, derived_projected, (ticks, HERO_COUNT), _ACTION_DTYPE),
        "executed_action": (match.executed_action, derived_executed, (ticks, HERO_COUNT), _ACTION_DTYPE),
        "executed_valid": (match.executed_valid, derived_valid, (ticks, HERO_COUNT), _MASK_DTYPE),
        "rejection_reason": (match.rejection_reason, derived_rejection, (ticks, HERO_COUNT), _MASK_DTYPE),
        "rewards": (match.rewards, derived_rewards, (ticks, HERO_COUNT), _FLOAT_DTYPE),
        "done": (match.done, derived_done, (ticks,), _MASK_DTYPE),
        "winner": (match.winner, derived_winner, (ticks,), _INT_DTYPE),
    }
    for name, (supplied, fallback, shape, dtype) in optional.items():
        if supplied is None:
            raise AI42DatasetError(f"{output.match.match_id}.{name} authoritative raw evidence is required")
        arrays[name] = _copy_exact(supplied, dtype, shape, f"{output.match.match_id}.{name}")
        if not np.array_equal(arrays[name], fallback):
            raise AI42DatasetError(
                f"{output.match.match_id}.{name} disagrees with the authoritative trajectory"
            )
        _validate_mask(arrays[name], name)
        if name in _ACTION_ARRAY_NAMES:
            _validate_action_array(arrays[name], name)
    if not np.isin(arrays["teacher_status"], tuple(_TEACHER_STATUSES)).all():
        raise AI42DatasetError("teacher_status contains an unknown v13 status")
    if not np.isin(arrays["rejection_reason"], tuple(_REJECTION_NAMES)).all():
        raise AI42DatasetError("rejection_reason contains an unknown v13 code")
    if np.any(arrays["done"][:-1] != 0) or arrays["done"][-1] != 1:
        raise AI42DatasetError("match must end at its only terminal tick")
    teacher_fields = arrays["teacher_action"]
    for status in _TEACHER_STATUSES:
        rows = arrays["teacher_status"] == status
        if not rows.any():
            continue
        raw = teacher_fields[rows]
        nonzero = (
            (raw["kind"] != 0) | (raw["target"] != 0)
            | (raw["direction"] != 0) | (raw["distance"] != 0)
        )
        if status in {TEACHER_STATUS_NONE, TEACHER_STATUS_UNAVAILABLE, TEACHER_STATUS_WAIT, TEACHER_STATUS_HOLD, TEACHER_STATUS_CANCEL} and nonzero.any():
            raise AI42DatasetError("teacher control status must carry a zero teacher action")
        if status == TEACHER_STATUS_ACTION and (raw["kind"] == 0).any():
            raise AI42DatasetError("teacher action status cannot carry wait")
    valid = arrays["executed_valid"]
    reasons = arrays["rejection_reason"]
    if np.any((valid == 1) & (reasons != REJECTION_REASON_NONE)):
        raise AI42DatasetError("accepted executed actions must have rejection_reason=none")
    if np.any((valid == 0) & (reasons == REJECTION_REASON_NONE)):
        raise AI42DatasetError("rejected executed actions must have a rejection reason")
    rejected = arrays["executed_action"][valid == 0]
    if rejected.size and (
        (rejected["kind"] != 0) | (rejected["target"] != 0)
        | (rejected["direction"] != 0) | (rejected["distance"] != 0)
    ).any():
        raise AI42DatasetError("rejected executed actions must carry a zero action")
    arrays["step"] = np.asarray(steps, dtype=_STEP_DTYPE)
    arrays["elapsed"] = (
        np.zeros((ticks,), dtype=_FLOAT_DTYPE)
        if match.elapsed is None
        else _copy_exact(match.elapsed, _FLOAT_DTYPE, (ticks,), f"{output.match.match_id}.elapsed")
    )
    arrays["invalid"] = (
        np.zeros((ticks, HERO_COUNT), dtype=_MASK_DTYPE)
        if match.invalid is None
        else _copy_exact(
            match.invalid, _MASK_DTYPE, (ticks, HERO_COUNT), f"{output.match.match_id}.invalid"
        )
    )
    _validate_mask(arrays["invalid"], "invalid")

    for name in _FLOAT_ARRAY_NAMES:
        if not np.isfinite(arrays[name]).all():
            raise AI42DatasetError(f"{name} contains non-finite values")
    parent_ids = [list(row) for row in output.recurrent_parent_ids]
    boundary_ids = [list(row) for row in output.recurrent_boundary_ids]
    if len(parent_ids) != ticks or len(boundary_ids) != ticks:
        raise AI42DatasetError("recurrent boundary coverage is incomplete")
    for tick in range(ticks):
        if len(parent_ids[tick]) != HERO_COUNT or len(boundary_ids[tick]) != HERO_COUNT:
            raise AI42DatasetError("recurrent boundaries must contain ten slots per tick")
        for hero in range(HERO_COUNT):
            if not isinstance(parent_ids[tick][hero], str) or not parent_ids[tick][hero]:
                raise AI42DatasetError("recurrent parent IDs must be non-empty strings")
            if not isinstance(boundary_ids[tick][hero], str) or not boundary_ids[tick][hero]:
                raise AI42DatasetError("recurrent boundary IDs must be non-empty strings")
            if tick and parent_ids[tick][hero] != boundary_ids[tick - 1][hero]:
                raise AI42DatasetError("recurrent parent IDs are not contiguous")
    return _PreparedMatch(
        match_id=output.match.match_id,
        hero_ids=hero_ids,
        steps=steps,
        arrays=arrays,
        trajectory_ids=tuple(trajectory.trajectory_id for trajectory in trajectories),
        trajectory_hashes=tuple(hash_payload(trajectory.to_bytes()) for trajectory in trajectories),
        recurrent_parent_ids=tuple(tuple(row) for row in parent_ids),
        recurrent_boundary_ids=tuple(tuple(row) for row in boundary_ids),
        seed=_metadata_seed(match.seed),
        scenario=_metadata_scenario(match.scenario),
        controller_by_slot=_metadata_controllers(match.controller_by_slot),
        roster_ids=_metadata_roster(match.roster_ids, hero_ids),
        side_by_slot=_metadata_sides(match.side_by_slot),
    )


@dataclass(frozen=True, slots=True)
class AI42RawMatch:
    """Compact authoritative evidence used by the production fast path."""

    results: tuple[StepResult, ...]
    submitted_actions: tuple[Any, ...]
    previous_recurrent_boundaries: tuple[tuple[str, ...], ...]
    recurrent_boundaries: tuple[tuple[str, ...], ...]
    hero_ids: tuple[str, ...]
    match_id: str
    runtime_manifest: Mapping[str, Any]
    runtime_manifest_hash: str


@dataclass(frozen=True, slots=True)
class AI42DatasetMatch:
    """A bridge output or compact raw v13 evidence plus durable fields."""

    output: AI42BridgeOutput | None = None
    teacher_status: Any = None
    teacher_action: Any = None
    projected_action: Any = None
    executed_action: Any = None
    executed_valid: Any = None
    rejection_reason: Any = None
    rewards: Any = None
    done: Any = None
    winner: Any = None
    elapsed: Any = None
    invalid: Any = None
    seed: int | None = None
    scenario: str | None = None
    controller_by_slot: Any = None
    roster_ids: Any = None
    side_by_slot: Any = None
    raw_capture: AI42RawMatch | None = None


@dataclass(frozen=True, slots=True)
class _PreparedMatch:
    match_id: str
    hero_ids: tuple[str, ...]
    steps: tuple[int, ...]
    arrays: Mapping[str, np.ndarray]
    trajectory_ids: tuple[str, ...]
    trajectory_hashes: tuple[str, ...]
    recurrent_parent_ids: tuple[tuple[str, ...], ...]
    recurrent_boundary_ids: tuple[tuple[str, ...], ...]
    seed: int
    scenario: str
    controller_by_slot: tuple[int, ...]
    roster_ids: tuple[Any, ...]
    side_by_slot: tuple[int, ...]


def _raw_exact_array(result: StepResult, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    value = getattr(result, name, None)
    if value is None:
        raise AI42DatasetError(f"raw result.{name} is required")
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise AI42DatasetError(
            f"raw result.{name} must have dtype {dtype}, shape {shape}; got {array.dtype}, {array.shape}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise AI42DatasetError(f"raw result.{name} contains non-finite values")
    return array


def _fast_trajectory_hashes(
    *,
    match_id: str,
    hero_ids: Sequence[str],
    steps: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    parents: Sequence[Sequence[str]],
    boundaries: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    """Hash compact trajectory evidence without creating one record per tick."""

    hashes: list[str] = []
    for hero, hero_id in enumerate(hero_ids):
        digest = hashlib.sha256()
        digest.update(b"AI42-fast-trajectory-v1\0")
        digest.update(canonical_json_bytes({
            "match_id": match_id,
            "hero_id": hero_id,
            "steps": list(steps),
            "parents": [row[hero] for row in parents],
            "boundaries": [row[hero] for row in boundaries],
        }))
        for name in _ARRAY_NAMES:
            array = arrays[name]
            if array.ndim >= 2 and array.shape[1] == HERO_COUNT:
                value = np.ascontiguousarray(array[:, hero])
            else:
                value = np.ascontiguousarray(array)
            digest.update(name.encode("ascii"))
            digest.update(hashlib.sha256(value.tobytes()).digest())
        hashes.append(digest.hexdigest())
    return tuple(hashes)


def _prepared_from_raw_match(match: "AI42DatasetMatch", raw: AI42RawMatch) -> "_PreparedMatch":
    results = tuple(raw.results)
    if not results:
        raise AI42DatasetError("raw match must contain at least one policy tick")
    if any(not isinstance(result, StepResult) for result in results):
        raise AI42DatasetError("raw match contains a non-v13 StepResult")
    if len(raw.hero_ids) != HERO_COUNT or len(set(raw.hero_ids)) != HERO_COUNT:
        raise AI42DatasetError("raw match must contain ten unique hero IDs")
    manifest = _as_mapping(raw.runtime_manifest, "runtime_manifest")
    if _runtime_manifest_hash(manifest) != _hash_hex(raw.runtime_manifest_hash, "runtime_manifest_hash"):
        raise AI42DatasetError("raw runtime manifest hash mismatch")
    ticks = tuple(int(result.step) for result in results)
    if ticks != tuple(range(ticks[0], ticks[0] + len(ticks))):
        raise AI42DatasetError("raw match ticks are not contiguous")
    if any(bool(result.done) for result in results[:-1]) or not bool(results[-1].done):
        raise AI42DatasetError("raw match must end at its only terminal tick")
    field_specs = {
        "hero": ("hero", (HERO_COUNT, HERO_FEATURES), _FLOAT_DTYPE),
        "abilities": ("abilities", (HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES), _FLOAT_DTYPE),
        "entities": ("entities", (HERO_COUNT, MAX_ENTITIES, ENTITY_FEATURES), _FLOAT_DTYPE),
        "global": ("global_state", (HERO_COUNT, GLOBAL_FEATURES), _FLOAT_DTYPE),
        "entity_mask": ("entity_mask", (HERO_COUNT, MAX_ENTITIES), _MASK_DTYPE),
        "kind_mask": ("kind_mask", (HERO_COUNT, ACTION_KINDS), _MASK_DTYPE),
        "target_mask": ("target_mask", (HERO_COUNT, MAX_ENTITIES), _MASK_DTYPE),
        "skill_target_mask": ("skill_target_mask", (HERO_COUNT, ABILITY_COUNT, MAX_ENTITIES), _MASK_DTYPE),
        "teacher_status": ("teacher_status", (HERO_COUNT,), _MASK_DTYPE),
        "teacher_action": ("teacher_intent", (HERO_COUNT,), _ACTION_DTYPE),
        "executed_action": ("executed_actions", (HERO_COUNT,), _ACTION_DTYPE),
        "executed_valid": ("executed_valid", (HERO_COUNT,), _MASK_DTYPE),
        "rejection_reason": ("rejection_reason", (HERO_COUNT,), _MASK_DTYPE),
        "rewards": ("rewards", (HERO_COUNT,), _FLOAT_DTYPE),
        "invalid": ("invalid", (HERO_COUNT,), _MASK_DTYPE),
    }
    arrays: dict[str, np.ndarray] = {}
    for durable, (source, shape, dtype) in field_specs.items():
        arrays[durable] = np.stack(
            [_raw_exact_array(result, source, dtype, shape) for result in results], axis=0,
        )
    arrays["projected_action"] = _wire_actions_fast(
        raw.submitted_actions, "submitted_actions", len(results),
    )
    arrays["done"] = np.asarray([int(result.done) for result in results], dtype=_MASK_DTYPE)
    arrays["winner"] = np.asarray([int(result.winner) for result in results], dtype=_INT_DTYPE)
    arrays["step"] = np.asarray(ticks, dtype=_STEP_DTYPE)
    arrays["elapsed"] = np.asarray([result.elapsed for result in results], dtype=_FLOAT_DTYPE)
    arrays["invalid"] = np.asarray(arrays["invalid"], dtype=_MASK_DTYPE)
    for name in ("entity_mask", "kind_mask", "target_mask", "skill_target_mask", "executed_valid", "invalid"):
        _validate_mask(arrays[name], name)
    for name in _ACTION_ARRAY_NAMES:
        _validate_action_array(arrays[name], name)
    if not np.isin(arrays["teacher_status"], tuple(_TEACHER_STATUSES)).all():
        raise AI42DatasetError("raw teacher_status contains an unknown v13 status")
    if not np.isin(arrays["rejection_reason"], tuple(_REJECTION_NAMES)).all():
        raise AI42DatasetError("raw rejection_reason contains an unknown v13 code")
    teacher = arrays["teacher_action"]
    zero_teacher = (
        (teacher["kind"] == 0) & (teacher["target"] == 0)
        & (teacher["direction"] == 0) & (teacher["distance"] == 0)
    )
    for status in (TEACHER_STATUS_NONE, TEACHER_STATUS_UNAVAILABLE, TEACHER_STATUS_WAIT, TEACHER_STATUS_HOLD, TEACHER_STATUS_CANCEL):
        if np.any((arrays["teacher_status"] == status) & ~zero_teacher):
            raise AI42DatasetError("raw control teacher status carries a non-zero action")
    if np.any((arrays["teacher_status"] == TEACHER_STATUS_ACTION) & zero_teacher):
        raise AI42DatasetError("raw action teacher status carries wait")
    valid = arrays["executed_valid"]
    reasons = arrays["rejection_reason"]
    if np.any((valid == 1) & (reasons != REJECTION_REASON_NONE)):
        raise AI42DatasetError("raw accepted action has a rejection reason")
    if np.any((valid == 0) & (reasons == REJECTION_REASON_NONE)):
        raise AI42DatasetError("raw rejected action has no rejection reason")
    rejected = arrays["executed_action"][valid == 0]
    if rejected.size and (
        (rejected["kind"] != 0) | (rejected["target"] != 0)
        | (rejected["direction"] != 0) | (rejected["distance"] != 0)
    ).any():
        raise AI42DatasetError("raw rejected action is non-zero")
    parent_ids = tuple(tuple(row) for row in raw.previous_recurrent_boundaries)
    boundary_ids = tuple(tuple(row) for row in raw.recurrent_boundaries)
    if len(parent_ids) != len(results) or len(boundary_ids) != len(results):
        raise AI42DatasetError("raw recurrent boundary coverage is incomplete")
    for tick in range(len(results)):
        if len(parent_ids[tick]) != HERO_COUNT or len(boundary_ids[tick]) != HERO_COUNT:
            raise AI42DatasetError("raw recurrent boundaries must contain ten slots per tick")
        for hero in range(HERO_COUNT):
            if not isinstance(parent_ids[tick][hero], str) or not parent_ids[tick][hero]:
                raise AI42DatasetError("raw recurrent parent ID is invalid")
            if not isinstance(boundary_ids[tick][hero], str) or not boundary_ids[tick][hero]:
                raise AI42DatasetError("raw recurrent boundary ID is invalid")
            if parent_ids[tick][hero] == boundary_ids[tick][hero]:
                raise AI42DatasetError("raw recurrent parent and boundary are identical")
            if tick and parent_ids[tick][hero] != boundary_ids[tick - 1][hero]:
                raise AI42DatasetError("raw recurrent parent IDs are not contiguous")
    for result in results:
        if not isinstance(result, StepResult):
            raise AI42DatasetError("raw match contains a non-v13 StepResult")
        if isinstance(result.done, (bool, np.bool_)) is False:
            raise AI42DatasetError("raw result.done must be boolean")
        if isinstance(result.winner, (bool, np.bool_)) or not isinstance(result.winner, (int, np.integer)):
            raise AI42DatasetError("raw result.winner must be an integer")
        if not isinstance(result.elapsed, (int, float, np.integer, np.floating)) or not np.isfinite(result.elapsed):
            raise AI42DatasetError("raw result.elapsed must be finite")
        if result.schema_hash is None or bytes(result.schema_hash) != AI42_SCHEMA_HASH:
            raise AI42DatasetError("raw result schema hash mismatch")
        if result.reward_hash is None or bytes(result.reward_hash) != AI42_REWARD_HASH:
            raise AI42DatasetError("raw result reward hash mismatch")
    _validate_mask(arrays["done"], "done")
    if np.any(arrays["done"][:-1] != 0) or arrays["done"][-1] != 1:
        raise AI42DatasetError("raw match must contain one terminal tick")
    for hero in range(HERO_COUNT):
        roots: dict[str, str] = {}
        cancelled: set[str] = set()
        for tick, (parent_row, boundary_row) in enumerate(zip(parent_ids, boundary_ids)):
            parent, boundary = parent_row[hero], boundary_row[hero]
            status = int(arrays["teacher_status"][tick, hero])
            if status == TEACHER_STATUS_HOLD:
                if tick == 0 or int(arrays["teacher_status"][tick - 1, hero]) in {
                    TEACHER_STATUS_WAIT, TEACHER_STATUS_CANCEL,
                }:
                    raise AI42DatasetError("raw HOLD does not reference a holdable teacher lineage")
                roots[boundary] = roots[parent]
            elif status == TEACHER_STATUS_CANCEL:
                if tick == 0 or int(arrays["teacher_status"][tick - 1, hero]) in {
                    TEACHER_STATUS_WAIT, TEACHER_STATUS_CANCEL,
                }:
                    raise AI42DatasetError("raw CANCEL does not reference a cancellable teacher lineage")
                root = roots[parent]
                if root in cancelled:
                    raise AI42DatasetError("raw CANCEL references an already cancelled lineage")
                cancelled.add(root)
                roots[boundary] = boundary
            else:
                roots[boundary] = boundary
    for name in _FLOAT_ARRAY_NAMES:
        if not np.isfinite(arrays[name]).all():
            raise AI42DatasetError(f"raw {name} contains non-finite values")
    return _PreparedMatch(
        match_id=raw.match_id,
        hero_ids=tuple(raw.hero_ids),
        steps=ticks,
        arrays=arrays,
        trajectory_ids=tuple(f"{raw.match_id}:hero:{hero}" for hero in raw.hero_ids),
        trajectory_hashes=_fast_trajectory_hashes(
            match_id=raw.match_id, hero_ids=raw.hero_ids, steps=ticks,
            arrays=arrays, parents=parent_ids, boundaries=boundary_ids,
        ),
        recurrent_parent_ids=parent_ids,
        recurrent_boundary_ids=boundary_ids,
        seed=_metadata_seed(match.seed),
        scenario=_metadata_scenario(match.scenario),
        controller_by_slot=_metadata_controllers(match.controller_by_slot),
        roster_ids=_metadata_roster(match.roster_ids, raw.hero_ids),
        side_by_slot=_metadata_sides(match.side_by_slot),
    )


def _validate_prepared_match(prepared: "_PreparedMatch", path: str) -> None:
    if not isinstance(prepared.match_id, str) or not prepared.match_id:
        raise AI42DatasetError(f"{path}.match_id must be a non-empty string")
    if len(prepared.hero_ids) != HERO_COUNT or len(set(prepared.hero_ids)) != HERO_COUNT:
        raise AI42DatasetError(f"{path} must contain ten unique hero IDs")
    if not prepared.steps or any(
        prepared.steps[index] != prepared.steps[0] + index
        for index in range(len(prepared.steps))
    ):
        raise AI42DatasetError(f"{path} ticks are not contiguous")
    if len(prepared.trajectory_ids) != HERO_COUNT or len(prepared.trajectory_hashes) != HERO_COUNT:
        raise AI42DatasetError(f"{path} has incomplete trajectory metadata")
    if len(prepared.recurrent_parent_ids) != len(prepared.steps) or len(prepared.recurrent_boundary_ids) != len(prepared.steps):
        raise AI42DatasetError(f"{path} has incomplete recurrent metadata")
    _metadata_seed(prepared.seed)
    _metadata_scenario(prepared.scenario)
    _metadata_controllers(prepared.controller_by_slot)
    _metadata_roster(prepared.roster_ids, prepared.hero_ids)
    _metadata_sides(prepared.side_by_slot)
    rows = _validate_shard_arrays(prepared.arrays, path)
    if rows != len(prepared.steps):
        raise AI42DatasetError(f"{path} array rows do not match tick count")
    done = prepared.arrays["done"]
    if np.any(done[:-1] != 0) or done[-1] != 1:
        raise AI42DatasetError(f"{path} must contain one terminal tick")


def _deterministic_split(match_ids: Sequence[str], validation_fraction: float, seed: int) -> dict[str, str]:
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise AI42DatasetError("validation_fraction must be numeric")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise AI42DatasetError("validation_fraction must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AI42DatasetError("split_seed must be an integer")
    ids = tuple(sorted(match_ids))
    if len(set(ids)) != len(ids):
        raise AI42DatasetError("match IDs must be unique")
    if not ids or fraction <= 0.0:
        return {match_id: "train" for match_id in ids}
    if fraction >= 1.0:
        return {match_id: "validation" for match_id in ids}
    count = min(len(ids) - 1, max(1, int(math.ceil(len(ids) * fraction)))) if len(ids) > 1 else 0
    ranked = sorted(ids, key=lambda match_id: hashlib.sha256(f"{seed}\0{match_id}".encode()).digest())
    validation = set(ranked[:count])
    return {match_id: ("validation" if match_id in validation else "train") for match_id in ids}


def _deterministic_stratified_split(
    match_scenarios: Sequence[tuple[str, str]],
    scenario_mix: Mapping[str, Any],
    seed: int,
    *,
    validation_fraction: float | None = None,
) -> dict[str, str]:
    """Assign a frozen split with exact per-scenario train/validation quotas.

    Ranking is based only on the match ID and split seed.  The scenario is
    used to apply each configured quota, so worker scheduling and completion
    order cannot affect either the assignment or the resulting counts.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AI42DatasetError("split_seed must be an integer")
    if not isinstance(scenario_mix, Mapping) or not scenario_mix:
        raise AI42DatasetError("scenario_mix must be a non-empty mapping")
    pairs: list[tuple[str, str]] = []
    for item in match_scenarios:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise AI42DatasetError("match_scenarios must contain (match_id, scenario) pairs")
        match_id, scenario = item
        if not isinstance(match_id, str) or not match_id:
            raise AI42DatasetError("match IDs must be non-empty strings")
        if not isinstance(scenario, str) or not scenario:
            raise AI42DatasetError("scenario names must be non-empty strings")
        pairs.append((match_id, scenario))
    ids = [match_id for match_id, _ in pairs]
    if not ids or len(set(ids)) != len(ids):
        raise AI42DatasetError("match IDs must be unique")

    normalized: dict[str, dict[str, int]] = {}
    for scenario, quota in scenario_mix.items():
        if not isinstance(scenario, str) or not scenario:
            raise AI42DatasetError("scenario_mix keys must be non-empty strings")
        if not isinstance(quota, Mapping) or set(quota) != {"train", "validation"}:
            raise AI42DatasetError(
                f"scenario_mix[{scenario!r}] must contain exactly train and validation quotas"
            )
        values: dict[str, int] = {}
        for split in ("train", "validation"):
            value = quota[split]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AI42DatasetError(f"scenario_mix[{scenario!r}].{split} must be a non-negative integer")
            values[split] = value
        normalized[scenario] = values

    observed = {scenario for _, scenario in pairs}
    if set(normalized) != observed:
        raise AI42DatasetError("scenario_mix scenarios must exactly match the schedule")
    grouped: dict[str, list[str]] = {scenario: [] for scenario in normalized}
    for match_id, scenario in pairs:
        grouped[scenario].append(match_id)
    validation_total = 0
    for scenario, match_ids in grouped.items():
        quota = normalized[scenario]
        if quota["train"] + quota["validation"] != len(match_ids):
            raise AI42DatasetError(f"scenario_mix quota does not match {scenario!r} match count")
        validation_total += quota["validation"]
    if validation_fraction is not None:
        if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
            raise AI42DatasetError("validation_fraction must be numeric")
        fraction = float(validation_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise AI42DatasetError("validation_fraction must be between zero and one")
        if validation_total != len(ids) * fraction:
            raise AI42DatasetError("scenario_mix validation quotas do not match validation_fraction")

    assignments: dict[str, str] = {}
    for scenario, match_ids in grouped.items():
        ranked = sorted(
            match_ids,
            key=lambda match_id: (
                hashlib.sha256(f"{seed}\0{match_id}".encode("utf-8")).digest(),
                match_id,
            ),
        )
        validation_ids = set(ranked[: normalized[scenario]["validation"]])
        assignments.update({match_id: ("validation" if match_id in validation_ids else "train") for match_id in match_ids})
    return assignments


def _npy_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            if name not in _ARRAY_DTYPES:
                raise AI42DatasetError(f"unexpected shard array {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=6)
    return output.getvalue()


def _npz_raw_bytes(arrays: Mapping[str, np.ndarray]) -> int:
    """Return the deterministic ZIP_STORED size without retaining a copy."""

    total = 22  # end-of-central-directory record
    for name in sorted(arrays):
        if name not in _ARRAY_DTYPES:
            raise AI42DatasetError(f"unexpected shard array {name!r}")
        member = _npy_bytes(arrays[name])
        member_name = f"{name}.npy".encode("utf-8")
        total += 30 + len(member_name) + len(member)  # local file header + data
        total += 46 + len(member_name)  # central directory record
    return total


def _assert_no_zip_trailing(payload: bytes, path: str) -> None:
    marker = b"PK\x05\x06"
    offset = payload.rfind(marker)
    if offset < 0 or offset + 22 > len(payload):
        raise AI42DatasetError(f"{path} is not a complete ZIP archive")
    comment_length = struct.unpack_from("<H", payload, offset + 20)[0]
    if offset + 22 + comment_length != len(payload):
        raise AI42DatasetError(f"{path} has trailing bytes")


def _read_npz(payload: bytes, path: str) -> dict[str, np.ndarray]:
    _assert_no_zip_trailing(payload, path)
    try:
        with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise AI42DatasetError(f"{path} contains duplicate ZIP members")
            expected_names = {f"{name}.npy" for name in _ARRAY_NAMES}
            if set(names) != expected_names:
                missing = sorted(expected_names - set(names))
                extra = sorted(set(names) - expected_names)
                raise AI42DatasetError(f"{path} array set mismatch: missing={missing}, extra={extra}")
            result: dict[str, np.ndarray] = {}
            for name in _ARRAY_NAMES:
                raw = archive.read(f"{name}.npy")
                try:
                    array = np.load(BytesIO(raw), allow_pickle=False)
                except Exception as exc:
                    raise AI42DatasetError(f"{path}:{name} is not a valid NumPy array") from exc
                if not isinstance(array, np.ndarray):
                    raise AI42DatasetError(f"{path}:{name} is not an ndarray")
                result[name] = array
    except AI42DatasetError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AI42DatasetError(f"{path} is corrupt: {exc}") from exc
    return result


def _validate_shard_arrays(arrays: Mapping[str, np.ndarray], path: str) -> int:
    _exact_fields(arrays, frozenset(_ARRAY_NAMES), path)
    rows: int | None = None
    for name in _ARRAY_NAMES:
        array = arrays[name]
        expected_dtype = _ARRAY_DTYPES[name]
        if array.dtype != expected_dtype:
            raise AI42DatasetError(f"{path}:{name} has dtype {array.dtype}, expected {expected_dtype}")
        if not array.ndim or (rows is not None and array.shape[0] != rows):
            raise AI42DatasetError(f"{path}:{name} has inconsistent leading dimension")
        rows = int(array.shape[0])
        if name == "hero" and array.shape[1:] != (HERO_COUNT, HERO_FEATURES):
            raise AI42DatasetError(f"{path}:hero has an invalid shape")
        if name == "abilities" and array.shape[1:] != (HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES):
            raise AI42DatasetError(f"{path}:abilities has an invalid shape")
        if name == "entities" and array.shape[1:] != (HERO_COUNT, MAX_ENTITIES, ENTITY_FEATURES):
            raise AI42DatasetError(f"{path}:entities has an invalid shape")
        if name == "global" and array.shape[1:] != (HERO_COUNT, GLOBAL_FEATURES):
            raise AI42DatasetError(f"{path}:global has an invalid shape")
        if name in {"entity_mask", "target_mask"} and array.shape[1:] != (HERO_COUNT, MAX_ENTITIES):
            raise AI42DatasetError(f"{path}:{name} has an invalid shape")
        if name == "kind_mask" and array.shape[1:] != (HERO_COUNT, ACTION_KINDS):
            raise AI42DatasetError(f"{path}:kind_mask has an invalid shape")
        if name == "skill_target_mask" and array.shape[1:] != (HERO_COUNT, ABILITY_COUNT, MAX_ENTITIES):
            raise AI42DatasetError(f"{path}:skill_target_mask has an invalid shape")
        if name in {"teacher_status", "teacher_action", "projected_action", "executed_action", "executed_valid", "rejection_reason", "rewards", "invalid"} and array.shape[1:] != (HERO_COUNT,):
            raise AI42DatasetError(f"{path}:{name} has an invalid shape")
        if name in {"done", "winner", "step", "elapsed"} and array.shape[1:] != ():
            raise AI42DatasetError(f"{path}:{name} has an invalid shape")
        if name in _FLOAT_ARRAY_NAMES and not np.isfinite(array).all():
            raise AI42DatasetError(f"{path}:{name} contains non-finite values")
        _validate_mask(array, name)
        if name in _ACTION_ARRAY_NAMES:
            _validate_action_array(array, name)
    assert rows is not None
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


_STAGING_SCHEMA_VERSION = "AI42-staging-v1"
_STAGING_MATCH_FIELDS = frozenset(
    {
        "staging_schema_version", "match_id", "metadata_hash", "arrays_hash",
        "hero_ids", "steps", "trajectory_ids", "trajectory_hashes",
        "recurrent_parent_ids", "recurrent_boundary_ids", "seed", "scenario",
        "controller_by_slot", "roster_ids", "side_by_slot",
    }
)


def _stage_directory_name(match_id: str) -> str:
    return f"match-{hashlib.sha256(match_id.encode('utf-8')).hexdigest()}"


def _prepared_metadata(prepared: "_PreparedMatch", arrays_hash: str) -> dict[str, Any]:
    return {
        "staging_schema_version": _STAGING_SCHEMA_VERSION,
        "match_id": prepared.match_id,
        "metadata_hash": "",
        "arrays_hash": arrays_hash,
        "hero_ids": list(prepared.hero_ids),
        "steps": list(prepared.steps),
        "trajectory_ids": list(prepared.trajectory_ids),
        "trajectory_hashes": list(prepared.trajectory_hashes),
        "recurrent_parent_ids": [list(row) for row in prepared.recurrent_parent_ids],
        "recurrent_boundary_ids": [list(row) for row in prepared.recurrent_boundary_ids],
        "seed": prepared.seed,
        "scenario": prepared.scenario,
        "controller_by_slot": list(prepared.controller_by_slot),
        "roster_ids": list(prepared.roster_ids),
        "side_by_slot": list(prepared.side_by_slot),
    }


def _prepared_from_stage(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> "_PreparedMatch":
    _exact_fields(metadata, _STAGING_MATCH_FIELDS, "staged match")
    if metadata["staging_schema_version"] != _STAGING_SCHEMA_VERSION:
        raise AI42DatasetError("staged match schema version mismatch")
    metadata_hash = _hash_hex(metadata["metadata_hash"], "staged match.metadata_hash")
    unsigned = {key: value for key, value in metadata.items() if key != "metadata_hash"}
    if hash_payload(unsigned) != metadata_hash:
        raise AI42DatasetError("staged match metadata_hash mismatch")
    if not isinstance(metadata["arrays_hash"], str):
        raise AI42DatasetError("staged match arrays_hash is missing")
    _hash_hex(metadata["arrays_hash"], "staged match.arrays_hash")
    prepared = _PreparedMatch(
        match_id=metadata["match_id"],
        hero_ids=tuple(metadata["hero_ids"]),
        steps=tuple(metadata["steps"]),
        arrays=arrays,
        trajectory_ids=tuple(metadata["trajectory_ids"]),
        trajectory_hashes=tuple(metadata["trajectory_hashes"]),
        recurrent_parent_ids=tuple(tuple(row) for row in metadata["recurrent_parent_ids"]),
        recurrent_boundary_ids=tuple(tuple(row) for row in metadata["recurrent_boundary_ids"]),
        seed=metadata["seed"],
        scenario=metadata["scenario"],
        controller_by_slot=tuple(metadata["controller_by_slot"]),
        roster_ids=tuple(metadata["roster_ids"]),
        side_by_slot=tuple(metadata["side_by_slot"]),
    )
    _validate_prepared_match(prepared, f"staged match {prepared.match_id!r}")
    return prepared


class AI42DatasetStaging:
    """Crash-safe, resumable one-match spool for AI42 collection.

    A staging generation contains one immutable directory per complete match.
    Each directory has a canonical metadata file, deterministic NPZ payload and
    completion digest. Partial or modified directories are rejected. The
    contract is written once and must compare byte-for-byte on resume.
    """

    def __init__(self, root: str | os.PathLike[str], contract: Mapping[str, Any]) -> None:
        self.root = Path(root)
        supplied = dict(_as_mapping(contract, "staging contract"))
        contract_bytes = canonical_json_bytes(supplied)
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink():
                raise AI42DatasetError("staging root must be a directory")
            try:
                actual_bytes = (self.root / "contract.json").read_bytes()
            except OSError as exc:
                raise AI42DatasetError(f"staging contract is missing: {exc}") from exc
            actual = _strict_json(actual_bytes, str(self.root / "contract.json"))
            if canonical_json_bytes(actual) != contract_bytes:
                raise AI42DatasetError("staging contract mismatch; resume requires the exact contract")
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.root.mkdir()
            _atomic_write(self.root / "contract.json", contract_bytes)
        self.contract = supplied
        self.matches_root = self.root / "matches"
        self.matches_root.mkdir(exist_ok=True)
        self._validate_layout()

    def _validate_layout(self) -> None:
        allowed = {"contract.json", "matches"}
        actual = {item.name for item in self.root.iterdir()}
        extra = actual - allowed
        if extra:
            raise AI42DatasetError(f"staging root contains unexpected entries: {sorted(extra)}")
        for item in self.matches_root.iterdir():
            if not item.is_dir() or item.is_symlink() or not item.name.startswith("match-"):
                raise AI42DatasetError(f"staging contains a partial or unexpected match entry: {item.name}")
            if {child.name for child in item.iterdir()} != {"metadata.json", "arrays.npz", "COMPLETE"}:
                raise AI42DatasetError(f"staged match {item.name} is partial or has unexpected files")

    def _match_path(self, match_id: str) -> Path:
        return self.matches_root / _stage_directory_name(match_id)

    def _load_path(self, path: Path) -> "_PreparedMatch":
        expected = {"metadata.json", "arrays.npz", "COMPLETE"}
        try:
            names = {item.name for item in path.iterdir()}
        except OSError as exc:
            raise AI42DatasetError(f"cannot inspect staged match {path.name}: {exc}") from exc
        if names != expected:
            raise AI42DatasetError(f"staged match {path.name} is partial or has unexpected files")
        try:
            metadata_bytes = (path / "metadata.json").read_bytes()
            arrays_payload = (path / "arrays.npz").read_bytes()
            marker = (path / "COMPLETE").read_bytes()
        except OSError as exc:
            raise AI42DatasetError(f"staged match {path.name} is unreadable: {exc}") from exc
        if marker != hashlib.sha256(metadata_bytes + arrays_payload).hexdigest().encode("ascii"):
            raise AI42DatasetError(f"staged match {path.name} completion digest mismatch")
        metadata = _strict_json(metadata_bytes, str(path / "metadata.json"))
        if not isinstance(metadata, Mapping):
            raise AI42DatasetError(f"staged match {path.name} metadata is not an object")
        if _sha256_bytes(arrays_payload) != metadata.get("arrays_hash"):
            raise AI42DatasetError(f"staged match {path.name} arrays hash mismatch")
        arrays = _read_npz(arrays_payload, str(path / "arrays.npz"))
        return _prepared_from_stage(metadata, arrays)

    def _metadata_only(self, path: Path) -> Mapping[str, Any]:
        expected = {"metadata.json", "arrays.npz", "COMPLETE"}
        if {item.name for item in path.iterdir()} != expected:
            raise AI42DatasetError(f"staged match {path.name} is partial or has unexpected files")
        try:
            metadata_bytes = (path / "metadata.json").read_bytes()
        except OSError as exc:
            raise AI42DatasetError(f"staged match {path.name} metadata is unreadable: {exc}") from exc
        metadata = _strict_json(metadata_bytes, str(path / "metadata.json"))
        if not isinstance(metadata, Mapping):
            raise AI42DatasetError(f"staged match {path.name} metadata is not an object")
        _exact_fields(metadata, _STAGING_MATCH_FIELDS, "staged match")
        if metadata["staging_schema_version"] != _STAGING_SCHEMA_VERSION:
            raise AI42DatasetError("staged match schema version mismatch")
        metadata_hash = _hash_hex(metadata["metadata_hash"], "staged match.metadata_hash")
        unsigned = {key: value for key, value in metadata.items() if key != "metadata_hash"}
        if hash_payload(unsigned) != metadata_hash:
            raise AI42DatasetError("staged match metadata_hash mismatch")
        return metadata

    @property
    def match_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for path in sorted(self.matches_root.iterdir(), key=lambda item: item.name):
            metadata = self._metadata_only(path)
            ids.append(metadata["match_id"])
        if len(ids) != len(set(ids)):
            raise AI42DatasetError("staging contains duplicate match IDs")
        return tuple(sorted(ids))

    def contains(self, match_id: str) -> bool:
        return self._match_path(match_id).exists()

    def load_prepared(self, match_id: str) -> "_PreparedMatch":
        path = self._match_path(match_id)
        if not path.exists():
            raise KeyError(match_id)
        prepared = self._load_path(path)
        if prepared.match_id != match_id:
            raise AI42DatasetError(f"staged match path identity mismatch for {match_id!r}")
        return prepared

    def add_prepared(self, prepared: "_PreparedMatch", *, validate: bool = True) -> None:
        if validate:
            _validate_prepared_match(prepared, "staged match")
        destination = self._match_path(prepared.match_id)
        if destination.exists():
            raise AI42DatasetError(f"duplicate staged match_id {prepared.match_id!r}")
        arrays_payload = _deterministic_npz(prepared.arrays)
        metadata = _prepared_metadata(prepared, _sha256_bytes(arrays_payload))
        metadata["metadata_hash"] = hash_payload(
            {key: value for key, value in metadata.items() if key != "metadata_hash"}
        )
        metadata_bytes = canonical_json_bytes(metadata)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(self.matches_root)))
        try:
            _atomic_write(temporary / "metadata.json", metadata_bytes)
            _atomic_write(temporary / "arrays.npz", arrays_payload)
            _atomic_write(
                temporary / "COMPLETE",
                hashlib.sha256(metadata_bytes + arrays_payload).hexdigest().encode("ascii"),
            )
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def add_match(self, match: AI42DatasetMatch) -> None:
        if isinstance(match, AI42BridgeOutput):
            raise AI42DatasetError("staging requires AI42DatasetMatch authoritative evidence")
        # _prepared_from_match already performs the complete strict replay and
        # array validation; avoid repeating that pass before spooling.
        self.add_prepared(_prepared_from_match(match), validate=False)

    def iter_prepared(self) -> Iterator["_PreparedMatch"]:
        for match_id in self.match_ids:
            yield self.load_prepared(match_id)


AI42DatasetSpool = AI42DatasetStaging


class AI42DatasetWriter:
    """Collect validated matches and publish deterministic durable shards."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        runtime_manifest: Mapping[str, Any] | Any | None = None,
        runtime_manifest_hash: str | bytes | None = None,
        shard_size: int = 1,
        validation_fraction: float = 0.2,
        split_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
            raise AI42DatasetError("shard_size must be a positive integer")
        self.shard_size = shard_size
        self.validation_fraction = float(validation_fraction)
        self.split_seed = split_seed
        self._matches: dict[str, _PreparedMatch] = {}
        self._runtime_manifest = None if runtime_manifest is None else dict(_as_mapping(runtime_manifest, "runtime_manifest"))
        self._runtime_manifest_hash = None if runtime_manifest_hash is None else _hash_hex(runtime_manifest_hash, "runtime_manifest_hash")
        if self._runtime_manifest is not None:
            actual_hash = _runtime_manifest_hash(self._runtime_manifest)
            if self._runtime_manifest_hash is not None and self._runtime_manifest_hash != actual_hash:
                raise AI42DatasetError("runtime_manifest_hash does not match runtime_manifest")
            self._runtime_manifest_hash = actual_hash

    def add_match(self, match: AI42DatasetMatch | AI42BridgeOutput) -> None:
        if isinstance(match, AI42BridgeOutput):
            raise AI42DatasetError(
                "AI42BridgeOutput lacks authoritative raw teacher/execution fields; use AI42DatasetMatch"
            )
        prepared = _prepared_from_match(match)
        output_manifest = dict(_as_mapping(match.output.manifest, "match.output.manifest"))
        output_manifest_hash = _hash_hex(match.output.manifest_hash, "match.output.manifest_hash")
        if _runtime_manifest_hash(output_manifest) != output_manifest_hash:
            raise AI42DatasetError("bridge runtime manifest hash is invalid")
        if self._runtime_manifest is None:
            if self._runtime_manifest_hash is not None and self._runtime_manifest_hash != output_manifest_hash:
                raise AI42DatasetError("match runtime manifest hash does not match the configured hash")
            self._runtime_manifest = output_manifest
            self._runtime_manifest_hash = output_manifest_hash
        elif self._runtime_manifest_hash != output_manifest_hash:
            raise AI42DatasetError("all matches must use one runtime manifest hash")
        if prepared.match_id in self._matches:
            raise AI42DatasetError(f"duplicate match_id {prepared.match_id!r}")
        self._matches[prepared.match_id] = prepared

    def add_prepared(self, prepared: "_PreparedMatch") -> None:
        """Add one already staged and independently validated match.

        The disk spool uses this narrow hook during streaming publication. It
        intentionally does not accept arbitrary mappings or bridge outputs;
        callers must obtain the value from :class:`AI42DatasetStaging`.
        """

        if not isinstance(prepared, _PreparedMatch):
            raise AI42DatasetError("prepared match has an invalid type")
        _validate_prepared_match(prepared, "prepared match")
        if prepared.match_id in self._matches:
            raise AI42DatasetError(f"duplicate match_id {prepared.match_id!r}")
        self._matches[prepared.match_id] = prepared

    add_capture = add_match

    def write(self) -> "AI42Dataset":
        """Publish one complete dataset generation by an atomic directory swap."""

        destination = self.root
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise AI42DatasetError(
                    "dataset destination must be absent or an empty directory; generations are immutable"
                )
            destination.rmdir()
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        self.root = staging
        try:
            self._write_generation()
            os.replace(staging, destination)
            self.root = destination
            return load_dataset(destination)
        except Exception:
            self.root = destination
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _write_generation(self) -> "AI42Dataset":
        return self._write_generation_from_ids(
            tuple(sorted(self._matches)), self._matches.__getitem__,
        )

    def _write_generation_from_ids(
        self,
        match_ids: Sequence[str],
        loader: Any,
        *,
        validate_prepared: bool = True,
    ) -> "AI42Dataset":
        if not match_ids:
            raise AI42DatasetError("cannot publish an empty AI42 dataset")
        if self._runtime_manifest is None or self._runtime_manifest_hash is None:
            raise AI42DatasetError("runtime_manifest is required")
        ordered_ids = tuple(sorted(match_ids))
        split = _deterministic_split(ordered_ids, self.validation_fraction, self.split_seed)
        self.root.mkdir(parents=True, exist_ok=True)
        shard_entries: list[dict[str, Any]] = []
        match_entries: list[dict[str, Any]] = []
        for shard_number in range(0, len(ordered_ids), self.shard_size):
            shard_ids = ordered_ids[shard_number : shard_number + self.shard_size]
            arrays: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_NAMES}
            offset = 0
            shard_match_ids: list[str] = []
            for match_id in shard_ids:
                prepared = loader(match_id)
                if not isinstance(prepared, _PreparedMatch):
                    raise AI42DatasetError(f"loader returned an invalid match for {match_id!r}")
                if validate_prepared:
                    _validate_prepared_match(prepared, f"match {match_id!r}")
                if prepared.match_id != match_id:
                    raise AI42DatasetError(f"loader match ID mismatch for {match_id!r}")
                shard_match_ids.append(prepared.match_id)
                tick_count = len(prepared.steps)
                for name in _ARRAY_NAMES:
                    arrays[name].append(prepared.arrays[name])
                match_entries.append(
                    {
                        "match_id": prepared.match_id,
                        "split": split[prepared.match_id],
                        "shard": f"{SHARD_PREFIX}{shard_number // self.shard_size:06d}{SHARD_SUFFIX}",
                        "row_offset": offset,
                        "tick_count": tick_count,
                        "ticks": list(prepared.steps),
                        "hero_ids": list(prepared.hero_ids),
                        "trajectory_ids": list(prepared.trajectory_ids),
                        "trajectory_hashes": list(prepared.trajectory_hashes),
                        "recurrent_parent_ids": [list(row) for row in prepared.recurrent_parent_ids],
                        "recurrent_boundary_ids": [list(row) for row in prepared.recurrent_boundary_ids],
                        "seed": prepared.seed,
                        "scenario": prepared.scenario,
                        "controller_by_slot": list(prepared.controller_by_slot),
                        "roster_ids": list(prepared.roster_ids),
                        "side_by_slot": list(prepared.side_by_slot),
                    }
                )
                offset += tick_count
            shard_arrays = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
            if validate_prepared:
                _validate_shard_arrays(shard_arrays, "new shard")
            payload = _deterministic_npz(shard_arrays)
            shard_name = f"{SHARD_PREFIX}{shard_number // self.shard_size:06d}{SHARD_SUFFIX}"
            _atomic_write(self.root / shard_name, payload)
            shard_entries.append(
                {
                    "name": shard_name,
                    "sha256": _sha256_bytes(payload),
                    "match_ids": shard_match_ids,
                    "row_count": offset,
                    "raw_bytes": _npz_raw_bytes(shard_arrays),
                    "stored_bytes": len(payload),
                    "compression": "deflate-6",
                }
            )
        manifest: dict[str, Any] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "shard_schema_version": SHARD_SCHEMA_VERSION,
            "protocol_version": AI42_PROTOCOL_VERSION,
            "schema_hash": AI42_SCHEMA_HASH.hex(),
            "reward_hash": AI42_REWARD_HASH.hex(),
            "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
            "runtime_manifest_hash": self._runtime_manifest_hash,
            "runtime_manifest": self._runtime_manifest,
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "matches": sorted(match_entries, key=lambda item: item["match_id"]),
            "shards": shard_entries,
        }
        manifest["manifest_hash"] = hash_payload({key: value for key, value in manifest.items()})
        _atomic_write(self.root / MANIFEST_FILENAME, canonical_json_bytes(manifest))
        return load_dataset(self.root) if validate_prepared else None  # type: ignore[return-value]


def publish_staged_dataset(
    root: str | os.PathLike[str],
    staging: AI42DatasetStaging,
    *,
    runtime_manifest: Mapping[str, Any] | Any | None = None,
    runtime_manifest_hash: str | bytes | None = None,
    shard_size: int | None = None,
    validation_fraction: float | None = None,
    split_seed: int | None = None,
) -> "AI42Dataset":
    """Stream a complete staging generation into an immutable final tree."""

    if not isinstance(staging, AI42DatasetStaging):
        raise AI42DatasetError("staging must be an AI42DatasetStaging")
    contract = staging.contract
    configured_manifest = contract.get("runtime_manifest")
    if runtime_manifest is None:
        runtime_manifest = configured_manifest
    if runtime_manifest is None:
        raise AI42DatasetError("staging contract has no runtime_manifest")
    configured_shard_size = contract.get("shard_size", 1)
    configured_fraction = contract.get("validation_fraction", 0.2)
    configured_split_seed = contract.get("split_seed", 0)
    shard_size = configured_shard_size if shard_size is None else shard_size
    validation_fraction = configured_fraction if validation_fraction is None else validation_fraction
    split_seed = configured_split_seed if split_seed is None else split_seed
    destination = Path(root)
    if destination.exists():
        raise AI42DatasetError("final dataset destination already exists; generations are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    match_ids = staging.match_ids
    expected_match_ids = contract.get("match_ids")
    if expected_match_ids is not None:
        if not isinstance(expected_match_ids, list) or tuple(expected_match_ids) != match_ids:
            raise AI42DatasetError("staging match set does not exactly satisfy its contract")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=str(destination.parent)))
    writer = AI42DatasetWriter(
        temporary,
        runtime_manifest=runtime_manifest,
        runtime_manifest_hash=runtime_manifest_hash,
        shard_size=int(shard_size),
        validation_fraction=float(validation_fraction),
        split_seed=int(split_seed),
    )
    try:
        # load_prepared has already validated strict stage metadata, payload
        # hashes, array contracts, and terminal boundaries. The final tree is
        # validated once after its atomic rename.
        writer._write_generation_from_ids(
            match_ids, staging.load_prepared, validate_prepared=False,
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_dataset(destination, expected_runtime_manifest=runtime_manifest)


@dataclass(frozen=True, slots=True)
class AI42Dataset:
    """Validated, read-only view of a published dataset."""

    root: Path
    manifest: Mapping[str, Any]
    # At most one validated shard is retained. Training iterators therefore
    # never cause an entire multi-match generation to remain resident.
    _shards: dict[str, Mapping[str, np.ndarray]]

    def __len__(self) -> int:
        return len(self.manifest["matches"])

    @property
    def manifest_hash(self) -> str:
        return self.manifest["manifest_hash"]

    @property
    def runtime_manifest_hash(self) -> str:
        return self.manifest["runtime_manifest_hash"]

    @property
    def compression(self) -> str:
        values = {shard["compression"] for shard in self.manifest["shards"]}
        return next(iter(values)) if len(values) == 1 else "mixed"

    def match_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split not in (None, "train", "validation"):
            raise AI42DatasetError(f"unknown split {split!r}")
        return tuple(
            entry["match_id"] for entry in self.manifest["matches"] if split is None or entry["split"] == split
        )

    def split_match_ids(self) -> dict[str, tuple[str, ...]]:
        return {split: self.match_ids(split) for split in ("train", "validation")}

    def _entry(self, match_id: str) -> Mapping[str, Any]:
        for entry in self.manifest["matches"]:
            if entry["match_id"] == match_id:
                return entry
        raise KeyError(match_id)

    def _load_shard(self, name: str) -> Mapping[str, np.ndarray]:
        cached = self._shards.get(name)
        if cached is not None:
            return cached
        shard = next((item for item in self.manifest["shards"] if item["name"] == name), None)
        if shard is None:
            raise AI42DatasetError(f"missing shard metadata for {name!r}")
        path = self.root / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AI42DatasetError(f"cannot read {path}: {exc}") from exc
        if _sha256_bytes(payload) != shard["sha256"]:
            raise AI42DatasetError(f"shard hash mismatch for {name}")
        arrays = _read_npz(payload, str(path))
        if len(payload) != shard["stored_bytes"]:
            raise AI42DatasetError(f"stored shard size mismatch for {name}")
        rows = _validate_shard_arrays(arrays, str(path))
        if rows != shard["row_count"]:
            raise AI42DatasetError(f"shard row_count mismatch for {name}")
        self._shards.clear()
        self._shards[name] = arrays
        return arrays

    def arrays_for_match(self, match_id: str) -> dict[str, np.ndarray]:
        entry = self._entry(match_id)
        arrays = self._load_shard(entry["shard"])
        start = int(entry["row_offset"])
        end = start + int(entry["tick_count"])
        return {name: np.array(array[start:end], copy=True) for name, array in arrays.items()}

    def iter_matches(self, split: str | None = None) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        for match_id in self.match_ids(split):
            yield match_id, self.arrays_for_match(match_id)

    def iter_rows(self, split: str | None = None) -> Iterator[dict[str, Any]]:
        for entry in self.manifest["matches"]:
            if split is not None and entry["split"] != split:
                continue
            arrays = self.arrays_for_match(entry["match_id"])
            for tick in range(int(entry["tick_count"])):
                yield {
                    "match_id": entry["match_id"],
                    "tick": int(entry["ticks"][tick]),
                    "arrays": {name: array[tick].copy() for name, array in arrays.items()},
                }


def _validate_manifest(manifest: Mapping[str, Any], expected_runtime_manifest_hash: str | bytes | None) -> None:
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, "manifest")
    if manifest["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise AI42DatasetError("dataset schema version mismatch")
    if manifest["shard_schema_version"] != SHARD_SCHEMA_VERSION:
        raise AI42DatasetError("shard schema version mismatch")
    if manifest["protocol_version"] != AI42_PROTOCOL_VERSION:
        raise AI42DatasetError("protocol version mismatch")
    expected = {
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise AI42DatasetError(f"manifest {field} mismatch")
    runtime = _as_mapping(manifest["runtime_manifest"], "manifest.runtime_manifest")
    runtime_hash = _hash_hex(manifest["runtime_manifest_hash"], "manifest.runtime_manifest_hash")
    if _runtime_manifest_hash(runtime) != runtime_hash:
        raise AI42DatasetError("manifest runtime_manifest_hash mismatch")
    if expected_runtime_manifest_hash is not None and runtime_hash != _hash_hex(expected_runtime_manifest_hash, "expected_runtime_manifest_hash"):
        raise AI42DatasetError("runtime manifest hash mismatch")
    if isinstance(manifest["split_seed"], bool) or not isinstance(manifest["split_seed"], int):
        raise AI42DatasetError("manifest.split_seed must be an integer")
    if not isinstance(manifest["validation_fraction"], (int, float)) or isinstance(manifest["validation_fraction"], bool):
        raise AI42DatasetError("manifest.validation_fraction must be numeric")
    if not isinstance(manifest["matches"], list) or not isinstance(manifest["shards"], list):
        raise AI42DatasetError("manifest matches and shards must be arrays")
    if not manifest["matches"] or not manifest["shards"]:
        raise AI42DatasetError("manifest must contain at least one match and shard")
    split = _deterministic_split(
        tuple(entry["match_id"] for entry in manifest["matches"]),
        float(manifest["validation_fraction"]),
        manifest["split_seed"],
    )
    shard_names: set[str] = set()
    for shard in manifest["shards"]:
        if not isinstance(shard, Mapping):
            raise AI42DatasetError("manifest shard entry must be an object")
        _exact_fields(shard, _SHARD_FIELDS, "manifest.shard")
        name = shard["name"]
        if not isinstance(name, str) or Path(name).name != name or not name.startswith(SHARD_PREFIX) or not name.endswith(SHARD_SUFFIX):
            raise AI42DatasetError(f"invalid shard name {name!r}")
        if name in shard_names:
            raise AI42DatasetError(f"duplicate shard name {name!r}")
        shard_names.add(name)
        _hash_hex(shard["sha256"], f"manifest.shards[{name}].sha256")
        if (
            not isinstance(shard["match_ids"], list)
            or isinstance(shard["row_count"], bool)
            or not isinstance(shard["row_count"], int)
            or shard["row_count"] < 1
            or isinstance(shard["raw_bytes"], bool)
            or not isinstance(shard["raw_bytes"], int)
            or shard["raw_bytes"] < 1
            or isinstance(shard["stored_bytes"], bool)
            or not isinstance(shard["stored_bytes"], int)
            or shard["stored_bytes"] < 1
            or shard["compression"] != "deflate-6"
        ):
            raise AI42DatasetError(f"invalid shard metadata for {name!r}")
    if [shard["name"] for shard in manifest["shards"]] != sorted(shard_names):
        raise AI42DatasetError("manifest shards must be canonically ordered")
    match_ids: list[str] = []
    for entry in manifest["matches"]:
        if not isinstance(entry, Mapping):
            raise AI42DatasetError("manifest match entry must be an object")
        _exact_fields(entry, _MATCH_FIELDS, "manifest.match")
        match_id = entry["match_id"]
        if not isinstance(match_id, str) or not match_id:
            raise AI42DatasetError("manifest match_id must be a non-empty string")
        match_ids.append(match_id)
        if entry["split"] not in {"train", "validation"} or split.get(match_id) != entry["split"]:
            raise AI42DatasetError(f"invalid or non-deterministic split for {match_id!r}")
        if entry["shard"] not in shard_names:
            raise AI42DatasetError(f"match {match_id!r} references a missing shard")
        if isinstance(entry["row_offset"], bool) or not isinstance(entry["row_offset"], int) or entry["row_offset"] < 0:
            raise AI42DatasetError(f"invalid row_offset for {match_id!r}")
        if isinstance(entry["tick_count"], bool) or not isinstance(entry["tick_count"], int) or entry["tick_count"] < 1:
            raise AI42DatasetError(f"invalid tick_count for {match_id!r}")
        if (
            not isinstance(entry["ticks"], list)
            or not entry["ticks"]
            or any(isinstance(value, bool) or not isinstance(value, int) for value in entry["ticks"])
            or tuple(entry["ticks"]) != tuple(range(entry["ticks"][0], entry["ticks"][0] + entry["tick_count"]))
        ):
            raise AI42DatasetError(f"match {match_id!r} ticks are not contiguous")
        if isinstance(entry["seed"], bool) or not isinstance(entry["seed"], int):
            raise AI42DatasetError(f"match {match_id!r} seed is invalid")
        if not isinstance(entry["scenario"], str) or not entry["scenario"]:
            raise AI42DatasetError(f"match {match_id!r} scenario is invalid")
        try:
            _metadata_controllers(entry["controller_by_slot"])
            _metadata_roster(entry["roster_ids"], entry["hero_ids"])
            _metadata_sides(entry["side_by_slot"])
        except AI42DatasetError as exc:
            raise AI42DatasetError(f"match {match_id!r} provenance is invalid: {exc}") from exc
        if (
            not isinstance(entry["hero_ids"], list)
            or any(not isinstance(value, str) or not value for value in entry["hero_ids"])
            or len(entry["hero_ids"]) != HERO_COUNT
            or len(set(entry["hero_ids"])) != HERO_COUNT
        ):
            raise AI42DatasetError(f"match {match_id!r} must contain ten unique hero IDs")
        for field in ("trajectory_ids", "trajectory_hashes"):
            if not isinstance(entry[field], list) or len(entry[field]) != HERO_COUNT:
                raise AI42DatasetError(f"match {match_id!r} has incomplete {field}")
            if field == "trajectory_ids" and any(not isinstance(value, str) or not value for value in entry[field]):
                raise AI42DatasetError(f"match {match_id!r} has invalid trajectory IDs")
            if field == "trajectory_hashes":
                for index, value in enumerate(entry[field]):
                    _hash_hex(value, f"match {match_id}.trajectory_hashes[{index}]")
        for field in ("recurrent_parent_ids", "recurrent_boundary_ids"):
            rows = entry[field]
            if len(rows) != entry["tick_count"] or any(len(row) != HERO_COUNT for row in rows):
                raise AI42DatasetError(f"match {match_id!r} has incomplete recurrent IDs")
            for tick, row in enumerate(rows):
                if any(not isinstance(value, str) or not value for value in row):
                    raise AI42DatasetError(f"match {match_id!r} has invalid recurrent IDs")
                if field == "recurrent_parent_ids" and any(
                    row[hero] == entry["recurrent_boundary_ids"][tick][hero]
                    for hero in range(HERO_COUNT)
                ):
                    raise AI42DatasetError(f"match {match_id!r} has identical recurrent parent/boundary IDs")
                if field == "recurrent_parent_ids" and tick and any(
                    row[hero] != entry["recurrent_boundary_ids"][tick - 1][hero]
                    for hero in range(HERO_COUNT)
                ):
                    raise AI42DatasetError(f"match {match_id!r} has non-contiguous recurrent IDs")
    if match_ids != sorted(match_ids) or len(set(match_ids)) != len(match_ids):
        raise AI42DatasetError("manifest matches must be unique and canonically ordered")
    expected_shard_matches = {shard["name"]: shard["match_ids"] for shard in manifest["shards"]}
    for name, ids in expected_shard_matches.items():
        if ids != [entry["match_id"] for entry in manifest["matches"] if entry["shard"] == name]:
            raise AI42DatasetError(f"manifest shard match ordering mismatch for {name}")
    if len({entry["match_id"] for entry in manifest["matches"] if entry["split"] == "train"} & {entry["match_id"] for entry in manifest["matches"] if entry["split"] == "validation"}):
        raise AI42DatasetError("train and validation splits leak match IDs")


def load_dataset(
    root: str | os.PathLike[str],
    *,
    expected_runtime_manifest: Mapping[str, Any] | Any | None = None,
    expected_runtime_manifest_hash: str | bytes | None = None,
) -> AI42Dataset:
    """Load and strictly verify one published dataset."""

    path = Path(root)
    manifest_path = path / MANIFEST_FILENAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise AI42DatasetError(f"cannot read {manifest_path}: {exc}") from exc
    manifest = _strict_json(manifest_bytes, str(manifest_path))
    if not isinstance(manifest, Mapping):
        raise AI42DatasetError("manifest root must be an object")
    supplied_manifest_hash = _hash_hex(manifest.get("manifest_hash", ""), "manifest.manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if hash_payload(unsigned) != supplied_manifest_hash:
        raise AI42DatasetError("manifest_hash mismatch")
    if manifest.get("shard_schema_version") in {"AI42-go-shard-v1", "AI42-go-shard-v2"}:
        from .go_shard_ai42 import load_go_dataset

        return load_go_dataset(
            path,
            expected_runtime_manifest=expected_runtime_manifest,
            expected_runtime_manifest_hash=expected_runtime_manifest_hash,
        )  # type: ignore[return-value]
    expected_hash = expected_runtime_manifest_hash
    if expected_runtime_manifest is not None:
        expected_mapping = _as_mapping(expected_runtime_manifest, "expected_runtime_manifest")
        actual_expected_hash = _runtime_manifest_hash(expected_mapping)
        if expected_hash is not None and _hash_hex(expected_hash, "expected_runtime_manifest_hash") != actual_expected_hash:
            raise AI42DatasetError("expected runtime manifest/hash disagree")
        expected_hash = actual_expected_hash
    _validate_manifest(manifest, expected_hash)
    shard_by_name = {shard["name"]: shard for shard in manifest["shards"]}
    for name in sorted(shard_by_name):
        shard_path = path / name
        try:
            payload = shard_path.read_bytes()
        except OSError as exc:
            raise AI42DatasetError(f"cannot read {shard_path}: {exc}") from exc
        if _sha256_bytes(payload) != shard_by_name[name]["sha256"]:
            raise AI42DatasetError(f"shard hash mismatch for {name}")
        arrays = _read_npz(payload, str(shard_path))
        if len(payload) != shard_by_name[name]["stored_bytes"]:
            raise AI42DatasetError(f"stored shard size mismatch for {name}")
        rows = _validate_shard_arrays(arrays, str(shard_path))
        if rows != shard_by_name[name]["row_count"]:
            raise AI42DatasetError(f"shard row_count mismatch for {name}")
        for entry in (item for item in manifest["matches"] if item["shard"] == name):
            start = entry["row_offset"]
            end = start + entry["tick_count"]
            if end > arrays["step"].shape[0]:
                raise AI42DatasetError(f"match {entry['match_id']!r} exceeds its shard")
            steps = tuple(int(value) for value in arrays["step"][start:end])
            if steps != tuple(entry["ticks"]):
                raise AI42DatasetError(f"match {entry['match_id']!r} tick data mismatch")
            done = arrays["done"][start:end]
            if done.shape != (entry["tick_count"],) or np.any(done[:-1] != 0) or done[-1] != 1:
                raise AI42DatasetError(f"match {entry['match_id']!r} has an invalid terminal boundary")
            for array_name in _ACTION_ARRAY_NAMES:
                _validate_action_array(arrays[array_name][start:end], f"{entry['match_id']}:{array_name}")
    for shard in manifest["shards"]:
        cursor = 0
        for entry in (item for item in manifest["matches"] if item["shard"] == shard["name"]):
            if entry["row_offset"] != cursor:
                raise AI42DatasetError(f"shard {shard['name']} has a missing or overlapping row range")
            cursor += entry["tick_count"]
        if cursor != shard["row_count"]:
            raise AI42DatasetError(f"shard {shard['name']} has missing or extra rows")
    return AI42Dataset(path, manifest, {})


read_dataset = load_dataset
validate_dataset = load_dataset


def write_dataset(
    root: str | os.PathLike[str],
    matches: Iterable[AI42DatasetMatch | AI42BridgeOutput],
    *,
    runtime_manifest: Mapping[str, Any] | Any | None = None,
    runtime_manifest_hash: str | bytes | None = None,
    shard_size: int = 1,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
) -> AI42Dataset:
    writer = AI42DatasetWriter(
        root,
        runtime_manifest=runtime_manifest,
        runtime_manifest_hash=runtime_manifest_hash,
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    for match in matches:
        writer.add_match(match)
    return writer.write()


__all__ = [
    "AI42Dataset",
    "AI42DatasetError",
    "AI42DatasetMatch",
    "AI42RawMatch",
    "AI42DatasetSpool",
    "AI42DatasetStaging",
    "AI42DatasetWriter",
    "DATASET_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "SHARD_SCHEMA_VERSION",
    "DatasetError",
    "load_dataset",
    "publish_staged_dataset",
    "read_dataset",
    "validate_dataset",
    "write_dataset",
]
