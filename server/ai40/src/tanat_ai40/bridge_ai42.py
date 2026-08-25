"""Torch-free v13 adapter from environment results to complete AI-42 records.

The adapter is intentionally a boundary module: it accepts only the public
v13 result and the caller's submitted actions/recurrent/outcome evidence, and
it emits immutable :mod:`trajectory_ai42` objects.  It never imports a model,
training module, or privileged server state.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .env import (
    ACTION_DTYPE,
    AI42_PROTOCOL_VERSION,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    HERO_COUNT,
    StepResult,
)
from .trajectory_ai42 import (
    AI42_TRAJECTORY_SCHEMA_HASH,
    Action,
    Outcome,
    RejectionReason,
    TeacherMatch,
    TeacherTrajectory,
    TeacherTrajectoryRecord,
    canonical_json_bytes,
    hash_payload,
    validate_integrity,
    validate_replay,
)


TEACHER_STATUS_NONE = 0
TEACHER_STATUS_ACTION = 1
TEACHER_STATUS_WAIT = 2
TEACHER_STATUS_HOLD = 3
TEACHER_STATUS_CANCEL = 4
TEACHER_STATUS_UNAVAILABLE = 5

REJECTION_REASON_NONE = 0
REJECTION_REASON_MASKED = 1
REJECTION_REASON_INVALID = 2
REJECTION_REASON_SERVER_REJECTED = 3
REJECTION_REASON_SAFETY = 4
REJECTION_REASON_TIMEOUT = 5
REJECTION_REASON_POLICY_ERROR = 6
REJECTION_REASON_UNKNOWN = 255

_STATUS_NAMES = {
    TEACHER_STATUS_NONE: "none",
    TEACHER_STATUS_ACTION: "action",
    TEACHER_STATUS_WAIT: "wait",
    TEACHER_STATUS_HOLD: "hold",
    TEACHER_STATUS_CANCEL: "cancel",
    TEACHER_STATUS_UNAVAILABLE: "unavailable",
}
_REASON_NAMES = {
    REJECTION_REASON_NONE: RejectionReason.NONE.value,
    REJECTION_REASON_MASKED: RejectionReason.MASKED.value,
    REJECTION_REASON_INVALID: RejectionReason.INVALID.value,
    REJECTION_REASON_SERVER_REJECTED: RejectionReason.SERVER_REJECTED.value,
    REJECTION_REASON_SAFETY: RejectionReason.SAFETY.value,
    REJECTION_REASON_TIMEOUT: RejectionReason.TIMEOUT.value,
    REJECTION_REASON_POLICY_ERROR: RejectionReason.POLICY_ERROR.value,
    REJECTION_REASON_UNKNOWN: RejectionReason.UNKNOWN.value,
}


class AI42BridgeError(ValueError):
    """Raised when a v13 result cannot be represented without fabrication."""


def _hash_hex(value: bytes | bytearray | memoryview | str, name: str) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) != 32:
            raise AI42BridgeError(f"{name} must contain exactly 32 bytes")
        return raw.hex()
    if not isinstance(value, str) or len(value) != 64:
        raise AI42BridgeError(f"{name} must be a 64-character SHA-256 hex string")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise AI42BridgeError(f"{name} must be hexadecimal") from exc
    if len(raw) != 32 or value.lower() != value:
        raise AI42BridgeError(f"{name} must use lower-case SHA-256 hexadecimal")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _slot_values(values: Iterable[Any], name: str, *, tick: int) -> tuple[Any, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise AI42BridgeError(f"{name}[{tick}] must be an iterable of 10 slots") from exc
    if len(result) != HERO_COUNT:
        raise AI42BridgeError(f"{name}[{tick}] must contain exactly {HERO_COUNT} slots")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise AI42BridgeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AI42BridgeError(f"{name} must be an integer") from exc
    if isinstance(value, (float, np.floating)) and result != value:
        raise AI42BridgeError(f"{name} must be an integer")
    return result


def _wire_tuple(value: Any, name: str) -> tuple[int, int, int, int]:
    if isinstance(value, np.void) and value.dtype.names:
        return tuple(_integer(value[field], f"{name}.{field}") for field in ACTION_DTYPE.names)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        fields = ("kind", "target", "direction", "distance")
        if set(value) != set(fields):
            raise AI42BridgeError(f"{name} must contain exactly kind,target,direction,distance")
        return tuple(_integer(value[field], f"{name}.{field}") for field in fields)  # type: ignore[return-value]
    if all(hasattr(value, field) for field in ("kind", "target", "direction", "distance")):
        return (int(value.kind), int(value.target), int(value.direction), int(value.distance))
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise AI42BridgeError(f"{name} is not a v13 action") from exc
    if len(raw) != 4:
        raise AI42BridgeError(f"{name} must contain exactly 4 fields")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(raw))  # type: ignore[return-value]


def _wire_actions(values: Any, name: str, *, tick: int) -> tuple[tuple[int, int, int, int], ...]:
    if isinstance(values, np.ndarray):
        if values.dtype.names:
            if values.shape != (HERO_COUNT,):
                raise AI42BridgeError(f"{name}[{tick}] must have shape ({HERO_COUNT},)")
            return tuple(_wire_tuple(value, f"{name}[{tick}][{index}]") for index, value in enumerate(values))
        if values.shape != (HERO_COUNT, 4):
            raise AI42BridgeError(f"{name}[{tick}] must have shape ({HERO_COUNT}, 4)")
        return tuple(_wire_tuple(row, f"{name}[{tick}][{index}]") for index, row in enumerate(values))
    return tuple(
        _wire_tuple(value, f"{name}[{tick}][{index}]")
        for index, value in enumerate(_slot_values(values, name, tick=tick))
    )


def _action_from_wire(raw: tuple[int, int, int, int], name: str) -> Action:
    kind, target, direction, distance = raw
    if not 0 <= kind < 8:
        raise AI42BridgeError(f"{name}.kind={kind} is outside v13 action kinds")
    if not 0 <= target <= 0xFFFF or not 0 <= direction <= 0xFF or not 0 <= distance <= 0xFF:
        raise AI42BridgeError(f"{name} contains a field outside its wire range")
    if kind == 0:
        return Action("wait")
    if 3 <= kind <= 6 and distance != 0:
        raise AI42BridgeError(f"{name} skill actions cannot use navigation anchors in v13")
    navigation: dict[str, Any] = {}
    if kind == 1 or 3 <= kind <= 6:
        if direction >= 81 or distance >= 15:
            raise AI42BridgeError(f"{name} navigation fields are outside offset81/anchor15")
        if distance == 0:
            navigation["point"] = (direction % 9 - 4, direction // 9 - 4)
        else:
            navigation["anchor"] = distance
    if kind == 1:
        return Action("move", **navigation)
    if kind == 2:
        return Action("attack", target=target)
    if 3 <= kind <= 6:
        return Action("skill", target=target, skill=kind - 2, **navigation)
    return Action("teleport", target=target if target else None)


def _is_zero_wire(raw: tuple[int, int, int, int]) -> bool:
    return raw == (0, 0, 0, 0)


def _teacher_action(
    raw: tuple[int, int, int, int], status: int, parent: str, name: str,
) -> Action:
    if status not in _STATUS_NAMES:
        raise AI42BridgeError(f"{name}.status={status} is not a v13 teacher status")
    if status in (TEACHER_STATUS_NONE, TEACHER_STATUS_UNAVAILABLE):
        if not _is_zero_wire(raw):
            raise AI42BridgeError(f"{name} has bytes despite status={_STATUS_NAMES[status]}")
        return Action("unavailable")
    if status == TEACHER_STATUS_ACTION:
        action = _action_from_wire(raw, name)
        if action.kind == "wait":
            raise AI42BridgeError(f"{name} action status cannot carry a wait action")
        return action
    if not _is_zero_wire(raw):
        raise AI42BridgeError(f"{name} control status must carry a zero action payload")
    if status == TEACHER_STATUS_WAIT:
        return Action("wait")
    if status == TEACHER_STATUS_HOLD:
        return Action("hold", lineage_id=parent)
    return Action("cancel", lineage_id=parent)


def _observation_payload(result: StepResult, hero: int) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = {
        "hero": result.hero,
        "abilities": result.abilities,
        "entities": result.entities,
        "global": result.global_state,
        "entity_mask": result.entity_mask,
        "kind_mask": result.kind_mask,
        "target_mask": result.target_mask,
        "skill_target_mask": result.skill_target_mask,
    }
    if result.abilities is None:
        raise AI42BridgeError("v13 result must include abilities")
    observation = {name: np.asarray(value[hero]).tolist() for name, value in arrays.items()}
    mask = {
        name: observation[name]
        for name in ("entity_mask", "kind_mask", "target_mask", "skill_target_mask")
    }
    return observation, mask


def _validate_result(result: Any, tick: int, schema_hash: str, reward_hash: str) -> None:
    if not isinstance(result, StepResult):
        raise AI42BridgeError(f"results[{tick}] must be a v13 StepResult")
    expected = {
        "hero": (HERO_COUNT, 32),
        "abilities": (HERO_COUNT, 4, 40),
        "entities": (HERO_COUNT, 96, 16),
        "global_state": (HERO_COUNT, 32),
        "entity_mask": (HERO_COUNT, 96),
        "kind_mask": (HERO_COUNT, 8),
        "target_mask": (HERO_COUNT, 96),
        "skill_target_mask": (HERO_COUNT, 4, 96),
        "invalid": (HERO_COUNT,),
        "rewards": (HERO_COUNT,),
        "teacher_intent": (HERO_COUNT,),
        "teacher_status": (HERO_COUNT,),
        "executed_actions": (HERO_COUNT,),
        "executed_valid": (HERO_COUNT,),
        "rejection_reason": (HERO_COUNT,),
    }
    for name, shape in expected.items():
        value = getattr(result, name, None)
        if value is None or np.asarray(value).shape != shape:
            raise AI42BridgeError(f"results[{tick}].{name} must have shape {shape}")
    if getattr(result, "schema_hash", None) is None:
        raise AI42BridgeError(f"results[{tick}].schema_hash is required")
    if _hash_hex(result.schema_hash, f"results[{tick}].schema_hash") != schema_hash:
        raise AI42BridgeError(f"results[{tick}].schema_hash does not match the supplied hash")
    if getattr(result, "reward_hash", None) is None:
        raise AI42BridgeError(f"results[{tick}].reward_hash is required")
    if _hash_hex(result.reward_hash, f"results[{tick}].reward_hash") != reward_hash:
        raise AI42BridgeError(f"results[{tick}].reward_hash does not match the supplied hash")


def _outcome(value: Outcome | Mapping[str, Any], name: str) -> Outcome:
    if isinstance(value, Outcome):
        return value
    if isinstance(value, Mapping):
        try:
            return Outcome.from_dict(value, path=name)
        except ValueError as exc:
            raise AI42BridgeError(str(exc)) from exc
    raise AI42BridgeError(f"{name} must be an Outcome or exact outcome mapping")


def _ids(values: Iterable[str], name: str, *, tick: int) -> tuple[str, ...]:
    result = _slot_values(values, name, tick=tick)
    if any(not isinstance(value, str) or not value for value in result):
        raise AI42BridgeError(f"{name}[{tick}] must contain 10 non-empty IDs")
    return result  # type: ignore[return-value]


def _expected_nested(values: Any, name: str, ticks: int) -> tuple[tuple[Any, ...], ...]:
    rows = tuple(values)
    if len(rows) != ticks:
        raise AI42BridgeError(f"{name} must contain exactly {ticks} ticks")
    return tuple(_slot_values(row, name, tick=index) for index, row in enumerate(rows))


@dataclass(frozen=True, slots=True)
class AI42BridgeOutput:
    """Immutable complete match output plus replay evidence."""

    match: TeacherMatch
    protocol_version: int
    schema_hash: str
    reward_hash: str
    trajectory_schema_hash: str
    manifest_hash: str
    manifest: Mapping[str, Any]
    observation_payloads: tuple[tuple[Mapping[str, Any], ...], ...]
    mask_payloads: tuple[tuple[Mapping[str, Any], ...], ...]
    action_payloads: tuple[tuple[Mapping[str, Any], ...], ...]
    executed_actions: tuple[tuple[Action, ...], ...]
    outcomes: tuple[tuple[Outcome, ...], ...]
    recurrent_parent_ids: tuple[tuple[str, ...], ...]
    recurrent_boundary_ids: tuple[tuple[str, ...], ...]

    @property
    def trajectories(self) -> tuple[TeacherTrajectory, ...]:
        return self.match.trajectories

    def validate_integrity(self):
        return tuple(validate_integrity(trajectory) for trajectory in self.trajectories)

    def validate_replay(self, expected_outputs: Mapping[str, Any] | None = None, **kwargs: Any):
        """Run strict public trajectory replay validation for all 10 heroes.

        Every expected output is mandatory.  Stored adapter payloads are not
        silently reused as an independent replay source.
        """

        supplied = dict(expected_outputs or {})
        supplied.update(kwargs)
        required = {
            "observation_payloads", "mask_payloads", "action_payloads",
            "executed_actions", "outcomes", "recurrent_parent_ids",
            "recurrent_boundary_ids",
        }
        missing = sorted(required - set(supplied))
        if missing:
            raise AI42BridgeError("strict replay requires " + ", ".join(missing))
        ticks = len(self.observation_payloads)
        observations = _expected_nested(supplied["observation_payloads"], "observation_payloads", ticks)
        masks = _expected_nested(supplied["mask_payloads"], "mask_payloads", ticks)
        actions = _expected_nested(supplied["action_payloads"], "action_payloads", ticks)
        executed = _expected_nested(supplied["executed_actions"], "executed_actions", ticks)
        outcomes = _expected_nested(supplied["outcomes"], "outcomes", ticks)
        parents = _expected_nested(supplied["recurrent_parent_ids"], "recurrent_parent_ids", ticks)
        boundaries = _expected_nested(supplied["recurrent_boundary_ids"], "recurrent_boundary_ids", ticks)
        results = []
        for hero, trajectory in enumerate(self.trajectories):
            results.append(validate_replay(
                trajectory,
                observation_payloads=[row[hero] for row in observations],
                mask_payloads=[row[hero] for row in masks],
                action_payloads=[row[hero] for row in actions],
                executed_actions=[row[hero] for row in executed],
                outcomes=[row[hero] for row in outcomes],
                recurrent_parent_ids=[row[hero] for row in parents],
                recurrent_boundary_ids=[row[hero] for row in boundaries],
            ))
        return tuple(results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "schema_hash": self.schema_hash,
            "reward_hash": self.reward_hash,
            "trajectory_schema_hash": self.trajectory_schema_hash,
            "manifest_hash": self.manifest_hash,
            "manifest": dict(self.manifest),
            "match_id": self.match.match_id,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }


def build_ai42_trajectory(
    results: Iterable[StepResult],
    submitted_actions: Iterable[Any],
    previous_recurrent_boundaries: Iterable[Iterable[str]],
    outcomes: Iterable[Iterable[Outcome | Mapping[str, Any]]],
    *,
    recurrent_boundaries: Iterable[Iterable[str]] | None = None,
    hero_ids: Iterable[str],
    match_id: str,
    trajectory_schema_hash: bytes | str,
    schema_hash: bytes | str,
    reward_hash: bytes | str,
    manifest: Mapping[str, Any],
    manifest_hash: bytes | str,
) -> AI42BridgeOutput:
    """Build one complete immutable 10-hero v13 teacher match.

    All protocol/artifact hashes, the manifest and both recurrent boundary
    sides are explicit inputs.  A missing authoritative field is an error;
    this adapter never derives a teacher action or outcome from observations.
    """

    if AI42_PROTOCOL_VERSION != 13:
        raise AI42BridgeError("AI-42 bridge constants are inconsistent")
    schema = _hash_hex(schema_hash, "schema_hash")
    reward = _hash_hex(reward_hash, "reward_hash")
    trajectory_schema = _hash_hex(trajectory_schema_hash, "trajectory_schema_hash")
    supplied_manifest_hash = _hash_hex(manifest_hash, "manifest_hash")
    if schema != AI42_SCHEMA_HASH.hex():
        raise AI42BridgeError("schema_hash is not the active derived v13 schema")
    if reward != AI42_REWARD_HASH.hex():
        raise AI42BridgeError("reward_hash is not strategic V5")
    if trajectory_schema != AI42_TRAJECTORY_SCHEMA_HASH:
        raise AI42BridgeError("trajectory_schema_hash is not the public AI-42 trajectory schema")
    if not isinstance(manifest, Mapping) or not manifest:
        raise AI42BridgeError("manifest must be a non-empty explicit mapping")
    try:
        manifest_bytes = canonical_json_bytes(manifest)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AI42BridgeError(f"manifest is not canonicalizable: {exc}") from exc
    if hash_payload(manifest_bytes) != supplied_manifest_hash:
        raise AI42BridgeError("manifest_hash does not match manifest")
    heroes = tuple(hero_ids)
    if len(heroes) != HERO_COUNT or len(set(heroes)) != HERO_COUNT or any(not isinstance(hero, str) or not hero for hero in heroes):
        raise AI42BridgeError("hero_ids must contain exactly 10 unique non-empty IDs")
    result_rows = tuple(results)
    if not result_rows:
        raise AI42BridgeError("at least one v13 result tick is required")
    for tick, result in enumerate(result_rows):
        _validate_result(result, tick, schema, reward)
    projected_rows = _expected_nested(submitted_actions, "submitted_actions", len(result_rows))
    parent_rows = _expected_nested(previous_recurrent_boundaries, "previous_recurrent_boundaries", len(result_rows))
    outcome_rows = _expected_nested(outcomes, "outcomes", len(result_rows))
    if recurrent_boundaries is None:
        raise AI42BridgeError("recurrent_boundaries is required")
    boundary_rows = _expected_nested(recurrent_boundaries, "recurrent_boundaries", len(result_rows))
    observation_rows: list[tuple[Mapping[str, Any], ...]] = []
    mask_rows: list[tuple[Mapping[str, Any], ...]] = []
    action_rows: list[tuple[Mapping[str, Any], ...]] = []
    executed_rows: list[tuple[Action, ...]] = []
    normalized_outcomes: list[tuple[Outcome, ...]] = []
    records_by_hero: list[list[TeacherTrajectoryRecord]] = [[] for _ in range(HERO_COUNT)]
    for tick, result in enumerate(result_rows):
        submitted = _wire_actions(projected_rows[tick], "submitted_actions", tick=tick)
        status = tuple(_integer(value, f"results[{tick}].teacher_status[{hero}]") for hero, value in enumerate(result.teacher_status))
        valid = tuple(_integer(value, f"results[{tick}].executed_valid[{hero}]") for hero, value in enumerate(result.executed_valid))
        reasons = tuple(_integer(value, f"results[{tick}].rejection_reason[{hero}]") for hero, value in enumerate(result.rejection_reason))
        teacher_raw = tuple(_wire_tuple(value, f"results[{tick}].teacher_intent[{hero}]") for hero, value in enumerate(result.teacher_intent))
        executed_raw = tuple(_wire_tuple(value, f"results[{tick}].executed_actions[{hero}]") for hero, value in enumerate(result.executed_actions))
        tick_observations: list[Mapping[str, Any]] = []
        tick_masks: list[Mapping[str, Any]] = []
        tick_actions: list[Mapping[str, Any]] = []
        tick_executed: list[Action] = []
        tick_outcomes: list[Outcome] = []
        for hero in range(HERO_COUNT):
            if valid[hero] not in (0, 1):
                raise AI42BridgeError(f"results[{tick}].executed_valid[{hero}] must be 0 or 1")
            if reasons[hero] not in _REASON_NAMES:
                raise AI42BridgeError(f"results[{tick}].rejection_reason[{hero}] is unknown")
            if valid[hero] == 1 and reasons[hero] != REJECTION_REASON_NONE:
                raise AI42BridgeError(f"results[{tick}][{hero}] accepted action has a rejection reason")
            if valid[hero] == 0 and reasons[hero] == REJECTION_REASON_NONE:
                raise AI42BridgeError(f"results[{tick}][{hero}] rejected action has reason none")
            parent = parent_rows[tick][hero]
            boundary = boundary_rows[tick][hero]
            if not isinstance(parent, str) or not parent or not isinstance(boundary, str) or not boundary or parent == boundary:
                raise AI42BridgeError(f"results[{tick}][{hero}] has invalid recurrent boundaries")
            if tick and parent != boundary_rows[tick - 1][hero]:
                raise AI42BridgeError(f"results[{tick}][{hero}] recurrent parent does not match prior boundary")
            original = _teacher_action(teacher_raw[hero], status[hero], parent, f"results[{tick}].teacher_intent[{hero}]")
            projected = _action_from_wire(submitted[hero], f"submitted_actions[{tick}][{hero}]")
            if valid[hero]:
                executed = _action_from_wire(executed_raw[hero], f"results[{tick}].executed_actions[{hero}]")
            else:
                if not _is_zero_wire(executed_raw[hero]):
                    raise AI42BridgeError(f"results[{tick}].executed_actions[{hero}] is non-zero while rejected")
                executed = Action("wait")
            observation, mask = _observation_payload(result, hero)
            outcome = _outcome(outcome_rows[tick][hero], f"outcomes[{tick}][{hero}]")
            tick_observations.append(_freeze(observation))
            tick_masks.append(_freeze(mask))
            tick_actions.append(_freeze(projected.to_dict()))
            tick_executed.append(executed)
            tick_outcomes.append(outcome)
            records_by_hero[hero].append(TeacherTrajectoryRecord.from_payload(
                tick=int(result.step), sequence=tick, hero_id=heroes[hero],
                recurrent_parent_id=parent, recurrent_boundary_id=boundary,
                observation=observation, mask=mask,
                original_ai30_intent=original,
                projected_neural_action=projected,
                executed_action=executed, valid=bool(valid[hero]),
                rejection_reason=_REASON_NAMES[reasons[hero]], outcome=outcome,
                action=projected.to_dict(),
            ))
        observation_rows.append(tuple(tick_observations))
        mask_rows.append(tuple(tick_masks))
        action_rows.append(tuple(tick_actions))
        executed_rows.append(tuple(tick_executed))
        normalized_outcomes.append(tuple(tick_outcomes))
    ticks = tuple(int(result.step) for result in result_rows)
    trajectories = tuple(
        TeacherTrajectory(
            trajectory_id=f"{match_id}:hero:{heroes[hero]}", match_id=match_id,
            hero_id=heroes[hero], records=tuple(records_by_hero[hero]), expected_ticks=ticks,
        ) for hero in range(HERO_COUNT)
    )
    match = TeacherMatch(match_id=match_id, trajectories=trajectories, expected_hero_ids=heroes)
    output = AI42BridgeOutput(
        match=match, protocol_version=AI42_PROTOCOL_VERSION, schema_hash=schema,
        reward_hash=reward, trajectory_schema_hash=trajectory_schema,
        manifest_hash=supplied_manifest_hash, manifest=_freeze(manifest),
        observation_payloads=tuple(observation_rows), mask_payloads=tuple(mask_rows),
        action_payloads=tuple(action_rows), executed_actions=tuple(executed_rows),
        outcomes=tuple(normalized_outcomes),
        recurrent_parent_ids=tuple(tuple(row) for row in parent_rows),
        recurrent_boundary_ids=tuple(tuple(row) for row in boundary_rows),
    )
    output.validate_integrity()
    return output


build_trajectory = build_ai42_trajectory


__all__ = [
    "AI42BridgeError", "AI42BridgeOutput", "REJECTION_REASON_INVALID",
    "REJECTION_REASON_MASKED", "REJECTION_REASON_NONE", "REJECTION_REASON_POLICY_ERROR",
    "REJECTION_REASON_SAFETY", "REJECTION_REASON_SERVER_REJECTED", "REJECTION_REASON_TIMEOUT",
    "REJECTION_REASON_UNKNOWN", "TEACHER_STATUS_ACTION", "TEACHER_STATUS_CANCEL",
    "TEACHER_STATUS_HOLD", "TEACHER_STATUS_NONE", "TEACHER_STATUS_UNAVAILABLE",
    "TEACHER_STATUS_WAIT", "build_ai42_trajectory", "build_trajectory",
]
