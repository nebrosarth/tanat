"""Capture adapter for already-created AssaultVectorEnv v13 results.

This module only consumes results and caller-supplied action/recurrent/outcome
evidence.  It never constructs or launches an environment and never imports
torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bridge_ai42 import build_ai42_trajectory
from .dataset_ai42 import AI42DatasetError, AI42DatasetMatch, AI42RawMatch, _wire_actions
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
from .trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, Outcome, canonical_json_bytes, hash_payload


def _manifest_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping) or not value:
        raise AI42DatasetError("runtime_manifest must be a non-empty mapping")
    return value


def _strict_array(value: Any, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise AI42DatasetError(f"{name} must have dtype {dtype}, shape {shape}; got {array.dtype}, {array.shape}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise AI42DatasetError(f"{name} contains non-finite values")
    return np.array(array, copy=True)


def _validate_result(result: Any, index: int) -> StepResult:
    if not isinstance(result, StepResult):
        raise AI42DatasetError(f"results[{index}] must be a v13 StepResult")
    if isinstance(result.step, bool) or not isinstance(result.step, (int, np.integer)) or int(result.step) < 0:
        raise AI42DatasetError(f"results[{index}].step must be a non-negative integer")
    if isinstance(result.done, (np.bool_, bool)) is False:
        raise AI42DatasetError(f"results[{index}].done must be boolean")
    if isinstance(result.winner, bool) or not isinstance(result.winner, (int, np.integer)):
        raise AI42DatasetError(f"results[{index}].winner must be an integer")
    if not isinstance(result.elapsed, (int, float, np.integer, np.floating)) or not np.isfinite(result.elapsed):
        raise AI42DatasetError(f"results[{index}].elapsed must be finite")
    fields = {
        "invalid": ((HERO_COUNT,), np.dtype("u1")),
        "rewards": ((HERO_COUNT,), np.dtype("<f4")),
        "hero": ((HERO_COUNT, HERO_FEATURES), np.dtype("<f4")),
        "abilities": ((HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES), np.dtype("<f4")),
        "entities": ((HERO_COUNT, MAX_ENTITIES, ENTITY_FEATURES), np.dtype("<f4")),
        "global_state": ((HERO_COUNT, GLOBAL_FEATURES), np.dtype("<f4")),
        "entity_mask": ((HERO_COUNT, MAX_ENTITIES), np.dtype("u1")),
        "kind_mask": ((HERO_COUNT, ACTION_KINDS), np.dtype("u1")),
        "target_mask": ((HERO_COUNT, MAX_ENTITIES), np.dtype("u1")),
        "skill_target_mask": ((HERO_COUNT, ABILITY_COUNT, MAX_ENTITIES), np.dtype("u1")),
        "teacher_intent": ((HERO_COUNT,), np.dtype(ACTION_DTYPE)),
        "teacher_status": ((HERO_COUNT,), np.dtype("u1")),
        "executed_actions": ((HERO_COUNT,), np.dtype(ACTION_DTYPE)),
        "executed_valid": ((HERO_COUNT,), np.dtype("u1")),
        "rejection_reason": ((HERO_COUNT,), np.dtype("u1")),
    }
    if result.abilities is None:
        raise AI42DatasetError(f"results[{index}].abilities is required")
    for name, (shape, dtype) in fields.items():
        value = getattr(result, name, None)
        if value is None:
            raise AI42DatasetError(f"results[{index}].{name} is required")
        array = _strict_array(value, dtype, shape, f"results[{index}].{name}")
        if name in {"invalid", "entity_mask", "kind_mask", "target_mask", "skill_target_mask", "executed_valid"} and np.any((array != 0) & (array != 1)):
            raise AI42DatasetError(f"results[{index}].{name} must contain only zero/one values")
    if result.schema_hash is None or bytes(result.schema_hash) != AI42_SCHEMA_HASH:
        raise AI42DatasetError(f"results[{index}].schema_hash does not match v13")
    if result.reward_hash is None or bytes(result.reward_hash) != AI42_REWARD_HASH:
        raise AI42DatasetError(f"results[{index}].reward_hash does not match v13")
    return result


def _canonical_ticks(results: Sequence[StepResult], expected_ticks: int | Sequence[int] | None) -> tuple[int, ...]:
    if not results:
        raise AI42DatasetError("at least one v13 result tick is required")
    ticks = tuple(int(result.step) for result in results)
    if any(value != ticks[0] + index for index, value in enumerate(ticks)):
        raise AI42DatasetError(f"result steps are not contiguous: {ticks!r}")
    if expected_ticks is not None:
        if isinstance(expected_ticks, int) and not isinstance(expected_ticks, bool):
            if len(results) != expected_ticks:
                raise AI42DatasetError(f"expected {expected_ticks} ticks, got {len(results)}")
        else:
            expected = tuple(int(value) for value in expected_ticks)
            if ticks != expected:
                raise AI42DatasetError(f"result ticks {ticks!r} do not equal expected_ticks {expected!r}")
    return ticks


def collect_ai42_match(
    results: Iterable[StepResult],
    submitted_actions: Iterable[Any],
    previous_recurrent_boundaries: Iterable[Iterable[str]],
    outcomes: Iterable[Iterable[Outcome | Mapping[str, Any]]],
    *,
    recurrent_boundaries: Iterable[Iterable[str]],
    hero_ids: Iterable[str],
    match_id: str,
    runtime_manifest: Mapping[str, Any] | Any,
    runtime_manifest_hash: bytes | str | None = None,
    expected_ticks: int | Sequence[int] | None = None,
    seed: int | None = None,
    scenario: str | None = None,
    controller_by_slot: Iterable[int] | None = None,
    roster_ids: Iterable[Any] | None = None,
    side_by_slot: Iterable[int] | None = None,
) -> AI42DatasetMatch:
    """Turn supplied v13 results into one validated durable match capture."""

    if AI42_PROTOCOL_VERSION != 13:
        raise AI42DatasetError("active runtime protocol is not v13")
    manifest = _manifest_mapping(runtime_manifest)
    computed_manifest_hash = hash_payload(manifest)
    supplied_manifest_hash = computed_manifest_hash if runtime_manifest_hash is None else runtime_manifest_hash
    if isinstance(supplied_manifest_hash, (bytes, bytearray, memoryview)):
        supplied_manifest_hash = bytes(supplied_manifest_hash).hex()
    if supplied_manifest_hash != computed_manifest_hash:
        raise AI42DatasetError("runtime_manifest_hash does not match runtime_manifest")
    result_rows = tuple(_validate_result(result, index) for index, result in enumerate(results))
    if any(result.done for result in result_rows[:-1]) or not result_rows[-1].done:
        raise AI42DatasetError("AI42 match capture must end at its only terminal tick")
    ticks = _canonical_ticks(result_rows, expected_ticks)
    submitted_rows = tuple(tuple(row) for row in submitted_actions)
    parent_rows = tuple(tuple(row) for row in previous_recurrent_boundaries)
    boundary_rows = tuple(tuple(row) for row in recurrent_boundaries)
    outcome_rows = tuple(tuple(row) for row in outcomes)
    output = build_ai42_trajectory(
        result_rows,
        submitted_rows,
        parent_rows,
        outcome_rows,
        recurrent_boundaries=boundary_rows,
        hero_ids=hero_ids,
        match_id=match_id,
        trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
        schema_hash=AI42_SCHEMA_HASH,
        reward_hash=AI42_REWARD_HASH,
        manifest=manifest,
        manifest_hash=supplied_manifest_hash,
    )
    projected = _wire_actions(submitted_rows, "submitted_actions", len(result_rows))
    teacher_status = np.stack([np.array(result.teacher_status, copy=True) for result in result_rows]).astype(np.uint8, copy=False)
    teacher_action = np.stack([np.array(result.teacher_intent, copy=True) for result in result_rows])
    executed_action = np.stack([np.array(result.executed_actions, copy=True) for result in result_rows])
    executed_valid = np.stack([np.array(result.executed_valid, copy=True) for result in result_rows]).astype(np.uint8, copy=False)
    rejection_reason = np.stack([np.array(result.rejection_reason, copy=True) for result in result_rows]).astype(np.uint8, copy=False)
    rewards = np.stack([np.array(result.rewards, copy=True) for result in result_rows])
    done = np.asarray([int(result.done) for result in result_rows], dtype=np.uint8)
    winner = np.asarray([int(result.winner) for result in result_rows], dtype=np.dtype("<i4"))
    elapsed = np.asarray([result.elapsed for result in result_rows], dtype=np.dtype("<f4"))
    invalid = np.stack([np.array(result.invalid, copy=True) for result in result_rows]).astype(np.uint8, copy=False)
    return AI42DatasetMatch(
        output=output,
        teacher_status=teacher_status,
        teacher_action=teacher_action,
        projected_action=projected,
        executed_action=executed_action,
        executed_valid=executed_valid,
        rejection_reason=rejection_reason,
        rewards=rewards,
        done=done,
        winner=winner,
        elapsed=elapsed,
        invalid=invalid,
        seed=seed,
        scenario=scenario,
        controller_by_slot=None if controller_by_slot is None else tuple(controller_by_slot),
        roster_ids=None if roster_ids is None else tuple(roster_ids),
        side_by_slot=None if side_by_slot is None else tuple(side_by_slot),
    )


collect_match = collect_ai42_match


@dataclass(slots=True)
class AI42Collector:
    """Incremental collector for results returned by an existing vector env."""

    match_id: str
    hero_ids: tuple[str, ...]
    runtime_manifest: Mapping[str, Any] | Any
    expected_ticks: int | Sequence[int] | None = None
    runtime_manifest_hash: bytes | str | None = None
    seed: int | None = None
    scenario: str | None = None
    controller_by_slot: tuple[int, ...] | None = None
    roster_ids: tuple[Any, ...] | None = None
    side_by_slot: tuple[int, ...] | None = None
    _results: list[StepResult] = field(default_factory=list, init=False)
    _submitted: list[Any] = field(default_factory=list, init=False)
    _parents: list[Iterable[str]] = field(default_factory=list, init=False)
    _boundaries: list[Iterable[str]] = field(default_factory=list, init=False)
    _outcomes: list[Iterable[Outcome | Mapping[str, Any]]] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)
    _fast_mode: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.hero_ids = tuple(self.hero_ids)
        if len(self.hero_ids) != HERO_COUNT or len(set(self.hero_ids)) != HERO_COUNT:
            raise AI42DatasetError("hero_ids must contain ten unique IDs")

    def record_tick(
        self,
        result: StepResult,
        submitted_actions: Any,
        previous_recurrent_boundaries: Iterable[str],
        recurrent_boundaries: Iterable[str],
        outcomes: Iterable[Outcome | Mapping[str, Any]],
    ) -> None:
        if self._finished:
            raise AI42DatasetError("collector is already finalized")
        _validate_result(result, len(self._results))
        # Capture immutable snapshots. AssaultVectorEnv reuses NumPy-backed
        # result buffers, and callers may reuse action/recurrent containers.
        self._results.append(copy.deepcopy(result))
        self._submitted.append(copy.deepcopy(submitted_actions))
        self._parents.append(tuple(previous_recurrent_boundaries))
        self._boundaries.append(tuple(recurrent_boundaries))
        self._outcomes.append(tuple(copy.deepcopy(tuple(outcomes))))

    add_tick = record_tick
    add_step = record_tick

    def record_tick_fast(
        self,
        result: StepResult,
        submitted_actions: Any,
        previous_recurrent_boundaries: Iterable[str],
        recurrent_boundaries: Iterable[str],
    ) -> None:
        """Snapshot one tick for the vectorized production finalizer.

        Full v13 field/replay validation runs once over stacked arrays at
        finish_fast; this avoids constructing ten trajectory dataclasses and
        Outcome objects for every policy tick.
        """

        if self._finished:
            raise AI42DatasetError("collector is already finalized")
        if self._results and not self._fast_mode:
            raise AI42DatasetError("cannot mix regular and fast collector modes")
        if not isinstance(result, StepResult):
            raise AI42DatasetError("result must be a v13 StepResult")
        self._fast_mode = True
        self._results.append(copy.deepcopy(result))
        self._submitted.append(copy.deepcopy(submitted_actions))
        self._parents.append(tuple(previous_recurrent_boundaries))
        self._boundaries.append(tuple(recurrent_boundaries))

    add_tick_fast = record_tick_fast

    def finish_fast(self) -> AI42DatasetMatch:
        if self._finished:
            raise AI42DatasetError("collector is already finalized")
        if not self._fast_mode:
            raise AI42DatasetError("collector has no fast ticks")
        if not self._results:
            raise AI42DatasetError("collector has no policy ticks")
        if not bool(self._results[-1].done):
            raise AI42DatasetError("collector can finalize only a terminal match")
        self._finished = True
        manifest = _manifest_mapping(self.runtime_manifest)
        manifest_hash = hash_payload(manifest)
        return AI42DatasetMatch(
            raw_capture=AI42RawMatch(
                results=tuple(self._results),
                submitted_actions=tuple(self._submitted),
                previous_recurrent_boundaries=tuple(tuple(row) for row in self._parents),
                recurrent_boundaries=tuple(tuple(row) for row in self._boundaries),
                hero_ids=self.hero_ids,
                match_id=self.match_id,
                runtime_manifest=manifest,
                runtime_manifest_hash=manifest_hash,
            ),
            seed=self.seed,
            scenario=self.scenario,
            controller_by_slot=self.controller_by_slot,
            roster_ids=self.roster_ids,
            side_by_slot=self.side_by_slot,
        )

    def finish(self) -> AI42DatasetMatch:
        if self._finished:
            raise AI42DatasetError("collector is already finalized")
        if not self._results:
            raise AI42DatasetError("collector has no policy ticks")
        if not bool(self._results[-1].done):
            raise AI42DatasetError("collector can finalize only a terminal match")
        self._finished = True
        return collect_ai42_match(
            self._results,
            self._submitted,
            self._parents,
            self._outcomes,
            recurrent_boundaries=self._boundaries,
            hero_ids=self.hero_ids,
            match_id=self.match_id,
            runtime_manifest=self.runtime_manifest,
            runtime_manifest_hash=self.runtime_manifest_hash,
            expected_ticks=self.expected_ticks,
            seed=self.seed,
            scenario=self.scenario,
            controller_by_slot=self.controller_by_slot,
            roster_ids=self.roster_ids,
            side_by_slot=self.side_by_slot,
        )

    collect = finish
    finalize = finish


collect_ai42 = collect_ai42_match


__all__ = ["AI42Collector", "AI42DatasetMatch", "collect_ai42", "collect_ai42_match", "collect_match"]
