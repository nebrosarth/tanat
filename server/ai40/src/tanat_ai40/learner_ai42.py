"""Actor-only AI-42 behavior-cloning learner primitives.

This module owns the learner boundary for the AI-42 actor.  It intentionally
does not import either of the older training modules.  The future dataset
worker can adapt its serialized records to :class:`AI42Batch` (or the
``from_mapping`` protocol) without coupling that worker to PyTorch internals.

The batch convention is ``[sequence, time, ...]``.  A sequence is contiguous;
``reset_mask`` and ``death_mask`` reset recurrent state before the marked
step, while ``padding_mask`` marks rows that must not update state or loss.
Teacher control statuses are mapped to the explicit recurrent control head:
ACTION->ISSUE, WAIT->WAIT, HOLD->HOLD, and CANCEL->CANCEL.  Only ISSUE rows
provide the v13 kind/target/offset/anchor labels.  NONE and UNAVAILABLE are
excluded entirely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
    MAX_ENTITIES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from .model_ai42_actor import (
    AI42Actor,
    CONTROL_CANCEL,
    CONTROL_CLASSES,
    CONTROL_HOLD,
    CONTROL_ISSUE,
    CONTROL_NAMES,
    CONTROL_WAIT,
)


class AI42LearnerError(ValueError):
    """Base class for malformed learner input or incompatible artifacts."""


class NonFiniteError(AI42LearnerError):
    """Raised whenever an observation, loss, gradient, or checkpoint is non-finite."""


class CheckpointError(AI42LearnerError):
    """Raised for an incomplete, corrupt, or manifest-incompatible checkpoint."""


class TeacherStatus(IntEnum):
    NONE = 0
    ACTION = 1
    WAIT = 2
    HOLD = 3
    CANCEL = 4
    UNAVAILABLE = 5


CONTROL_STATUSES = frozenset({
    int(TeacherStatus.ACTION),
    int(TeacherStatus.WAIT),
    int(TeacherStatus.HOLD),
    int(TeacherStatus.CANCEL),
})
EXCLUDED_STATUSES = frozenset({
    int(TeacherStatus.NONE),
    int(TeacherStatus.UNAVAILABLE),
})
ACTION_STATUS = int(TeacherStatus.ACTION)
# These are the single source of truth for the actor-v13 supervision
# vocabulary.  The class-profile and metrics modules import them instead of
# repeating head sizes or relying on model implementation details.
HEAD_NAMES = ("control", "kind", "target", "offset", "anchor")
HEAD_CLASS_COUNTS = {
    "control": CONTROL_CLASSES,
    "kind": ACTION_KINDS,
    "target": MAX_ENTITIES,
    "offset": NAVIGATION_OFFSETS,
    "anchor": NAVIGATION_ANCHORS,
}
# Kept as a public compatibility constant for callers that use the v13
# action vocabulary.  Control WAIT is not mapped to this action kind.
WAIT_KIND = 0
CONTROL_STATUS_TO_CLASS = {
    ACTION_STATUS: CONTROL_ISSUE,
    int(TeacherStatus.WAIT): CONTROL_WAIT,
    int(TeacherStatus.HOLD): CONTROL_HOLD,
    int(TeacherStatus.CANCEL): CONTROL_CANCEL,
}
MOVE_KIND = 1
ATTACK_KIND = 2
SKILL_KINDS = tuple(range(3, 7))
TELEPORT_KIND = 7
TARGET_KINDS = frozenset({ATTACK_KIND, *SKILL_KINDS})
SPATIAL_KINDS = frozenset({MOVE_KIND, *SKILL_KINDS})

_STATUS_NAMES = {
    "none": TeacherStatus.NONE,
    "action": TeacherStatus.ACTION,
    "wait": TeacherStatus.WAIT,
    "hold": TeacherStatus.HOLD,
    "cancel": TeacherStatus.CANCEL,
    "unavailable": TeacherStatus.UNAVAILABLE,
}
_ACTION_ALIASES = {
    "offset": "offset",
    "direction": "offset",
    "anchor": "anchor",
    "distance": "anchor",
    "kind": "kind",
    "target": "target",
}


def _as_tensor(value: Any, *, dtype: torch.dtype, name: str) -> Tensor:
    try:
        result = value if isinstance(value, Tensor) else torch.as_tensor(value)
        result = result.to(dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AI42LearnerError(f"{name} cannot be converted to a tensor") from exc
    return result


def _finite_tensor(value: Tensor, name: str) -> None:
    if value.is_floating_point() or value.is_complex():
        if not bool(torch.isfinite(value).all()):
            raise NonFiniteError(f"{name} contains NaN or Inf")


def _bool_tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if value.dtype not in (torch.bool, torch.uint8):
            raise AI42LearnerError(f"{name} must use bool or uint8 zero/one values")
        if value.dtype == torch.uint8 and bool(((value != 0) & (value != 1)).any()):
            raise AI42LearnerError(f"{name} must contain only zero/one values")
        return value.to(dtype=torch.bool)
    array = np.asarray(value)
    if array.dtype not in (np.dtype("bool"), np.dtype("u1")):
        raise AI42LearnerError(f"{name} must use bool or uint8 zero/one values")
    if array.dtype == np.dtype("u1") and np.any((array != 0) & (array != 1)):
        raise AI42LearnerError(f"{name} must contain only zero/one values")
    return torch.as_tensor(array, dtype=torch.bool)


def _status_tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Tensor) and value.dtype in (
        torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
    ):
        # Preserve the source device.  Converting an already-normalized CUDA
        # status tensor through ``tolist`` silently moved it back to CPU when
        # AI42Batch.to() reconstructed the immutable dataclass.
        result = value.to(dtype=torch.int64)
        valid = torch.zeros_like(result, dtype=torch.bool)
        for status in TeacherStatus:
            valid |= result == int(status)
        if not bool(valid.all()):
            raise AI42LearnerError(f"{name} contains an unknown status")
        return result
    try:
        raw = value.tolist() if isinstance(value, np.ndarray) else value
    except AttributeError:
        raw = value

    def convert(item: Any) -> int:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in _STATUS_NAMES:
                raise AI42LearnerError(f"{name} contains unknown status {item!r}")
            return int(_STATUS_NAMES[key])
        if isinstance(item, (bool, np.bool_)):
            raise AI42LearnerError(f"{name} statuses must be integers or names")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AI42LearnerError(f"{name} contains a non-integer status") from exc
        if integer != item or integer not in _STATUS_NAMES.values():
            raise AI42LearnerError(f"{name} contains unknown status {item!r}")
        return integer

    if isinstance(raw, Tensor):
        raw = raw.detach().cpu().tolist()
    if isinstance(raw, (str, bytes)):
        raw = convert(raw)
    if isinstance(raw, Sequence):
        def recurse(item: Any) -> Any:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                return [recurse(child) for child in item]
            return convert(item)
        raw = recurse(raw)
    else:
        raw = convert(raw)
    return _as_tensor(raw, dtype=torch.int64, name=name)


def _action_tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        for key, item in value.items():
            canonical = _ACTION_ALIASES.get(str(key))
            if canonical is not None:
                fields[canonical] = item
        missing = [name for name in ("kind", "target", "offset", "anchor") if name not in fields]
        if missing:
            raise AI42LearnerError(f"{name} is missing action field(s): {', '.join(missing)}")
        value = torch.stack(
            [_as_tensor(fields[field], dtype=torch.int64, name=f"{name}.{field}") for field in ("kind", "target", "offset", "anchor")],
            dim=-1,
        )
    if isinstance(value, Tensor):
        if value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise AI42LearnerError(f"{name} must contain integer labels")
    else:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.integer):
            raise AI42LearnerError(f"{name} must contain integer labels")
    result = _as_tensor(value, dtype=torch.int64, name=name)
    if result.ndim < 1 or result.shape[-1] != 4:
        raise AI42LearnerError(f"{name} must have shape [..., 4] (kind,target,offset,anchor)")
    return result


def _pick(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True, slots=True)
class AI42Batch:
    """Dataset/learner protocol for contiguous recurrent teacher sequences.

    Required tensors use ``[B,T,...]`` and are converted to CPU/GPU-neutral
    PyTorch tensors without moving devices.  ``padding_mask=True`` means
    padding; ``loss_mask=True`` means eligible supervision.  The effective
    mask is ``loss_mask & ~padding_mask & status_is_supervised``.
    """

    hero: Any
    abilities: Any
    entities: Any
    global_state: Any
    entity_mask: Any
    teacher_actions: Any = None
    teacher_status: Any = None
    actions: Any = None
    kind_mask: Any = None
    target_mask: Any = None
    skill_target_mask: Any = None
    padding_mask: Any = None
    loss_mask: Any = None
    reset_mask: Any = None
    death_mask: Any = None
    hero_ids: Any = None
    sequence_ids: Any = None
    time_indices: Any = None

    def __post_init__(self) -> None:
        observations = {
            "hero": (_as_tensor(self.hero, dtype=torch.float32, name="hero"), (HERO_FEATURES,)),
            "abilities": (_as_tensor(self.abilities, dtype=torch.float32, name="abilities"), (ABILITY_COUNT, ABILITY_FEATURES)),
            "entities": (_as_tensor(self.entities, dtype=torch.float32, name="entities"), (MAX_ENTITIES, ENTITY_FEATURES)),
            "global_state": (_as_tensor(self.global_state, dtype=torch.float32, name="global_state"), (GLOBAL_FEATURES,)),
        }
        for name, (value, suffix) in observations.items():
            _finite_tensor(value, name)
            if value.ndim != 2 + len(suffix) or tuple(value.shape[2:]) != suffix:
                raise AI42LearnerError(f"{name} must have shape [B,T,{','.join(map(str, suffix))}]")
            object.__setattr__(self, name, value)

        entity_mask = _bool_tensor(self.entity_mask, name="entity_mask")
        if entity_mask.ndim != 3 or tuple(entity_mask.shape[2:]) != (MAX_ENTITIES,):
            raise AI42LearnerError(f"entity_mask must have shape [B,T,{MAX_ENTITIES}]")
        object.__setattr__(self, "entity_mask", entity_mask)
        batch_size, sequence_length = self.hero.shape[:2]
        for name in ("abilities", "entities", "global_state", "entity_mask"):
            value = getattr(self, name)
            if tuple(value.shape[:2]) != (batch_size, sequence_length):
                raise AI42LearnerError(f"{name} does not share the hero [B,T] prefix")

        raw_actions = self.teacher_actions if self.teacher_actions is not None else self.actions
        if raw_actions is None:
            raise AI42LearnerError("teacher_actions (or actions) is required")
        action_tensor = _action_tensor(raw_actions, name="teacher_actions")
        if tuple(action_tensor.shape[:2]) != (batch_size, sequence_length):
            raise AI42LearnerError("teacher_actions does not share the observation [B,T] prefix")
        object.__setattr__(self, "teacher_actions", action_tensor)
        object.__setattr__(self, "actions", action_tensor)

        if self.teacher_status is None:
            raise AI42LearnerError("teacher_status is required")
        statuses = _status_tensor(self.teacher_status, name="teacher_status")
        if tuple(statuses.shape) != (batch_size, sequence_length):
            raise AI42LearnerError("teacher_status must have shape [B,T]")
        object.__setattr__(self, "teacher_status", statuses)

        def sequence_bool(value: Any, name: str, default: Tensor) -> Tensor:
            result = default if value is None else _bool_tensor(value, name=name)
            if tuple(result.shape) != (batch_size, sequence_length):
                raise AI42LearnerError(f"{name} must have shape [B,T]")
            return result

        padding = sequence_bool(self.padding_mask, "padding_mask", torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=self.hero.device))
        loss = sequence_bool(self.loss_mask, "loss_mask", ~padding)
        loss = loss & ~padding
        reset_default = torch.zeros_like(padding)
        reset_default[:, 0] = True
        reset = sequence_bool(self.reset_mask, "reset_mask", reset_default)
        death = sequence_bool(self.death_mask, "death_mask", torch.zeros_like(padding))
        object.__setattr__(self, "padding_mask", padding)
        object.__setattr__(self, "loss_mask", loss)
        object.__setattr__(self, "reset_mask", reset)
        object.__setattr__(self, "death_mask", death)

        kind_mask = self.kind_mask
        if kind_mask is None:
            raise AI42LearnerError("kind_mask is required")
        kind_mask_tensor = _bool_tensor(kind_mask, name="kind_mask")
        if tuple(kind_mask_tensor.shape) != (batch_size, sequence_length, ACTION_KINDS):
            raise AI42LearnerError(f"kind_mask must have shape [B,T,{ACTION_KINDS}]")
        object.__setattr__(self, "kind_mask", kind_mask_tensor)

        target_mask = self.target_mask
        if target_mask is None:
            raise AI42LearnerError("target_mask is required")
        target_mask_tensor = _bool_tensor(target_mask, name="target_mask")
        expected_target = (batch_size, sequence_length, MAX_ENTITIES)
        expected_conditioned = (batch_size, sequence_length, ACTION_KINDS, MAX_ENTITIES)
        if tuple(target_mask_tensor.shape) not in (expected_target, expected_conditioned):
            raise AI42LearnerError("target_mask must have shape [B,T,N] or [B,T,K,N]")
        object.__setattr__(self, "target_mask", target_mask_tensor)

        skill_mask = self.skill_target_mask
        if skill_mask is None:
            raise AI42LearnerError("skill_target_mask is required")
        skill_mask_tensor = _bool_tensor(skill_mask, name="skill_target_mask")
        if tuple(skill_mask_tensor.shape) != (batch_size, sequence_length, 4, MAX_ENTITIES):
            raise AI42LearnerError(f"skill_target_mask must have shape [B,T,4,{MAX_ENTITIES}]")
        object.__setattr__(self, "skill_target_mask", skill_mask_tensor)

        if self.hero_ids is not None:
            hero_ids = _as_tensor(self.hero_ids, dtype=torch.int64, name="hero_ids")
            if tuple(hero_ids.shape) not in ((batch_size,), (batch_size, sequence_length)):
                raise AI42LearnerError("hero_ids must have shape [B] or [B,T]")
            object.__setattr__(self, "hero_ids", hero_ids)
        if self.sequence_ids is not None:
            sequence_ids = _as_tensor(self.sequence_ids, dtype=torch.int64, name="sequence_ids")
            if tuple(sequence_ids.shape) != (batch_size,):
                raise AI42LearnerError("sequence_ids must have shape [B]")
            object.__setattr__(self, "sequence_ids", sequence_ids)
        if self.time_indices is not None:
            time_indices = _as_tensor(self.time_indices, dtype=torch.int64, name="time_indices")
            if tuple(time_indices.shape) != (batch_size, sequence_length):
                raise AI42LearnerError("time_indices must have shape [B,T]")
            if sequence_length > 1 and not bool((time_indices[:, 1:] == time_indices[:, :-1] + 1).all()):
                raise AI42LearnerError("time_indices must be contiguous within every sequence")
            object.__setattr__(self, "time_indices", time_indices)

    @property
    def batch_size(self) -> int:
        return int(self.hero.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.hero.shape[1])

    @property
    def supervision_mask(self) -> Tensor:
        statuses = self.teacher_status
        status_mask = torch.zeros_like(statuses, dtype=torch.bool)
        for status in CONTROL_STATUS_TO_CLASS:
            status_mask |= statuses == status
        return self.loss_mask & status_mask

    def to(self, device: torch.device | str) -> "AI42Batch":
        """Return an immutable batch copy on ``device``."""

        values: dict[str, Any] = {}
        for name in (
            "hero", "abilities", "entities", "global_state", "entity_mask", "teacher_actions", "teacher_status",
            "kind_mask", "target_mask", "skill_target_mask", "padding_mask", "loss_mask", "reset_mask", "death_mask",
            "hero_ids", "sequence_ids", "time_indices",
        ):
            value = getattr(self, name)
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        values["actions"] = values["teacher_actions"]
        moved = replace(self, **values)
        expected = torch.device(device)
        for name in values:
            value = getattr(moved, name)
            wrong_device = (
                isinstance(value, Tensor)
                and (
                    value.device.type != expected.type
                    or (expected.index is not None and value.device.index != expected.index)
                )
            )
            if wrong_device:
                raise AI42LearnerError(f"AI42Batch.to left {name} on {value.device}, expected {expected}")
        return moved

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AI42Batch":
        """Adapt a future dataset record without requiring its concrete class.

        The adapter accepts either flat fields or ``observations``, ``masks``
        and ``labels`` sub-mappings.  It is deliberately strict about unknown
        tensor shapes in ``__post_init__``.
        """

        if not isinstance(value, Mapping):
            raise AI42LearnerError("AI42 batch payload must be a mapping")
        observations = value.get("observations", value)
        masks = value.get("masks", value)
        labels = value.get("labels", value)
        if not isinstance(observations, Mapping) or not isinstance(masks, Mapping) or not isinstance(labels, Mapping):
            raise AI42LearnerError("observations, masks, and labels must be mappings")
        actions = _pick(labels, "teacher_actions", "actions", "action")
        if actions is None:
            action_fields = {name: _pick(labels, name, alias) for name, alias in (("kind", "action_kind"), ("target", "action_target"), ("offset", "direction"), ("anchor", "distance"))}
            if all(item is not None for item in action_fields.values()):
                actions = action_fields
        return cls(
            hero=_pick(observations, "hero"),
            abilities=_pick(observations, "abilities"),
            entities=_pick(observations, "entities"),
            global_state=_pick(observations, "global_state", "global"),
            entity_mask=_pick(masks, "entity_mask", "entities_mask"),
            teacher_actions=actions,
            teacher_status=_pick(labels, "teacher_status", "status"),
            kind_mask=_pick(masks, "kind_mask", "kind"),
            target_mask=_pick(masks, "target_mask", "target"),
            skill_target_mask=_pick(masks, "skill_target_mask", "skill_target"),
            padding_mask=_pick(value, "padding_mask", "padded"),
            loss_mask=_pick(value, "loss_mask", "valid_mask"),
            reset_mask=_pick(value, "reset_mask", "sequence_start", "boundary_mask"),
            death_mask=_pick(value, "death_mask", "done_mask"),
            hero_ids=_pick(observations, "hero_ids"),
            sequence_ids=_pick(value, "sequence_ids"),
            time_indices=_pick(value, "time_indices"),
        )

    from_dict = from_mapping


AI42SequenceBatch = AI42Batch


def _wire_action_matrix(value: np.ndarray) -> np.ndarray:
    """Convert the dataset structured action dtype to [...,4] int64 labels."""

    array = np.asarray(value)
    if array.dtype.names is None or set(array.dtype.names) != {"kind", "target", "direction", "distance"}:
        raise AI42LearnerError("dataset teacher_action must use the v13 structured action dtype")
    return np.stack(
        [array["kind"], array["target"], array["direction"], array["distance"]], axis=-1,
    ).astype(np.int64, copy=False)


def iter_ai42_dataset_batches(
    dataset: Any,
    *,
    split: str = "train",
    sequence_length: int = 64,
    batch_size: int = 32,
    supervision_controllers: Sequence[int] | None = None,
) -> Iterable[AI42Batch]:
    """Stream deterministic, match-isolated recurrent batches from AI42Dataset.

    Windows never cross a match or hero slot.  Short tails are padded, every
    window begins with a recurrent reset, and death plus the first respawn tick
    reset memory.  The dataset loader is responsible for hash/schema/lineage
    validation before this adapter is called.
    """

    if split not in {"train", "validation"}:
        raise AI42LearnerError("split must be 'train' or 'validation'")
    if isinstance(sequence_length, bool) or sequence_length < 1:
        raise AI42LearnerError("sequence_length must be positive")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise AI42LearnerError("batch_size must be positive")
    if not hasattr(dataset, "iter_matches") or not callable(dataset.iter_matches):
        raise AI42LearnerError("dataset must be a validated AI42Dataset")
    controller_filter: frozenset[int] | None = None
    controller_by_match: dict[str, tuple[int, ...]] = {}
    if supervision_controllers is not None:
        if (
            isinstance(supervision_controllers, (str, bytes))
            or not isinstance(supervision_controllers, Sequence)
            or not supervision_controllers
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value not in range(4)
                for value in supervision_controllers
            )
        ):
            raise AI42LearnerError("supervision_controllers must contain unique controller IDs in [0, 3]")
        controller_filter = frozenset(supervision_controllers)
        if len(controller_filter) != len(supervision_controllers):
            raise AI42LearnerError("supervision_controllers must contain unique controller IDs in [0, 3]")
        # Selecting every known controller is the backward-compatible,
        # unfiltered mode. It must continue to work with validated legacy
        # adapters that predate per-match controller metadata.
        if controller_filter == frozenset(range(4)):
            controller_filter = None
    if controller_filter is not None:
        manifest = getattr(dataset, "manifest", None)
        entries = manifest.get("matches") if isinstance(manifest, Mapping) else None
        if not isinstance(entries, list):
            raise AI42LearnerError("controller-filtered batching requires match metadata")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AI42LearnerError("controller-filtered batching found malformed match metadata")
            match_id, controllers = entry.get("match_id"), entry.get("controller_by_slot")
            if (
                not isinstance(match_id, str)
                or not isinstance(controllers, list)
                or len(controllers) != HERO_COUNT
                or any(isinstance(value, bool) or not isinstance(value, int) for value in controllers)
            ):
                raise AI42LearnerError("controller-filtered batching found malformed match metadata")
            controller_by_match[match_id] = tuple(controllers)

    pending: list[dict[str, np.ndarray | int]] = []

    def emit(rows: list[dict[str, np.ndarray | int]]) -> AI42Batch:
        def stack(name: str) -> np.ndarray:
            return np.stack([np.asarray(row[name]) for row in rows], axis=0)

        return AI42Batch(
            hero=stack("hero"),
            abilities=stack("abilities"),
            entities=stack("entities"),
            global_state=stack("global_state"),
            entity_mask=stack("entity_mask"),
            teacher_actions=stack("teacher_actions"),
            teacher_status=stack("teacher_status"),
            kind_mask=stack("kind_mask"),
            target_mask=stack("target_mask"),
            skill_target_mask=stack("skill_target_mask"),
            padding_mask=stack("padding_mask"),
            loss_mask=stack("loss_mask"),
            reset_mask=stack("reset_mask"),
            death_mask=stack("death_mask"),
            hero_ids=stack("hero_ids"),
            sequence_ids=np.asarray([row["sequence_id"] for row in rows], dtype=np.int64),
            time_indices=stack("time_indices"),
        )

    required = {
        "hero", "abilities", "entities", "global", "entity_mask", "kind_mask",
        "target_mask", "skill_target_mask", "teacher_status", "teacher_action", "step",
        "done",
    }
    for match_id, arrays in dataset.iter_matches(split):
        missing = sorted(required - set(arrays))
        if missing:
            raise AI42LearnerError(f"dataset match {match_id!r} is missing arrays {missing}")
        ticks = int(np.asarray(arrays["hero"]).shape[0])
        if ticks < 1 or np.asarray(arrays["hero"]).shape[1] != HERO_COUNT:
            raise AI42LearnerError(f"dataset match {match_id!r} has invalid tick/slot coverage")
        steps = np.asarray(arrays["step"], dtype=np.int64)
        if steps.shape != (ticks,) or (ticks > 1 and not np.all(steps[1:] == steps[:-1] + 1)):
            raise AI42LearnerError(f"dataset match {match_id!r} has non-contiguous steps")
        done = np.asarray(arrays["done"])
        if done.shape != (ticks,) or np.any(done[:-1] != 0) or done[-1] != 1:
            raise AI42LearnerError(f"dataset match {match_id!r} has an invalid terminal boundary")

        hero_slots = range(HERO_COUNT)
        if controller_filter is not None:
            controllers = controller_by_match.get(str(match_id))
            if controllers is None:
                raise AI42LearnerError(f"dataset match {match_id!r} has no controller metadata")
            hero_slots = tuple(
                slot for slot, controller in enumerate(controllers)
                if controller in controller_filter
            )
        for hero_slot in hero_slots:
            dead = np.asarray(arrays["hero"][:, hero_slot, 9] >= 0.5, dtype=np.bool_)
            for start in range(0, ticks, sequence_length):
                stop = min(start + sequence_length, ticks)
                length = stop - start

                def pad(name: str, source: np.ndarray) -> np.ndarray:
                    output = np.zeros((sequence_length, *source.shape[1:]), dtype=source.dtype)
                    output[:length] = source[start:stop]
                    return output

                hero = np.asarray(arrays["hero"][:, hero_slot])
                actions = _wire_action_matrix(np.asarray(arrays["teacher_action"][:, hero_slot]))
                padding = np.ones(sequence_length, dtype=np.bool_)
                padding[:length] = False
                reset = np.zeros(sequence_length, dtype=np.bool_)
                reset[0] = True
                death_reset = np.zeros(sequence_length, dtype=np.bool_)
                for local, tick in enumerate(range(start, stop)):
                    death_reset[local] = bool(dead[tick] or (tick > 0 and dead[tick - 1]))
                time_indices = np.arange(int(steps[start]), int(steps[start]) + sequence_length, dtype=np.int64)
                hero_ids = np.zeros(sequence_length, dtype=np.int64)
                hero_ids[:length] = np.rint(hero[start:stop, 0] * 100).astype(np.int64)
                identity = hashlib.sha256(f"{match_id}\0{hero_slot}\0{int(steps[start])}".encode("utf-8")).digest()
                sequence_id = int.from_bytes(identity[:8], "little") & ((1 << 63) - 1)
                pending.append({
                    "hero": pad("hero", hero),
                    "abilities": pad("abilities", np.asarray(arrays["abilities"][:, hero_slot])),
                    "entities": pad("entities", np.asarray(arrays["entities"][:, hero_slot])),
                    "global_state": pad("global", np.asarray(arrays["global"][:, hero_slot])),
                    "entity_mask": pad("entity_mask", np.asarray(arrays["entity_mask"][:, hero_slot])),
                    "teacher_actions": pad("teacher_action", actions),
                    "teacher_status": pad("teacher_status", np.asarray(arrays["teacher_status"][:, hero_slot])),
                    "kind_mask": pad("kind_mask", np.asarray(arrays["kind_mask"][:, hero_slot])),
                    "target_mask": pad("target_mask", np.asarray(arrays["target_mask"][:, hero_slot])),
                    "skill_target_mask": pad("skill_target_mask", np.asarray(arrays["skill_target_mask"][:, hero_slot])),
                    "padding_mask": padding,
                    "loss_mask": ~padding,
                    "reset_mask": reset,
                    "death_mask": death_reset,
                    "hero_ids": hero_ids,
                    "time_indices": time_indices,
                    "sequence_id": sequence_id,
                })
                if len(pending) == batch_size:
                    yield emit(pending)
                    pending = []
    if pending:
        yield emit(pending)


@dataclass(frozen=True, slots=True)
class AI42LearnerConfig:
    """Loss/optimizer settings; no setting implicitly performs an update."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    class_balance_power: float = 0.5
    max_gradient_norm: float = 1.0
    trainable_scope: str = "all"
    head_weights: Mapping[str, float] = field(default_factory=lambda: {
        "control": 1.0, "kind": 1.0, "target": 1.0, "offset": 1.0, "anchor": 1.0,
    })
    class_weights: Mapping[str, Sequence[float]] = field(default_factory=dict)
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("learning_rate", "weight_decay", "class_balance_power", "max_gradient_norm"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number < 0.0 or (name == "max_gradient_norm" and number == 0.0):
                raise AI42LearnerError(f"{name} must be finite and positive where required")
        valid_heads = {"control", "kind", "target", "offset", "anchor"}
        if self.trainable_scope not in {"all", "supervised_heads"}:
            raise AI42LearnerError("trainable_scope must be 'all' or 'supervised_heads'")
        for name, value in self.head_weights.items():
            if name not in valid_heads or not math.isfinite(float(value)) or float(value) < 0:
                raise AI42LearnerError(f"invalid head weight {name!r}")
        for name, values in self.class_weights.items():
            if name not in valid_heads:
                raise AI42LearnerError(f"invalid class-weight head {name!r}")
            try:
                values = tuple(values)
            except TypeError as exc:
                raise AI42LearnerError(f"class_weights[{name!r}] must be a sequence") from exc
            if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
                raise AI42LearnerError(f"class_weights[{name!r}] contains a non-finite or negative value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "class_balance_power": self.class_balance_power,
            "max_gradient_norm": self.max_gradient_norm,
            "trainable_scope": self.trainable_scope,
            "head_weights": dict(self.head_weights),
            "class_weights": {key: list(value) for key, value in self.class_weights.items()},
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(frozen=True, slots=True)
class LossResult:
    loss: Tensor
    head_losses: Mapping[str, Tensor]
    metrics: Mapping[str, Any]
    class_counts: Mapping[str, tuple[int, ...]]
    skill_metrics: Mapping[str, Mapping[str, Any]]
    head_weighted_numerators: Mapping[str, float] = field(default_factory=dict)
    head_weighted_denominators: Mapping[str, float] = field(default_factory=dict)

    @property
    def control_counts(self) -> Mapping[str, int]:
        """Return auditable counts keyed by the four control class names."""

        counts = self.class_counts.get("control", ())
        return {
            name: int(counts[index]) if index < len(counts) else 0
            for index, name in enumerate(CONTROL_NAMES)
        }

    @property
    def control_metrics(self) -> Mapping[str, Any]:
        """Return aggregate metrics for the explicit control head."""

        return self.metrics.get("control", {})

    def __iter__(self):
        yield self.loss
        yield self.metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach().cpu().item()),
            "head_losses": {name: float(value.detach().cpu().item()) for name, value in self.head_losses.items()},
            "metrics": dict(self.metrics),
            "class_counts": {name: list(value) for name, value in self.class_counts.items()},
            "control_counts": dict(self.control_counts),
            "skill_metrics": {name: dict(value) for name, value in self.skill_metrics.items()},
            "head_weighted_numerators": dict(self.head_weighted_numerators),
            "head_weighted_denominators": dict(self.head_weighted_denominators),
        }


@dataclass(frozen=True, slots=True)
class AI42Supervision:
    """Prepared labels, applicability masks, and action masks for one batch.

    Every consumer of teacher supervision (loss, class profiles, and metrics)
    uses this object.  Tensors retain the batch device and are flattened only
    across the recurrent ``[B,T]`` prefix.
    """

    labels: Mapping[str, Tensor]
    active: Mapping[str, Tensor]
    masks: Mapping[str, Tensor]
    action_rows: Tensor
    control_rows: Tensor
    supervised_rows: Tensor
    target_applicable: Tensor
    move_rows: Tensor
    skill_rows: Tensor

    @property
    def control_active(self) -> Tensor:
        return self.active["control"]

    @property
    def action_active(self) -> Tensor:
        return self.active["kind"]

    @property
    def action_parameter_active(self) -> Tensor:
        return self.supervised_rows

def class_balance_weights(class_counts: Sequence[int], power: float = 1.0) -> Tensor:
    """Return mean-one inverse-frequency weights; absent classes have weight 0."""

    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if (
        counts.ndim != 1
        or not bool(torch.isfinite(counts).all())
        or bool((counts < 0).any())
        or power < 0
        or not math.isfinite(float(power))
    ):
        raise AI42LearnerError("class_counts must be one-dimensional and power finite")
    weights = torch.zeros_like(counts)
    present = counts > 0
    if bool(present.any()):
        total = counts[present].sum()
        weights[present] = (total / counts[present]).pow(float(power))
        weights[present] /= weights[present].mean().clamp_min(torch.finfo(weights.dtype).eps)
    return weights


def clip_grad_norm_finite(parameters: Iterable[nn.Parameter], max_norm: float, *, norm_type: float = 2.0) -> float:
    """Validate, clip, and revalidate gradients, failing closed on non-finite data."""

    if not math.isfinite(float(max_norm)) or max_norm <= 0:
        raise AI42LearnerError("max_norm must be finite and positive")
    values = [parameter for parameter in parameters if parameter.grad is not None]
    for parameter in values:
        _finite_tensor(parameter.grad, "gradient")
    norm = nn.utils.clip_grad_norm_(values, max_norm, norm_type=norm_type)
    norm_value = float(norm.detach().cpu().item()) if isinstance(norm, Tensor) else float(norm)
    if not math.isfinite(norm_value):
        raise NonFiniteError("gradient norm is non-finite")
    for parameter in values:
        _finite_tensor(parameter.grad, "clipped gradient")
    return norm_value


def _validate_model_device(actor: nn.Module, batch: AI42Batch) -> None:
    parameter = next(actor.parameters(), None)
    if parameter is None:
        raise AI42LearnerError("actor has no parameters")
    for name in ("hero", "abilities", "entities", "global_state", "entity_mask"):
        if getattr(batch, name).device != parameter.device:
            raise AI42LearnerError(f"batch.{name} device does not match actor device")
    for value_name in ("hero", "abilities", "entities", "global_state"):
        _finite_tensor(getattr(batch, value_name), f"batch.{value_name}")


def forward_batch(
    actor: AI42Actor,
    batch: AI42Batch,
    *,
    initial_state: tuple[Tensor, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Run a contiguous ``[B,T]`` batch while honoring recurrent boundaries."""

    if not isinstance(batch, AI42Batch):
        batch = AI42Batch.from_mapping(batch)  # type: ignore[arg-type]
    _validate_model_device(actor, batch)
    parameter = next(actor.parameters())
    if initial_state is None:
        h, c = actor.initial_state(batch.batch_size, parameter.device, dtype=parameter.dtype)
    else:
        if len(initial_state) != 2:
            raise AI42LearnerError("initial_state must contain h and c")
        h, c = initial_state
        if tuple(h.shape) != (batch.batch_size, actor.hidden_size) or tuple(c.shape) != tuple(h.shape):
            raise AI42LearnerError("initial recurrent state has the wrong shape")
        if h.device != parameter.device or c.device != parameter.device or h.dtype != parameter.dtype or c.dtype != parameter.dtype:
            raise AI42LearnerError("initial recurrent state device/dtype does not match actor")
        _finite_tensor(h, "initial_state.h")
        _finite_tensor(c, "initial_state.c")
    outputs: dict[str, list[Tensor]] = {}
    for time in range(batch.sequence_length):
        reset = batch.reset_mask[:, time] | batch.death_mask[:, time]
        zero_h = torch.zeros_like(h)
        zero_c = torch.zeros_like(c)
        h = torch.where(reset.unsqueeze(-1), zero_h, h)
        c = torch.where(reset.unsqueeze(-1), zero_c, c)
        hero_ids = batch.hero_ids
        if isinstance(hero_ids, Tensor) and hero_ids.ndim == 2:
            hero_ids = hero_ids[:, time]
        step = actor(
            batch.hero[:, time], batch.abilities[:, time], batch.entities[:, time],
            batch.global_state[:, time], batch.entity_mask[:, time], h, c,
            hero_ids=hero_ids,
        )
        for name, value in step.items():
            if isinstance(value, Tensor):
                _finite_tensor(value, f"actor output {name}")
        active = ~batch.padding_mask[:, time]
        h = torch.where(active.unsqueeze(-1), step["h"], h)
        c = torch.where(active.unsqueeze(-1), step["c"], c)
        for name, value in step.items():
            if not isinstance(value, Tensor):
                continue
            if name == "h":
                value = h
            elif name == "c":
                value = c
            outputs.setdefault(name, []).append(value)
    stacked = {name: torch.stack(values, dim=1) for name, values in outputs.items()}
    stacked["final_h"] = h
    stacked["final_c"] = c
    return stacked


def _validate_logits(logits: Tensor, name: str) -> None:
    if not isinstance(logits, Tensor) or logits.ndim < 2:
        raise AI42LearnerError(f"{name} logits are not a tensor of class scores")
    _finite_tensor(logits, f"{name} logits")


def _counts(labels: Tensor, active: Tensor, classes: int) -> tuple[int, ...]:
    if not bool(active.any()):
        return tuple(0 for _ in range(classes))
    return tuple(int(item) for item in torch.bincount(labels[active], minlength=classes).detach().cpu().tolist()[:classes])


def _weights_for(config: AI42LearnerConfig, head: str, counts: tuple[int, ...], device: torch.device) -> Tensor:
    explicit = config.class_weights.get(head)
    if explicit is not None:
        weights = torch.as_tensor(explicit, dtype=torch.float32, device=device)
        if tuple(weights.shape) != (len(counts),) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise AI42LearnerError(f"class_weights[{head!r}] has the wrong shape or non-finite value")
        return weights
    weights = class_balance_weights(counts, config.class_balance_power).to(device=device)
    if not bool(torch.isfinite(weights).all()):
        raise NonFiniteError(f"{head} class-balance weights are non-finite")
    return weights


def _head_loss(
    logits: Tensor,
    labels: Tensor,
    active: Tensor,
    mask: Tensor,
    *,
    head: str,
    classes: int,
    config: AI42LearnerConfig,
) -> tuple[Tensor, dict[str, Any], tuple[int, ...]]:
    _validate_logits(logits, head)
    if tuple(logits.shape[-1:]) != (classes,) or tuple(labels.shape) != tuple(active.shape) or tuple(mask.shape) != tuple(labels.shape) + (classes,):
        raise AI42LearnerError(f"{head} tensors have inconsistent shapes")
    missing_mask = active & ~mask.any(dim=-1)
    if bool(missing_mask.any()):
        raise AI42LearnerError(f"{head} has supervised rows with an empty action mask")
    if bool(active.any()):
        if bool((labels[active] < 0).any()) or bool((labels[active] >= classes).any()):
            raise AI42LearnerError(f"{head} label is outside the model vocabulary")
        selected_mask = mask[active]
        selected_labels = labels[active]
        if not bool(selected_mask.gather(1, selected_labels.unsqueeze(1)).squeeze(1).all()):
            raise AI42LearnerError(f"{head} teacher label is excluded by its action mask")
        selected_logits = logits[active]
        masked_logits = selected_logits.masked_fill(~selected_mask, -torch.inf)
        per_item = F.cross_entropy(masked_logits, selected_labels, reduction="none")
        counts = _counts(labels, active, classes)
        weights = _weights_for(config, head, counts, logits.device)
        sample_weights = weights[selected_labels]
        weighted_numerator = (per_item * sample_weights).sum()
        weighted_denominator = sample_weights.sum()
        denominator = weighted_denominator.clamp_min(torch.finfo(per_item.dtype).eps)
        value = weighted_numerator / denominator
        predicted = masked_logits.argmax(dim=-1)
        accuracy = float((predicted == selected_labels).float().mean().detach().cpu().item())
        mean_weight = float(sample_weights.mean().detach().cpu().item())
    else:
        counts = tuple(0 for _ in range(classes))
        value = logits.sum() * 0.0
        accuracy = 0.0
        mean_weight = 0.0
    if bool(active.any()):
        weighted_numerator_value = float(
            (per_item.to(dtype=torch.float64) * sample_weights.to(dtype=torch.float64)).sum().detach().cpu().item()
        )
        weighted_denominator_value = float(sample_weights.to(dtype=torch.float64).sum().detach().cpu().item())
    else:
        weighted_numerator_value = 0.0
        weighted_denominator_value = 0.0
    if not bool(torch.isfinite(value).all()):
        raise NonFiniteError(f"{head} loss is non-finite")
    metrics = {
        "loss": float(value.detach().cpu().item()), "accuracy": accuracy,
        "count": int(sum(counts)), "mean_class_weight": mean_weight,
        "class_counts": counts,
        "weighted_numerator": weighted_numerator_value,
        "weighted_denominator": weighted_denominator_value,
    }
    return value, metrics, counts


def _conditioned_target_mask(batch: AI42Batch, kinds: Tensor) -> Tensor:
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    target_source = batch.target_mask.to(device=kinds.device)
    if batch.target_mask.ndim == 3:
        target = target_source.reshape(-1, MAX_ENTITIES)
    else:
        conditioned = target_source.reshape(-1, ACTION_KINDS, MAX_ENTITIES)
        target = conditioned[rows, kinds]
    entity = batch.entity_mask.to(device=kinds.device).reshape(-1, MAX_ENTITIES)
    target = target & entity
    skill = (kinds >= SKILL_KINDS[0]) & (kinds <= SKILL_KINDS[-1])
    skill_mask = batch.skill_target_mask.to(device=kinds.device).reshape(-1, 4, MAX_ENTITIES)[
        rows, (kinds - SKILL_KINDS[0]).clamp(0, 3),
    ]
    return torch.where(skill.unsqueeze(-1), skill_mask & entity, target)


def prepare_ai42_supervision(batch: AI42Batch) -> AI42Supervision:
    """Prepare the canonical AI-42 v13 supervision masks exactly once.

    Control rows include ISSUE/WAIT/HOLD/CANCEL.  Parameter heads are active
    only for ISSUE and use the teacher kind to determine target/offset/anchor
    applicability.  The function performs no model work and is therefore
    safe to use for a streaming train-split profile pass.
    """

    if not isinstance(batch, AI42Batch):
        batch = AI42Batch.from_mapping(batch)  # type: ignore[arg-type]
    flat_status = batch.teacher_status.reshape(-1)
    flat_base = batch.loss_mask.reshape(-1)
    flat_actions = batch.teacher_actions.reshape(-1, 4)
    action_rows = flat_status == ACTION_STATUS
    control_rows = torch.zeros_like(flat_base)
    control_labels = torch.zeros_like(flat_status)
    for status, control_class in CONTROL_STATUS_TO_CLASS.items():
        rows_for_status = flat_status == status
        control_rows |= rows_for_status
        control_labels = torch.where(
            rows_for_status,
            torch.full_like(control_labels, control_class),
            control_labels,
        )

    supervised = flat_base & action_rows
    kinds = flat_actions[:, 0]
    targets = flat_actions[:, 1]
    offsets = flat_actions[:, 2]
    anchors = flat_actions[:, 3]
    if bool((kinds < 0).any()) or bool((kinds >= ACTION_KINDS).any()):
        raise AI42LearnerError("kind label is outside the model vocabulary")
    target_mask = _conditioned_target_mask(batch, kinds)
    move_rows = kinds == MOVE_KIND
    skill_rows = (kinds >= SKILL_KINDS[0]) & (kinds <= SKILL_KINDS[-1])
    target_applicable = (kinds == ATTACK_KIND) | (skill_rows & target_mask.any(dim=-1))

    active = {
        "control": flat_base & control_rows,
        "kind": supervised,
        "target": supervised & target_applicable,
        "offset": supervised & (skill_rows | (move_rows & (anchors == 0))),
        "anchor": supervised & move_rows & (anchors > 0),
    }
    masks = {
        "control": torch.ones((*control_labels.shape, CONTROL_CLASSES), dtype=torch.bool, device=control_labels.device),
        "kind": batch.kind_mask.reshape(-1, ACTION_KINDS),
        "target": target_mask,
        "offset": torch.ones((*offsets.shape, NAVIGATION_OFFSETS), dtype=torch.bool, device=offsets.device),
        "anchor": torch.ones((*anchors.shape, NAVIGATION_ANCHORS), dtype=torch.bool, device=anchors.device),
    }
    labels = {
        "control": control_labels,
        "kind": kinds,
        "target": targets,
        "offset": offsets,
        "anchor": anchors,
    }
    return AI42Supervision(
        labels=labels,
        active=active,
        masks=masks,
        action_rows=action_rows,
        control_rows=control_rows,
        supervised_rows=supervised,
        target_applicable=target_applicable,
        move_rows=move_rows,
        skill_rows=skill_rows,
    )


def compute_behavior_cloning_loss(
    actor: AI42Actor,
    batch: AI42Batch,
    config: AI42LearnerConfig | None = None,
    *,
    outputs: Mapping[str, Tensor] | None = None,
) -> LossResult:
    """Compute masked actor-only BC loss and auditable metrics.

    Timing logits are validated for finiteness but intentionally contribute no
    loss or metric in this first learner contract.
    """

    config = AI42LearnerConfig() if config is None else config
    if not isinstance(batch, AI42Batch):
        batch = AI42Batch.from_mapping(batch)  # type: ignore[arg-type]
    output = forward_batch(actor, batch) if outputs is None else outputs
    for name in ("control", "kind", "target", "offset", "anchor", "timing", "timing_aux"):
        if name not in output:
            raise AI42LearnerError(f"actor output is missing {name}")
        _validate_logits(output[name], name)

    prepared = prepare_ai42_supervision(batch)
    labels = prepared.labels
    active = prepared.active
    masks = prepared.masks
    control_labels = labels["control"]
    control_active = active["control"]
    control_logits = output["control"].reshape(-1, CONTROL_CLASSES)
    control_value, control_metrics, control_counts = _head_loss(
        control_logits,
        control_labels,
        control_active,
        masks["control"],
        head="control",
        classes=CONTROL_CLASSES,
        config=config,
    )

    # Action parameters are meaningful only when the recurrent control
    # decision is ISSUE. WAIT/HOLD/CANCEL must not fabricate a v13 action.
    supervised = prepared.supervised_rows
    kinds = labels["kind"]
    targets = labels["target"]
    offsets = labels["offset"]
    anchors = labels["anchor"]
    kind_logits = output["kind"].reshape(-1, ACTION_KINDS)
    kind_value, kind_metrics, kind_counts = _head_loss(
        kind_logits, kinds, active["kind"], masks["kind"], head="kind", classes=ACTION_KINDS, config=config,
    )

    rows = torch.arange(kinds.shape[0], device=kinds.device)
    target_logits = output["target"].reshape(-1, ACTION_KINDS, MAX_ENTITIES)[rows, kinds]
    offset_logits = output["offset"].reshape(-1, ACTION_KINDS, NAVIGATION_OFFSETS)[rows, kinds]
    anchor_logits = output["anchor"].reshape(-1, ACTION_KINDS, NAVIGATION_ANCHORS)[rows, kinds]
    target_mask = masks["target"]
    target_active = active["target"]
    offset_active = active["offset"]
    anchor_active = active["anchor"]
    target_value, target_metrics, target_counts = _head_loss(
        target_logits, targets, target_active, target_mask, head="target", classes=MAX_ENTITIES, config=config,
    )
    offset_value, offset_metrics, offset_counts = _head_loss(
        offset_logits, offsets, offset_active, masks["offset"], head="offset", classes=NAVIGATION_OFFSETS, config=config,
    )
    anchor_value, anchor_metrics, anchor_counts = _head_loss(
        anchor_logits, anchors, anchor_active, masks["anchor"], head="anchor", classes=NAVIGATION_ANCHORS, config=config,
    )

    head_losses = {
        "control": control_value,
        "kind": kind_value,
        "target": target_value,
        "offset": offset_value,
        "anchor": anchor_value,
    }
    metrics_by_head = {
        "control": control_metrics,
        "kind": kind_metrics,
        "target": target_metrics,
        "offset": offset_metrics,
        "anchor": anchor_metrics,
    }
    total = sum((head_losses[name] * float(config.head_weights.get(name, 1.0)) for name in head_losses), kind_value.new_zeros(()))
    if not bool(torch.isfinite(total).all()):
        raise NonFiniteError("behavior-cloning loss is non-finite")

    skill_metrics: dict[str, dict[str, Any]] = {}
    for skill in SKILL_KINDS:
        skill_rows = offset_active & (kinds == skill)
        skill_target_rows = target_active & (kinds == skill)
        skill_entry: dict[str, Any] = {"count": int(skill_rows.sum().detach().cpu().item()), "target_count": int(skill_target_rows.sum().detach().cpu().item())}
        for head_name, logits, labels, active, classes, mask in (
            ("target", target_logits, targets, skill_target_rows, MAX_ENTITIES, target_mask),
            ("offset", offset_logits, offsets, skill_rows, NAVIGATION_OFFSETS, masks["offset"]),
            # Skill navigation is offset-only in protocol v13.  A skill anchor
            # is rejected by the bridge/dataset and is never a supervised row.
            ("anchor", anchor_logits, anchors, torch.zeros_like(skill_rows), NAVIGATION_ANCHORS, masks["anchor"]),
        ):
            if bool(active.any()):
                selected = logits[active].masked_fill(~mask[active], -torch.inf)
                predicted = selected.argmax(dim=-1)
                accuracy = float((predicted == labels[active]).float().mean().detach().cpu().item())
            else:
                accuracy = 0.0
            skill_entry[head_name] = {"accuracy": accuracy, "count": int(active.sum().detach().cpu().item()), "class_counts": _counts(labels, active, classes)}
        skill_metrics[str(skill - SKILL_KINDS[0] + 1)] = skill_entry

    metrics = {
        "heads": metrics_by_head,
        "skills": skill_metrics,
        "supervised_count": int(control_active.sum().detach().cpu().item()),
        "action_parameter_count": int(supervised.sum().detach().cpu().item()),
        "action_count": int(prepared.action_rows.logical_and(batch.loss_mask.reshape(-1)).sum().detach().cpu().item()),
        "control_count": int(control_active.sum().detach().cpu().item()),
        "excluded_count": int((batch.loss_mask.reshape(-1) & ~prepared.control_rows).sum().detach().cpu().item()),
        "timing": {"excluded": True, "reason": "timing heads are reserved by the initial AI-42 BC contract"},
    }
    return LossResult(
        loss=total,
        head_losses=head_losses,
        metrics=metrics,
        class_counts={
            "control": control_counts,
            "kind": kind_counts,
            "target": target_counts,
            "offset": offset_counts,
            "anchor": anchor_counts,
        },
        skill_metrics=skill_metrics,
        head_weighted_numerators={
            name: float(metrics_by_head[name]["weighted_numerator"]) for name in HEAD_NAMES
        },
        head_weighted_denominators={
            name: float(metrics_by_head[name]["weighted_denominator"]) for name in HEAD_NAMES
        },
    )


class AI42Learner:
    """Explicit actor learner facade; no method performs an implicit optimizer step."""

    def __init__(self, actor: AI42Actor, config: AI42LearnerConfig | None = None, *, optimizer: torch.optim.Optimizer | None = None):
        self.actor = actor
        self.config = AI42LearnerConfig() if config is None else config
        trainable = self._configure_trainable_parameters()
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
        )

    def _configure_trainable_parameters(self) -> list[nn.Parameter]:
        """Select the optimizer surface without changing the actor artifact.

        ``supervised_heads`` adapts only the terminal modules directly owned
        by the five supervised losses.  The observation encoders, attention,
        recurrent core, and shared action context remain fixed, preventing a
        rare-kind update from moving every other decision boundary.
        """

        prefixes = (
            "control_head.", "kind_head.", "target_query.", "entity_key.",
            "offset_head.", "anchor_head.",
        )
        selected: list[nn.Parameter] = []
        for name, parameter in self.actor.named_parameters():
            enabled = self.config.trainable_scope == "all" or name.startswith(prefixes)
            parameter.requires_grad_(enabled)
            if enabled:
                selected.append(parameter)
        if not selected:
            raise AI42LearnerError("trainable_scope selected no actor parameters")
        return selected

    def forward(self, batch: AI42Batch, *, initial_state: tuple[Tensor, Tensor] | None = None) -> dict[str, Tensor]:
        return forward_batch(self.actor, batch, initial_state=initial_state)

    def loss(self, batch: AI42Batch, *, outputs: Mapping[str, Tensor] | None = None) -> LossResult:
        return compute_behavior_cloning_loss(self.actor, batch, self.config, outputs=outputs)

    compute_loss = loss

    def backward(self, batch: AI42Batch, *, zero_grad: bool = True) -> LossResult:
        """Backpropagate once and validate gradients; deliberately no ``step``."""

        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)
        result = self.loss(batch)
        result.loss.backward()
        for parameter in self.actor.parameters():
            if parameter.grad is not None:
                _finite_tensor(parameter.grad, "gradient")
        return result

    def clip_gradients(self, max_norm: float | None = None) -> float:
        return clip_grad_norm_finite(self.actor.parameters(), self.config.max_gradient_norm if max_norm is None else max_norm)

    def scale_gradients(self, factor: float) -> None:
        """Scale an accumulated gradient set before clipping/stepping."""

        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise AI42LearnerError("gradient scale must be numeric")
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise AI42LearnerError("gradient scale must be finite and positive")
        for parameter in self.actor.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
                _finite_tensor(parameter.grad, "scaled gradient")

    def optimizer_step(self) -> None:
        """Perform the explicitly requested future update after finite checks."""

        for parameter in self.actor.parameters():
            if parameter.grad is not None:
                _finite_tensor(parameter.grad, "gradient")
        self.optimizer.step()
        for parameter in self.actor.parameters():
            _finite_tensor(parameter, "updated model parameter")
        _tree_finite(self.optimizer.state_dict(), "updated optimizer state")

    def save_checkpoint(self, path: str | os.PathLike[str], manifest: Mapping[str, Any], *, step: int = 0, epoch: int = 0, extra: Mapping[str, Any] | None = None) -> Path:
        return save_ai42_checkpoint(path, self.actor, self.optimizer, manifest, step=step, epoch=epoch, extra=extra)

    def load_checkpoint(self, path: str | os.PathLike[str], expected_manifest: Mapping[str, Any], *, map_location: str | torch.device = "cpu", restore_rng: bool = True) -> "ResumeState":
        return load_ai42_checkpoint(path, self.actor, self.optimizer, expected_manifest, map_location=map_location, restore_rng=restore_rng)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"manifest is not canonical JSON: {exc}") from exc


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(manifest))).hexdigest()


def validate_learner_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise CheckpointError("learner manifest must be a mapping")
    required = {"model_hash", "config_hash", "dataset_hash"}
    missing = sorted(required - set(manifest))
    if missing:
        raise CheckpointError("learner manifest is missing " + ", ".join(missing))
    normalized = dict(manifest)
    for field_name in required:
        value = normalized[field_name]
        if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
            raise CheckpointError(f"manifest.{field_name} must be lower-case SHA-256 hexadecimal")
        try:
            if len(bytes.fromhex(value)) != 32:
                raise ValueError
        except ValueError as exc:
            raise CheckpointError(f"manifest.{field_name} must be lower-case SHA-256 hexadecimal") from exc
    _canonical_json(normalized)
    return normalized


def build_learner_manifest(actor: nn.Module, config: AI42LearnerConfig | Mapping[str, Any], dataset_hash: str, **extra: Any) -> dict[str, Any]:
    """Build a deterministic learner manifest from model/config/dataset inputs."""

    config_payload = config.to_dict() if isinstance(config, AI42LearnerConfig) else dict(config)
    model_payload = {
        "class": f"{actor.__class__.__module__}.{actor.__class__.__qualname__}",
        "state": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in actor.state_dict().items()},
    }
    result = {
        "manifest_version": "AI42-bc-manifest-v1",
        "model_hash": hashlib.sha256(_canonical_json(model_payload)).hexdigest(),
        "config_hash": hashlib.sha256(_canonical_json(config_payload)).hexdigest(),
        "dataset_hash": dataset_hash,
    }
    result.update(extra)
    return validate_learner_manifest(result)


def _tree_finite(value: Any, name: str) -> None:
    if isinstance(value, Tensor):
        _finite_tensor(value, name)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _tree_finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _tree_finite(item, f"{name}[{index}]")


def _artifact_digest(value: Any) -> str:
    """Hash safe checkpoint trees deterministically, including tensor bytes."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"n")
        elif isinstance(item, bool):
            digest.update(b"b1" if item else b"b0")
        elif isinstance(item, int):
            digest.update(b"i" + str(item).encode("ascii") + b";")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise CheckpointError("checkpoint digest input contains a non-finite float")
            digest.update(b"f" + item.hex().encode("ascii") + b";")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"s" + len(encoded).to_bytes(8, "little") + encoded)
        elif isinstance(item, (bytes, bytearray)):
            raw = bytes(item)
            digest.update(b"y" + len(raw).to_bytes(8, "little") + raw)
        elif isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"t")
            update(str(tensor.dtype))
            update(tuple(tensor.shape))
            # Optimizer state includes scalar step tensors.  Reshaping first
            # makes the byte view valid for both scalar and non-scalar tensors
            # without changing the shape/type material included in the hash.
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            digest.update(len(raw).to_bytes(8, "little") + raw)
        elif isinstance(item, Mapping):
            digest.update(b"m" + len(item).to_bytes(8, "little"))
            encoded_items: list[tuple[str, Any]] = []
            for key, child in item.items():
                if not isinstance(key, (str, int)):
                    raise CheckpointError("checkpoint mapping keys must be strings or integers")
                encoded_items.append((f"{type(key).__name__}:{key}", child))
            for key, child in sorted(encoded_items, key=lambda pair: pair[0]):
                update(key)
                update(child)
        elif isinstance(item, (tuple, list)):
            digest.update(b"q" + len(item).to_bytes(8, "little"))
            for child in item:
                update(child)
        else:
            raise CheckpointError(f"unsupported checkpoint digest type {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Keep the checkpoint compatible with torch.load(weights_only=True):
        # NumPy ndarray pickle globals are intentionally not allow-listed.
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(np.asarray(numpy_state[1], dtype=np.uint32).copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch", "cuda"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CheckpointError("checkpoint RNG state is incomplete")
    try:
        random.setstate(value["python"])
        numpy_state = value["numpy"]
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "bit_generator", "state", "position", "has_gauss", "cached_gaussian",
        }:
            raise CheckpointError("checkpoint NumPy RNG state is invalid")
        numpy_tensor = numpy_state["state"]
        if not isinstance(numpy_tensor, Tensor) or numpy_tensor.dtype != torch.uint32:
            raise CheckpointError("checkpoint NumPy RNG tensor is invalid")
        np.random.set_state((
            str(numpy_state["bit_generator"]),
            numpy_tensor.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
        torch_state = value["torch"]
        if not isinstance(torch_state, Tensor) or torch_state.dtype != torch.uint8:
            raise CheckpointError("checkpoint CPU RNG state is invalid")
        torch.set_rng_state(torch_state.cpu())
        cuda_states = value["cuda"]
        if torch.cuda.is_available():
            if not isinstance(cuda_states, (list, tuple)) or len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("checkpoint CUDA RNG state does not match available devices")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"checkpoint RNG state cannot be restored: {exc}") from exc


def _state_dict_compatible(model: nn.Module, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointError("checkpoint model state is missing")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise CheckpointError("checkpoint model state keys are not an exact match")
    for name, expected_value in expected.items():
        value = state[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != tuple(expected_value.shape) or value.dtype != expected_value.dtype:
            raise CheckpointError(f"checkpoint model state {name!r} has an incompatible shape or dtype")
        _finite_tensor(value, f"checkpoint model state {name}")


def _checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, manifest: Mapping[str, Any], *, step: int, epoch: int, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = validate_learner_manifest(manifest)
    if isinstance(step, bool) or int(step) < 0 or isinstance(epoch, bool) or int(epoch) < 0:
        raise CheckpointError("checkpoint step and epoch must be non-negative integers")
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    extra_payload = dict(extra or {})
    _canonical_json(extra_payload)
    _state_dict_compatible(model, state)
    _tree_finite(optimizer_state, "optimizer state")
    artifacts = {
        "model": _artifact_digest(state),
        "optimizer": _artifact_digest(optimizer_state),
        "rng": _artifact_digest(_rng_state()),
        "extra": _artifact_digest(extra_payload),
    }
    payload = {
        "format": "AI42-bc-checkpoint-v1",
        "manifest": normalized,
        "manifest_digest": manifest_digest(normalized),
        "model_state_dict": state,
        "optimizer_state_dict": optimizer_state,
        "step": int(step),
        "epoch": int(epoch),
        "rng_state": _rng_state(),
        "extra": extra_payload,
        "artifact_hashes": artifacts,
    }
    payload["payload_digest"] = _artifact_digest({
        "format": payload["format"], "manifest_digest": payload["manifest_digest"],
        "step": payload["step"], "epoch": payload["epoch"], "artifact_hashes": artifacts,
    })
    return payload


def save_ai42_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    *,
    step: int = 0,
    epoch: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a complete model/optimizer/RNG checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, manifest, step=step, epoch=epoch, extra=extra)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
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


@dataclass(frozen=True, slots=True)
class ResumeState:
    step: int
    epoch: int
    manifest: Mapping[str, Any]
    extra: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Validated immutable checkpoint metadata without mutating a learner."""

    step: int
    epoch: int
    manifest: Mapping[str, Any]
    extra: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]
    payload_digest: str


@dataclass(frozen=True, slots=True)
class WarmStartState:
    """Provenance for a validated model-only warm start."""

    source_path: str
    source_file_sha256: str
    source_manifest_digest: str
    source_payload_digest: str
    source_model_hash: str
    source_model_artifact_hash: str
    source_dataset_hash: str
    source_step: int
    source_epoch: int


def inspect_ai42_checkpoint(
    path: str | os.PathLike[str],
    expected_manifest: Mapping[str, Any] | None = None,
    *,
    model: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointArtifact:
    """Validate every serialized checkpoint payload without loading state.

    This is intentionally separate from :func:`load_ai42_checkpoint`: promotion
    code needs to inspect a prior artifact while leaving the active learner and
    its optimizer/RNG untouched.  All payload and per-artifact digests are
    recomputed, including the model tensor tree.
    """

    expected = validate_learner_manifest(expected_manifest) if expected_manifest is not None else None
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("format") != "AI42-bc-checkpoint-v1":
        raise CheckpointError("checkpoint format is missing or incompatible")
    expected_fields = {
        "format", "manifest", "manifest_digest", "model_state_dict", "optimizer_state_dict",
        "step", "epoch", "rng_state", "extra", "artifact_hashes", "payload_digest",
    }
    if set(payload) != expected_fields:
        raise CheckpointError("checkpoint field set is incomplete or contains unknown data")
    actual_manifest = validate_learner_manifest(payload.get("manifest"))
    if expected is not None and actual_manifest != expected:
        raise CheckpointError("checkpoint manifest is not an exact match")
    if payload.get("manifest_digest") != manifest_digest(actual_manifest):
        raise CheckpointError("checkpoint manifest digest does not match its contents")

    model_state = payload.get("model_state_dict")
    if model is not None:
        _state_dict_compatible(model, model_state)
    elif not isinstance(model_state, Mapping):
        raise CheckpointError("checkpoint model state is missing")
    else:
        for name, value in model_state.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise CheckpointError("checkpoint model state contains an invalid entry")
            _finite_tensor(value, f"checkpoint model state {name}")

    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise CheckpointError("checkpoint optimizer state is missing")
    _tree_finite(optimizer_state, "checkpoint optimizer state")
    rng = payload.get("rng_state")
    if not isinstance(rng, Mapping):
        raise CheckpointError("checkpoint RNG state is missing")
    _tree_finite(rng, "checkpoint RNG state")
    step = payload.get("step")
    epoch = payload.get("epoch")
    if (
        isinstance(step, bool) or not isinstance(step, int) or step < 0
        or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
    ):
        raise CheckpointError("checkpoint step and epoch are invalid")
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        raise CheckpointError("checkpoint extra payload is not a mapping")
    _canonical_json(dict(extra))
    expected_artifacts = {
        "model": _artifact_digest(model_state),
        "optimizer": _artifact_digest(optimizer_state),
        "rng": _artifact_digest(rng),
        "extra": _artifact_digest(dict(extra)),
    }
    if payload.get("artifact_hashes") != expected_artifacts:
        raise CheckpointError("checkpoint artifact digest mismatch")
    expected_payload_digest = _artifact_digest({
        "format": payload["format"], "manifest_digest": payload["manifest_digest"],
        "step": step, "epoch": epoch, "artifact_hashes": expected_artifacts,
    })
    if payload.get("payload_digest") != expected_payload_digest:
        raise CheckpointError("checkpoint payload digest mismatch")
    return CheckpointArtifact(
        step=step,
        epoch=epoch,
        manifest=actual_manifest,
        extra=dict(extra),
        artifact_hashes=dict(expected_artifacts),
        payload_digest=str(expected_payload_digest),
    )


def load_ai42_model_warm_start(
    path: str | os.PathLike[str],
    model: nn.Module,
    expected_manifest: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    allow_dataset_change: bool = False,
) -> WarmStartState:
    """Restore model weights from a fully validated prior generation only.

    The source may have a different learner config (the first BC generation
    used batch-local balancing), but its protocol, dataset lineage, and model
    architecture must match.  Optimizer state, RNG state, step, epoch, and
    cursor are intentionally never read into the active learner.
    """

    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"warm-start checkpoint does not exist: {source}")
    target = validate_learner_manifest(expected_manifest)
    try:
        source_sha_before = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise CheckpointError(f"cannot hash warm-start checkpoint: {exc}") from exc
    artifact = inspect_ai42_checkpoint(source, None, model=model, map_location=map_location)
    source_manifest = dict(artifact.manifest)
    if not isinstance(allow_dataset_change, bool):
        raise CheckpointError("allow_dataset_change must be boolean")
    for field in ("protocol_version", "dataset_hash", "dataset_schema_version", "shard_schema_version"):
        if field == "dataset_hash" and allow_dataset_change:
            continue
        if field in target or field in source_manifest:
            if source_manifest.get(field) != target.get(field):
                raise CheckpointError(f"warm-start {field} is incompatible")
    if source_manifest.get("model_hash") != target.get("model_hash"):
        raise CheckpointError("warm-start model architecture hash is incompatible")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"cannot read warm-start checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("warm-start payload is not a mapping")
    model_state = payload.get("model_state_dict")
    _state_dict_compatible(model, model_state)
    try:
        source_sha_after = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise CheckpointError(f"cannot re-hash warm-start checkpoint: {exc}") from exc
    if source_sha_before != source_sha_after:
        raise CheckpointError("warm-start checkpoint changed during validation")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(model_state, strict=True)
        for name, value in model.state_dict().items():
            _finite_tensor(value, f"warm-start model state {name}")
    except Exception as exc:
        model.load_state_dict(before, strict=True)
        raise CheckpointError(f"warm-start model restore rolled back: {exc}") from exc
    return WarmStartState(
        source_path=str(source),
        source_file_sha256=source_sha_before,
        source_manifest_digest=manifest_digest(source_manifest),
        source_payload_digest=artifact.payload_digest,
        source_model_hash=str(source_manifest["model_hash"]),
        source_model_artifact_hash=artifact.artifact_hashes["model"],
        source_dataset_hash=str(source_manifest["dataset_hash"]),
        source_step=artifact.step,
        source_epoch=artifact.epoch,
    )


def load_ai42_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    expected_manifest: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> ResumeState:
    """Load only a complete, manifest-exact checkpoint, with rollback safety."""

    expected = validate_learner_manifest(expected_manifest)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("format") != "AI42-bc-checkpoint-v1":
        raise CheckpointError("checkpoint format is missing or incompatible")
    expected_fields = {
        "format", "manifest", "manifest_digest", "model_state_dict", "optimizer_state_dict",
        "step", "epoch", "rng_state", "extra", "artifact_hashes", "payload_digest",
    }
    if set(payload) != expected_fields:
        raise CheckpointError("checkpoint field set is incomplete or contains unknown data")
    actual_manifest = validate_learner_manifest(payload.get("manifest"))
    if actual_manifest != expected:
        raise CheckpointError("checkpoint manifest is not an exact match")
    if payload.get("manifest_digest") != manifest_digest(actual_manifest):
        raise CheckpointError("checkpoint manifest digest does not match its contents")
    model_state = payload.get("model_state_dict")
    _state_dict_compatible(model, model_state)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None:
        if not isinstance(optimizer_state, Mapping):
            raise CheckpointError("checkpoint optimizer state is missing")
        _tree_finite(optimizer_state, "checkpoint optimizer state")
    rng = payload.get("rng_state")
    if restore_rng:
        _tree_finite(rng, "checkpoint RNG state")
        if not isinstance(rng, Mapping):
            raise CheckpointError("checkpoint RNG state is missing")
    step = payload.get("step")
    epoch = payload.get("epoch")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointError("checkpoint step and epoch are invalid")
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        raise CheckpointError("checkpoint extra payload is not a mapping")
    _canonical_json(dict(extra))
    expected_artifacts = {
        "model": _artifact_digest(model_state),
        "optimizer": _artifact_digest(optimizer_state),
        "rng": _artifact_digest(rng),
        "extra": _artifact_digest(dict(extra)),
    }
    if payload.get("artifact_hashes") != expected_artifacts:
        raise CheckpointError("checkpoint artifact digest mismatch")
    expected_payload_digest = _artifact_digest({
        "format": payload["format"], "manifest_digest": payload["manifest_digest"],
        "step": step, "epoch": epoch, "artifact_hashes": expected_artifacts,
    })
    if payload.get("payload_digest") != expected_payload_digest:
        raise CheckpointError("checkpoint payload digest mismatch")

    before_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    before_rng = _rng_state() if restore_rng else None
    try:
        model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if restore_rng:
            _restore_rng_state(rng)
    except Exception as exc:
        model.load_state_dict(before_model, strict=True)
        if optimizer is not None and before_optimizer is not None:
            optimizer.load_state_dict(before_optimizer)
        if before_rng is not None:
            _restore_rng_state(before_rng)
        raise CheckpointError(f"checkpoint load rolled back: {exc}") from exc
    return ResumeState(step=step, epoch=epoch, manifest=actual_manifest, extra=dict(extra))


save_checkpoint = save_ai42_checkpoint
load_checkpoint = load_ai42_checkpoint
load_ai42_warm_start = load_ai42_model_warm_start
warm_start_ai42_checkpoint = load_ai42_model_warm_start


__all__ = [
    "ACTION_STATUS", "AI42Batch", "AI42Learner", "AI42LearnerConfig", "AI42LearnerError", "AI42SequenceBatch", "AI42Supervision", "HEAD_CLASS_COUNTS", "HEAD_NAMES",
    "ATTACK_KIND", "CheckpointError", "CONTROL_STATUSES", "EXCLUDED_STATUSES", "LossResult", "MOVE_KIND",
    "CheckpointArtifact", "NonFiniteError", "ResumeState", "WarmStartState", "SKILL_KINDS", "SPATIAL_KINDS", "TARGET_KINDS", "TeacherStatus", "TELEPORT_KIND",
    "WAIT_KIND", "build_learner_manifest", "class_balance_weights", "clip_grad_norm_finite", "compute_behavior_cloning_loss",
    "forward_batch", "inspect_ai42_checkpoint", "iter_ai42_dataset_batches", "load_ai42_checkpoint", "load_ai42_model_warm_start", "load_ai42_warm_start", "load_checkpoint", "manifest_digest", "prepare_ai42_supervision", "save_ai42_checkpoint", "save_checkpoint", "warm_start_ai42_checkpoint",
    "validate_learner_manifest",
]
