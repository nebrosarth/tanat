"""Batched ONNX Runtime actor adapter for AI-42 rollouts.

The adapter owns recurrent state and performs the same deterministic masks and
control mapping as ``AI42EvaluationActor``.  ONNX Runtime executes the model;
Python only assembles the fixed NumPy batch and projects logits to wire actions.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .env import HERO_COUNT
from .model_ai42_actor import (
    CONTROL_CANCEL,
    CONTROL_CLASSES,
    CONTROL_HOLD,
    CONTROL_ISSUE,
    CONTROL_WAIT,
)
from .runtime_ai42 import ACTOR_INPUT_NAMES, ACTOR_OUTPUT_NAMES
from .train import stack_observations


RUNTIME_CONTROL_ISSUE = 0
RUNTIME_CONTROL_HOLD = 1
RUNTIME_CONTROL_IDLE = 2


def create_onnx_session(model: str, *, cuda: bool) -> Any:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    available = ort.get_available_providers()
    if cuda and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"ONNX Runtime CUDA provider is unavailable: {available}")
    requested = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cuda else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model, providers=requested)
    active = list(session.get_providers())
    if cuda and (not active or active[0] != "CUDAExecutionProvider"):
        raise RuntimeError(
            f"ONNX Runtime silently fell back from CUDA; active providers are {active}"
        )
    return session


def _validate_interface(session: Any) -> None:
    inputs = tuple(value.name for value in session.get_inputs())
    outputs = tuple(value.name for value in session.get_outputs())
    if inputs != ACTOR_INPUT_NAMES:
        raise ValueError(f"ONNX actor inputs {inputs} != {ACTOR_INPUT_NAMES}")
    if outputs != ACTOR_OUTPUT_NAMES:
        raise ValueError(f"ONNX actor outputs {outputs} != {ACTOR_OUTPUT_NAMES}")


def _logsumexp_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    maximum = np.maximum(first, second)
    return maximum + np.log(np.exp(first - maximum) + np.exp(second - maximum))


def _top_two_margin(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("margin logits must be rank two with at least two classes")
    top_two = np.partition(logits, logits.shape[1] - 2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


class AI42ONNXEvaluationActor:
    """Fixed-shape greedy actor with fail-closed ONNX output validation."""

    def __init__(
        self,
        session: Any,
        batch_size: int,
        hidden_size: int,
        *,
        heroes_per_worker: int = HERO_COUNT // 2,
    ) -> None:
        if batch_size < 1 or hidden_size < 1 or heroes_per_worker < 1:
            raise ValueError("batch, hidden size, and worker width must be positive")
        if batch_size % heroes_per_worker:
            raise ValueError("batch size must be divisible by heroes_per_worker")
        _validate_interface(session)
        self.session = session
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.heroes_per_worker = heroes_per_worker
        self.h = np.zeros((batch_size, hidden_size), dtype=np.float32)
        self.c = np.zeros((batch_size, hidden_size), dtype=np.float32)
        self.previous_dead = np.zeros(batch_size, dtype=np.bool_)
        self.active_order = np.zeros(batch_size, dtype=np.bool_)
        self.last_decision_margin = np.full(batch_size, np.nan, dtype=np.float32)

    def reset_workers(self, worker_indices: Sequence[int]) -> None:
        workers = self.batch_size // self.heroes_per_worker
        for worker in worker_indices:
            if not 0 <= int(worker) < workers:
                raise IndexError(f"worker index {worker} out of range")
            start = int(worker) * self.heroes_per_worker
            stop = start + self.heroes_per_worker
            self.h[start:stop] = 0
            self.c[start:stop] = 0
            self.previous_dead[start:stop] = False
            self.active_order[start:stop] = False
            self.last_decision_margin[start:stop] = np.nan

    def act(
        self, observations: Sequence[Any], indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.intp)
        if indices.shape != (self.batch_size,):
            raise ValueError(f"indices must have shape ({self.batch_size},)")
        if all(getattr(result, "active_order", None) is not None for result in observations):
            active = np.concatenate([result.active_order for result in observations])[indices]
            self.active_order[:] = np.asarray(active, dtype=np.bool_)
        batch = stack_observations(observations)

        def array(name: str, dtype: np.dtype[Any]) -> np.ndarray:
            return np.ascontiguousarray(getattr(batch, name)[indices], dtype=dtype)

        hero = array("hero", np.dtype(np.float32))
        abilities = array("abilities", np.dtype(np.float32))
        entities = array("entities", np.dtype(np.float32))
        global_state = array("global_state", np.dtype(np.float32))
        entity_mask = array("entity_mask", np.dtype(np.bool_))
        kind_mask = array("kind_mask", np.dtype(np.bool_))
        target_mask = array("target_mask", np.dtype(np.bool_))
        skill_target_mask = array("skill_target_mask", np.dtype(np.bool_))
        dead = hero[:, 9] >= 0.5
        reset = dead | self.previous_dead
        self.h[reset] = 0
        self.c[reset] = 0
        self.active_order[reset] = False
        self.previous_dead[:] = dead
        feed = {
            "hero": hero,
            "abilities": abilities,
            "entities": entities,
            "global_state": global_state,
            "entity_mask": entity_mask,
            "h": self.h,
            "c": self.c,
        }
        raw = self.session.run(None, feed)
        if len(raw) != len(ACTOR_OUTPUT_NAMES):
            raise RuntimeError(f"ONNX actor returned {len(raw)} outputs")
        output = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in zip(ACTOR_OUTPUT_NAMES, raw)
        }
        if any(not np.isfinite(value).all() for value in output.values()):
            raise RuntimeError("ONNX actor returned non-finite output")
        expected = {
            "control": (self.batch_size, CONTROL_CLASSES),
            "kind": (self.batch_size, kind_mask.shape[1]),
            "target": (self.batch_size, kind_mask.shape[1], entity_mask.shape[1]),
            "next_h": (self.batch_size, self.hidden_size),
            "next_c": (self.batch_size, self.hidden_size),
        }
        for name, shape in expected.items():
            if output[name].shape != shape:
                raise RuntimeError(f"ONNX {name} shape {output[name].shape} != {shape}")
        self.h[:] = output["next_h"]
        self.c[:] = output["next_c"]

        control_logits = output["control"]
        model_controls = control_logits.argmax(axis=1)
        runtime_logits = np.stack(
            (
                control_logits[:, CONTROL_ISSUE],
                control_logits[:, CONTROL_HOLD],
                _logsumexp_pair(
                    control_logits[:, CONTROL_WAIT],
                    control_logits[:, CONTROL_CANCEL],
                ),
            ),
            axis=1,
        )
        runtime_logits[~self.active_order, RUNTIME_CONTROL_HOLD] = -1e9
        runtime_controls = runtime_logits.argmax(axis=1)
        self.active_order = np.where(
            runtime_controls == RUNTIME_CONTROL_ISSUE,
            True,
            np.where(
                runtime_controls == RUNTIME_CONTROL_IDLE,
                False,
                self.active_order,
            ),
        )

        kinds = np.where(kind_mask, output["kind"], -1e9).argmax(axis=1)
        rows = np.arange(self.batch_size)
        selected_skill = skill_target_mask[rows, np.clip(kinds - 3, 0, 3)]
        skill_rows = (kinds >= 3) & (kinds <= 6)
        effective_targets = np.where(skill_rows[:, None], selected_skill, target_mask)
        effective_targets &= entity_mask
        empty = ~effective_targets.any(axis=1)
        effective_targets[empty, 0] = True
        targets = np.where(
            effective_targets,
            output["target"][rows, kinds],
            -1e9,
        ).argmax(axis=1)
        offsets = output["offset"][rows, kinds].argmax(axis=1)
        anchors = output["anchor"][rows, kinds].argmax(axis=1)
        anchors = np.where(skill_rows, 0, anchors)
        # Canonical v13 wire: target belongs to attack/skills, local offsets to
        # move/skills, and semantic anchors to move only. The server ignores
        # irrelevant bytes, but durable strict replay intentionally does not.
        targets = np.where((kinds >= 2) & (kinds <= 6), targets, 0)
        offsets = np.where((kinds == 1) | skill_rows, offsets, 0)
        anchors = np.where(kinds == 1, anchors, 0)
        control_margin = _top_two_margin(runtime_logits)
        kind_margin = _top_two_margin(np.where(kind_mask, output["kind"], -1e9))
        selected_target_logits = np.where(
            effective_targets,
            output["target"][rows, kinds],
            -1e9,
        )
        target_margin = _top_two_margin(selected_target_logits)
        issue_margin = np.minimum(kind_margin, target_margin)
        self.last_decision_margin[:] = np.where(
            runtime_controls == RUNTIME_CONTROL_ISSUE,
            np.minimum(control_margin, issue_margin),
            control_margin,
        ).astype(np.float32, copy=False)
        actions = np.stack((kinds, targets, offsets, anchors), axis=1).astype(np.int16)
        actions = np.where(
            (runtime_controls == RUNTIME_CONTROL_ISSUE)[:, None],
            actions,
            np.zeros_like(actions),
        )
        return (
            actions,
            runtime_controls.astype(np.int64, copy=False),
            model_controls.astype(np.int64, copy=False),
        )


__all__ = ["AI42ONNXEvaluationActor", "create_onnx_session"]
