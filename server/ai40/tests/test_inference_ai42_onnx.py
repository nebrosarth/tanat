from __future__ import annotations

import unittest

import numpy as np

from tanat_ai40.env import (
    ABILITY_COUNT, ABILITY_FEATURES, ACTION_KINDS, GLOBAL_FEATURES, HERO_COUNT,
    HERO_FEATURES, MAX_ENTITIES, NAVIGATION_ANCHORS, NAVIGATION_OFFSETS,
)
from tanat_ai40.inference_ai42_onnx import AI42ONNXEvaluationActor
from tanat_ai40.model_ai42_actor import CONTROL_CANCEL, CONTROL_HOLD, CONTROL_ISSUE
from tanat_ai40.runtime_ai42 import ACTOR_INPUT_NAMES, ACTOR_OUTPUT_NAMES


class _Value:
    def __init__(self, name: str):
        self.name = name


class _Session:
    prefer_hold = False

    def get_inputs(self):
        return [_Value(name) for name in ACTOR_INPUT_NAMES]

    def get_outputs(self):
        return [_Value(name) for name in ACTOR_OUTPUT_NAMES]

    def run(self, _outputs, feed):
        batch = feed["hero"].shape[0]
        control = np.zeros((batch, 4), dtype=np.float32)
        control[:, CONTROL_ISSUE] = 10
        if self.prefer_hold:
            control[:, CONTROL_HOLD] = 20
        else:
            control[-1, CONTROL_CANCEL] = 20
        kind = np.zeros((batch, ACTION_KINDS), dtype=np.float32)
        kind[:, 3] = 10
        target = np.zeros((batch, ACTION_KINDS, MAX_ENTITIES), dtype=np.float32)
        target[:, :, 2] = 10
        offset = np.zeros((batch, ACTION_KINDS, NAVIGATION_OFFSETS), dtype=np.float32)
        offset[:, :, 7] = 10
        anchor = np.zeros((batch, ACTION_KINDS, NAVIGATION_ANCHORS), dtype=np.float32)
        anchor[:, :, 4] = 10
        timing = np.zeros((batch, ACTION_KINDS, 4), dtype=np.float32)
        return [
            control, kind, target, offset, anchor, timing, timing.copy(),
            feed["h"] + 1, feed["c"] + 1,
        ]


class _Observation:
    batched_observations = None

    def __init__(self, *, dead: bool = False, active_order=None):
        self.teacher_actions = None
        self.teacher_valid = None
        self.hero = np.zeros((HERO_COUNT, HERO_FEATURES), dtype=np.float32)
        self.hero[:, 9] = float(dead)
        self.abilities = np.zeros(
            (HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES), dtype=np.float32,
        )
        self.entities = np.zeros(
            (HERO_COUNT, MAX_ENTITIES, 16), dtype=np.float32,
        )
        self.global_state = np.zeros(
            (HERO_COUNT, GLOBAL_FEATURES), dtype=np.float32,
        )
        self.entity_mask = np.ones((HERO_COUNT, MAX_ENTITIES), dtype=np.uint8)
        self.kind_mask = np.ones((HERO_COUNT, ACTION_KINDS), dtype=np.uint8)
        self.target_mask = np.ones((HERO_COUNT, MAX_ENTITIES), dtype=np.uint8)
        self.skill_target_mask = np.ones(
            (HERO_COUNT, ABILITY_COUNT, MAX_ENTITIES), dtype=np.uint8,
        )
        self.active_order = active_order


class AI42ONNXInferenceTests(unittest.TestCase):
    def test_control_masks_skill_wire_and_recurrent_reset_match_torch_runtime(self):
        session = _Session()
        actor = AI42ONNXEvaluationActor(session, 5, hidden_size=4)
        indices = np.arange(5, dtype=np.intp)
        actions, runtime, model = actor.act([_Observation()], indices)
        self.assertTrue(np.all(actions[:4, 0] == 3))
        self.assertTrue(np.all(actions[:4, 1] == 2))
        self.assertTrue(np.all(actions[:4, 2] == 7))
        self.assertTrue(np.all(actions[:4, 3] == 0))
        self.assertTrue(np.all(actions[4] == 0))
        self.assertEqual(int(model[4]), CONTROL_CANCEL)
        self.assertEqual(int(runtime[0]), 0)
        self.assertEqual(int(runtime[4]), 2)

        session.prefer_hold = True
        _, runtime, model = actor.act([_Observation()], indices)
        self.assertTrue(np.all(model == CONTROL_HOLD))
        self.assertTrue(np.all(runtime[:4] == 1))
        self.assertEqual(int(runtime[4]), 0)

        actor.act([_Observation(dead=True)], indices)
        self.assertTrue(np.all(actor.h == 1))
        actor.act([_Observation(dead=False)], indices)
        self.assertTrue(np.all(actor.h == 1))

    def test_reset_worker_and_interface_fail_closed(self):
        actor = AI42ONNXEvaluationActor(_Session(), 10, hidden_size=4)
        actor.h[:] = 3
        actor.reset_workers([1])
        self.assertTrue(np.all(actor.h[:5] == 3))
        self.assertTrue(np.all(actor.h[5:] == 0))
        with self.assertRaises(IndexError):
            actor.reset_workers([2])

        bad = _Session()
        bad.get_outputs = lambda: [_Value("wrong")]
        with self.assertRaises(ValueError):
            AI42ONNXEvaluationActor(bad, 5, hidden_size=4)


if __name__ == "__main__":
    unittest.main()
