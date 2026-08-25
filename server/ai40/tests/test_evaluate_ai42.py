from __future__ import annotations

import unittest

import numpy as np
import torch

from tanat_ai40.env import ACTION_KINDS, HERO_COUNT, MAX_ENTITIES, NAVIGATION_ANCHORS, NAVIGATION_OFFSETS
from tanat_ai40.evaluate_ai42 import AI42EvaluationActor, controllers_for_side
from tanat_ai40.evaluate_ai42 import (
    RUNTIME_CONTROL_HOLD, RUNTIME_CONTROL_IDLE, RUNTIME_CONTROL_ISSUE,
)
from tanat_ai40.model_ai42_actor import CONTROL_CANCEL, CONTROL_HOLD, CONTROL_ISSUE


class _FixedActor(torch.nn.Module):
    hidden_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.seen_h: list[torch.Tensor] = []
        self.prefer_hold = False

    def initial_state(self, batch: int, device=None, dtype=None):
        dtype = dtype or torch.float32
        return (
            torch.zeros((batch, self.hidden_size), device=device, dtype=dtype),
            torch.zeros((batch, self.hidden_size), device=device, dtype=dtype),
        )

    def forward(self, hero, abilities, entities, global_state, entity_mask, h, c):
        self.seen_h.append(h.detach().clone())
        batch = hero.shape[0]
        control = torch.zeros((batch, 4), device=hero.device)
        control[:, CONTROL_ISSUE] = 10
        if self.prefer_hold:
            control[:, CONTROL_HOLD] = 20
        else:
            control[-1, CONTROL_CANCEL] = 20
        kind = torch.zeros((batch, ACTION_KINDS), device=hero.device)
        kind[:, 3] = 10
        target = torch.zeros((batch, ACTION_KINDS, MAX_ENTITIES), device=hero.device)
        target[:, :, 2] = 10
        offset = torch.zeros((batch, ACTION_KINDS, NAVIGATION_OFFSETS), device=hero.device)
        offset[:, :, 7] = 10
        anchor = torch.zeros((batch, ACTION_KINDS, NAVIGATION_ANCHORS), device=hero.device)
        anchor[:, :, 4] = 10
        return {
            "control": control, "kind": kind, "target": target,
            "offset": offset, "anchor": anchor, "h": h + 1, "c": c + 1,
        }


class _Observations:
    batched_observations = None

    def __init__(self, dead: bool = False, active_order=None):
        self.teacher_actions = None
        self.teacher_valid = None
        self.hero = np.zeros((HERO_COUNT, 32), dtype=np.float32)
        self.hero[:, 9] = float(dead)
        self.abilities = np.zeros((HERO_COUNT, 4, 40), dtype=np.float32)
        self.entities = np.zeros((HERO_COUNT, MAX_ENTITIES, 16), dtype=np.float32)
        self.global_state = np.zeros((HERO_COUNT, 32), dtype=np.float32)
        self.entity_mask = np.ones((HERO_COUNT, MAX_ENTITIES), dtype=np.uint8)
        self.kind_mask = np.ones((HERO_COUNT, ACTION_KINDS), dtype=np.uint8)
        self.target_mask = np.ones((HERO_COUNT, MAX_ENTITIES), dtype=np.uint8)
        self.skill_target_mask = np.ones((HERO_COUNT, 4, MAX_ENTITIES), dtype=np.uint8)
        self.active_order = active_order


class AI42EvaluationTests(unittest.TestCase):
    def test_controllers_alternate_candidate_side(self) -> None:
        self.assertEqual(controllers_for_side(1)[:5], (3,) * 5)
        self.assertEqual(controllers_for_side(1)[5:], (2,) * 5)
        self.assertEqual(controllers_for_side(2)[:5], (2,) * 5)
        self.assertEqual(controllers_for_side(2)[5:], (3,) * 5)

    def test_actor_applies_control_masks_and_skill_wire_contract(self) -> None:
        actor = _FixedActor()
        evaluator = AI42EvaluationActor(actor, 5, torch.device("cpu"))
        actions, runtime, model = evaluator.act(
            [_Observations()], np.arange(5, dtype=np.intp),
        )
        self.assertTrue(np.all(actions[:4, 0] == 3))
        self.assertTrue(np.all(actions[:4, 1] == 2))
        self.assertTrue(np.all(actions[:4, 2] == 7))
        self.assertTrue(np.all(actions[:4, 3] == 0))
        self.assertTrue(np.all(actions[4] == 0))
        self.assertEqual(int(model[4]), CONTROL_CANCEL)
        self.assertEqual(int(runtime[0]), RUNTIME_CONTROL_ISSUE)
        self.assertEqual(int(runtime[4]), RUNTIME_CONTROL_IDLE)

        actor.prefer_hold = True
        _, runtime, model = evaluator.act(
            [_Observations()], np.arange(5, dtype=np.intp),
        )
        self.assertTrue(np.all(model == CONTROL_HOLD))
        self.assertTrue(np.all(runtime[:4] == RUNTIME_CONTROL_HOLD))
        self.assertEqual(int(runtime[4]), RUNTIME_CONTROL_ISSUE)

        _, runtime, _ = evaluator.act(
            [_Observations(active_order=np.zeros(HERO_COUNT, dtype=np.uint8))],
            np.arange(5, dtype=np.intp),
        )
        self.assertTrue(np.all(runtime == RUNTIME_CONTROL_ISSUE))

    def test_death_and_first_respawn_tick_reset_recurrent_state(self) -> None:
        actor = _FixedActor()
        evaluator = AI42EvaluationActor(actor, 5, torch.device("cpu"))
        indices = np.arange(5, dtype=np.intp)
        evaluator.act([_Observations()], indices)
        evaluator.act([_Observations(dead=True)], indices)
        evaluator.act([_Observations(dead=False)], indices)
        evaluator.act([_Observations(dead=False)], indices)
        self.assertTrue(torch.equal(actor.seen_h[0], torch.zeros_like(actor.seen_h[0])))
        self.assertTrue(torch.equal(actor.seen_h[1], torch.zeros_like(actor.seen_h[1])))
        self.assertTrue(torch.equal(actor.seen_h[2], torch.zeros_like(actor.seen_h[2])))
        self.assertTrue(torch.equal(actor.seen_h[3], torch.ones_like(actor.seen_h[3])))


if __name__ == "__main__":
    unittest.main()
