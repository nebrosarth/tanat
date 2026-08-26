from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from tanat_ai40.env import ACTION_KINDS, HERO_COUNT, MAX_ENTITIES, NAVIGATION_ANCHORS, NAVIGATION_OFFSETS
from tanat_ai40.evaluate_ai42 import AI42EvaluationActor, controllers_for_side, evaluate_vs_ai30
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
        self.kind = 3

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
        kind[:, self.kind] = 10
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


class _TerminalObservations(_Observations):
    def __init__(self):
        super().__init__(active_order=np.zeros(HERO_COUNT, dtype=np.uint8))
        self.step = 1
        self.elapsed = 0.2
        self.done = True
        self.winner = 1
        self.invalid = np.zeros(HERO_COUNT, dtype=np.uint8)
        self.rewards = np.zeros(HERO_COUNT, dtype=np.float32)


class _OneStepEnv:
    def __init__(self, _executable, workers, _protocol):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def reset(self, *_args, **_kwargs):
        return [_Observations(active_order=np.zeros(HERO_COUNT, dtype=np.uint8))]

    def step(self, _actions):
        return [_TerminalObservations()]

    def reset_indices(self, *_args, **_kwargs):
        raise AssertionError("terminal one-match evaluation must not reset a worker")


class AI42EvaluationTests(unittest.TestCase):
    def test_onnx_backend_requires_an_explicit_artifact_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires onnx_path"):
            evaluate_vs_ai30(
                Path("checkpoint.pt"), Path("config.json"), Path("assaultenv.exe"),
                matches=1, workers=1, max_steps=1, device=torch.device("cpu"),
                backend="onnxruntime",
            )

    def test_unknown_backend_fails_before_loading_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend must be"):
            evaluate_vs_ai30(
                Path("checkpoint.pt"), Path("config.json"), Path("assaultenv.exe"),
                matches=1, workers=1, max_steps=1, device=torch.device("cpu"),
                backend="unknown",
            )

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

    def test_torch_attack_wire_discards_irrelevant_navigation_logits(self) -> None:
        actor = _FixedActor()
        actor.kind = 2
        evaluator = AI42EvaluationActor(actor, 5, torch.device("cpu"))
        actions, runtime, _ = evaluator.act(
            [_Observations()], np.arange(5, dtype=np.intp),
        )
        issued = runtime == RUNTIME_CONTROL_ISSUE
        self.assertTrue(issued.any())
        self.assertTrue(np.all(actions[issued, 0] == 2))
        self.assertTrue(np.all(actions[issued, 1] == 2))
        self.assertTrue(np.all(actions[issued, 2:] == 0))

    def test_evaluation_reports_inference_and_environment_phase_profile(self) -> None:
        actor = _FixedActor()
        lineage = {"checkpoint_sha256": "0" * 64}
        with (
            mock.patch("tanat_ai40.evaluate_ai42.load_actor", return_value=(actor, lineage)),
            mock.patch("tanat_ai40.evaluate_ai42.AssaultVectorEnv", _OneStepEnv),
        ):
            metrics = evaluate_vs_ai30(
                Path("checkpoint.pt"), Path("config.json"), Path("assaultenv.exe"),
                matches=1, workers=1, max_steps=1, device=torch.device("cpu"), seed=1,
            )
        profile = metrics["runtime_profile"]
        self.assertEqual(profile["version"], 1)
        self.assertEqual(profile["backend"], "torch")
        self.assertEqual(profile["device"], "cpu")
        self.assertEqual(profile["policy_batches"], 1)
        self.assertEqual(profile["policy_rows"], HERO_COUNT // 2)
        self.assertIsNone(profile["cuda_memory"])
        self.assertIsNone(profile["decision_margin"])
        self.assertGreaterEqual(profile["model_inference_seconds"], 0.0)
        self.assertGreaterEqual(profile["environment_step_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
