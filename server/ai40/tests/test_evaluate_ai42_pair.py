from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from tanat_ai40.env import (
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
from tanat_ai40.evaluate_ai42_pair import evaluate_pair


class _Actor(torch.nn.Module):
    hidden_size = 4

    def __init__(self, kind: int) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.kind = kind

    def initial_state(self, batch, device=None, dtype=None):
        dtype = dtype or torch.float32
        return (
            torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
            torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(self, hero, abilities, entities, global_state, entity_mask, h, c):
        batch = hero.shape[0]
        control = torch.zeros(batch, 4, device=hero.device)
        control[:, 0] = 10
        kind = torch.zeros(batch, ACTION_KINDS, device=hero.device)
        kind[:, self.kind] = 10
        return {
            "control": control,
            "kind": kind,
            "target": torch.zeros(batch, ACTION_KINDS, MAX_ENTITIES, device=hero.device),
            "offset": torch.zeros(batch, ACTION_KINDS, NAVIGATION_OFFSETS, device=hero.device),
            "anchor": torch.zeros(batch, ACTION_KINDS, NAVIGATION_ANCHORS, device=hero.device),
            "h": h + 1,
            "c": c + 1,
        }


class _Result:
    batched_observations = None

    def __init__(self, *, done=False, winner=0):
        self.hero = np.zeros((HERO_COUNT, HERO_FEATURES), np.float32)
        self.abilities = np.zeros(
            (HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES), np.float32,
        )
        self.entities = np.zeros(
            (HERO_COUNT, MAX_ENTITIES, ENTITY_FEATURES), np.float32,
        )
        self.global_state = np.zeros((HERO_COUNT, GLOBAL_FEATURES), np.float32)
        self.entity_mask = np.ones((HERO_COUNT, MAX_ENTITIES), np.uint8)
        self.kind_mask = np.ones((HERO_COUNT, ACTION_KINDS), np.uint8)
        self.target_mask = np.ones((HERO_COUNT, MAX_ENTITIES), np.uint8)
        self.skill_target_mask = np.ones((HERO_COUNT, 4, MAX_ENTITIES), np.uint8)
        self.active_order = np.zeros(HERO_COUNT, np.uint8)
        self.teacher_actions = None
        self.teacher_valid = None
        self.done = done
        self.winner = winner
        self.step = 10
        self.rewards = np.zeros(HERO_COUNT, np.float32)
        self.invalid = np.zeros(HERO_COUNT, np.uint8)


class _Env:
    def __init__(self, _path, workers, _protocol):
        self.workers = workers
        self.last_actions = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def reset(self, *_args, **_kwargs):
        return [_Result() for _ in range(self.workers)]

    def step(self, actions):
        self.last_actions = actions.copy()
        # Candidate is side 1 in worker 0 and side 2 in worker 1.
        assert np.all(actions[0, :5, 1] == 1)
        assert np.all(actions[0, 5:, 1] == 2)
        assert np.all(actions[1, :5, 1] == 2)
        assert np.all(actions[1, 5:, 1] == 1)
        return [_Result(done=True, winner=1), _Result(done=True, winner=1)]

    def reset_indices(self, *_args, **_kwargs):
        raise AssertionError("two terminal matches do not require replacement")


class PairEvaluationTests(unittest.TestCase):
    def test_balances_sides_and_drives_both_policies(self) -> None:
        actors = [(_Actor(1), {"id": "candidate"}), (_Actor(2), {"id": "champion"})]
        with (
            mock.patch("tanat_ai40.evaluate_ai42_pair.load_actor", side_effect=actors),
            mock.patch("tanat_ai40.evaluate_ai42_pair.AssaultVectorEnv", _Env),
        ):
            result = evaluate_pair(
                Path("candidate.pt"), Path("champion.pt"), Path("config.json"),
                Path("assaultenv.exe"), matches=2, workers=2, max_steps=10,
                device=torch.device("cpu"), seed=1,
            )
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["draws"], 0)
        self.assertEqual(result["side_1"]["win"], 1)
        self.assertEqual(result["side_2"]["loss"], 1)


if __name__ == "__main__":
    unittest.main()
