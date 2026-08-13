from __future__ import annotations

import os
import argparse
from pathlib import Path
import unittest

import numpy as np

from tanat_ai40.env import (
    AI40_ROSTER, AI40_SELF_PLAY_CONTROLLERS, AssaultEnvProcess, HeroAction,
    HERO_COUNT, SCHEMA_HASH, CONTROLLER_AI30, self_play_rosters,
)
from tanat_ai40.train_matches import build_schedule, policy_actor_mask
from tanat_ai40.train_async import partition_workers
from tanat_ai40.evaluate_ai30 import controllers_for_side, wilson_interval
from tanat_ai40.train_campaign import evaluation_plan


class AssaultEnvProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("TANAT_ASSAULTENV")
        if not configured:
            raise unittest.SkipTest("set TANAT_ASSAULTENV to the built Go runner")
        cls.executable = Path(configured)

    def run_trajectory(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=99, max_steps=50)
            snapshots = []
            for _ in range(45):
                result = env.step([HeroAction() for _ in range(HERO_COUNT)])
                snapshots.append((result.hero.copy(), result.entities.copy(), result.entity_mask.copy()))
            return result, snapshots

    def test_schema_shapes_and_determinism(self):
        first, first_trajectory = self.run_trajectory()
        second, second_trajectory = self.run_trajectory()
        self.assertEqual(first.step, 45)
        self.assertEqual(first.hero.shape, (10, 32))
        self.assertEqual(first.entities.shape, (10, 96, 16))
        self.assertEqual(first.skill_target_mask.shape, (10, 4, 96))
        self.assertGreater(int(first.entity_mask.sum()), 0)
        for a, b in zip(first_trajectory, second_trajectory):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_ai30_teacher_controller(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=101, max_steps=10,
                               controllers=[CONTROLLER_AI30] * HERO_COUNT)
            self.assertEqual(result.step, 0)
            result = env.step([HeroAction() for _ in range(HERO_COUNT)])
            self.assertEqual(result.step, 1)

    def test_ai40_mirror_self_play_controller(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=103, max_steps=10,
                               controllers=AI40_SELF_PLAY_CONTROLLERS)
            self.assertEqual(result.step, 0)
            result = env.step([HeroAction() for _ in range(HERO_COUNT)])
            self.assertEqual(result.step, 1)
            self.assertEqual(int(result.invalid.sum()), 0)


class SelfPlayRosterTest(unittest.TestCase):
    def test_rosters_are_reproducible_full_pool_permutations(self):
        first = self_play_rosters(np.random.default_rng(77), 4)
        second = self_play_rosters(np.random.default_rng(77), 4)
        self.assertEqual(first, second)
        expected = sorted(AI40_ROSTER.tolist())
        for roster in first:
            self.assertEqual(sorted(roster), expected)
        self.assertGreater(len({tuple(roster[:5]) for roster in first}), 1)

    def test_negative_roster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self_play_rosters(np.random.default_rng(1), -1)


class MixedTrainingScheduleTest(unittest.TestCase):
    def test_balanced_schedule_and_teacher_side_swap(self):
        schedule = build_schedule(3, 4)
        self.assertEqual(sum(match.opponent == "ai40" for match in schedule), 3)
        teachers = [match for match in schedule if match.opponent == "ai30"]
        self.assertEqual([match.ai40_side for match in teachers], [1, 2, 1, 2])

    def test_resumed_teacher_schedule_preserves_side_parity(self):
        teachers = build_schedule(0, 3, teacher_start_index=5)
        self.assertEqual([match.ai40_side for match in teachers], [2, 1, 2])

    def test_scripted_teacher_slots_are_excluded_from_policy_mask(self):
        teacher = next(match for match in build_schedule(0, 1))
        mask = policy_actor_mask([teacher])
        self.assertEqual(int(mask.sum()), HERO_COUNT // 2)
        expected = np.asarray(
            [controller != CONTROLLER_AI30 for controller in teacher.controllers],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_empty_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule(0, 0)


class AsyncRolloutGroupTest(unittest.TestCase):
    def test_partition_workers_keeps_remainder_group(self):
        self.assertEqual(partition_workers(50, 8), [8, 8, 8, 8, 8, 8, 2])

    def test_partition_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            partition_workers(0, 4)
        with self.assertRaises(ValueError):
            partition_workers(4, 0)


class AI30EvaluationTest(unittest.TestCase):
    def test_controller_sides_are_balanced(self):
        first = controllers_for_side(1)
        second = controllers_for_side(2)
        self.assertEqual(first[:5], (3,) * 5)
        self.assertEqual(first[5:], (2,) * 5)
        self.assertEqual(second[:5], (2,) * 5)
        self.assertEqual(second[5:], (3,) * 5)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(120, 200)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)
        self.assertGreater(low, 0.5)

    def test_empty_wilson_interval_is_uninformative(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_adaptive_evaluation_plan(self):
        args = argparse.Namespace(
            eval_matches=50, eval_medium_matches=200, eval_final_matches=500,
            eval_medium_win_rate=0.4, eval_final_win_rate=0.55,
        )
        self.assertEqual(evaluation_plan(args), (
            ("quick", 50, 0.4),
            ("medium", 200, 0.55),
            ("confirmation", 500, None),
        ))


if __name__ == "__main__":
    unittest.main()
