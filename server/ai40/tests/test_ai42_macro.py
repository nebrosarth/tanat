from __future__ import annotations

import inspect
import unittest

import torch

from tanat_ai40.macro_ai42 import (
    ALLIED_HERO_COUNT,
    MACRO_MODES,
    NO_OBJECTIVE,
    RECOVER_MODE,
    AI42MacroPolicy,
    parameter_count,
)


class AI42MacroPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.policy = AI42MacroPolicy(
            hero_features=8,
            team_features=10,
            objective_features=6,
            previous_plan_features=7,
            model_width=24,
            transformer_layers=1,
            num_heads=4,
            ff_multiplier=2,
            role_count=4,
            lane_count=3,
            horizon_bins=3,
        ).eval()
        self.inputs = self.make_inputs()

    @staticmethod
    def make_inputs(batch: int = 2, objective_slots: int = 4):
        torch.manual_seed(7)
        heroes = torch.randn(batch, ALLIED_HERO_COUNT, 8)
        heroes[..., 0] = 1.0  # alive convention
        heroes[..., 1] = 0.0  # not retreating
        team = torch.randn(batch, 10)
        objectives = torch.randn(batch, objective_slots, 6)
        objective_mask = torch.tensor(
            [[True, False, True, False], [False, False, False, False]], dtype=torch.bool
        )[:batch, :objective_slots]
        previous_plan = torch.randn(batch, ALLIED_HERO_COUNT, 7)
        return heroes, team, objectives, objective_mask, previous_plan

    def run_decode(self, **overrides):
        values = list(self.inputs)
        names = ("allied_heroes", "team_state", "objectives", "objective_mask", "previous_plan")
        for index, name in enumerate(names):
            if name in overrides:
                values[index] = overrides[name]
        with torch.no_grad():
            return self.policy.greedy_decode(*values, **{
                key: value for key, value in overrides.items()
                if key in ("dead_mask", "retreating_mask")
            })

    def test_entity_team_heads_and_exact_five_assignments(self):
        output = self.policy(*self.inputs)
        self.assertEqual(output["mode"].shape, (2, 5, len(MACRO_MODES)))
        self.assertEqual(output["objective"].shape, (2, 5, 5))  # four slots + null
        self.assertEqual(output["role"].shape, (2, 5, 4))
        self.assertEqual(output["lane"].shape, (2, 5, 3))
        self.assertEqual(output["commitment"].shape, (2, 5, 2))
        self.assertEqual(output["horizon"].shape, (2, 5, 3))
        self.assertEqual(output["team_mode"].shape, (2, len(MACRO_MODES)))
        decoded = self.run_decode()
        for key in ("mode", "objective", "role", "lane", "commitment", "horizon"):
            self.assertEqual(decoded[key].shape, (2, 5), key)

    def test_masked_objectives_and_no_objective_state_are_finite_and_safe(self):
        output = self.policy(*self.inputs)
        for key, value in output.items():
            self.assertTrue(bool(torch.isfinite(value).all()), key)
        decoded = self.run_decode()
        self.assertTrue(bool((decoded["objective"][0] != 1).all()))
        self.assertTrue(bool((decoded["objective"][0] == NO_OBJECTIVE).logical_or(decoded["objective"][0] == 2).all()))
        self.assertTrue(bool((decoded["objective"][1] == NO_OBJECTIVE).all()))

    def test_dead_and_retreating_rows_force_recovery_and_no_commit(self):
        dead = torch.tensor([[True, False, False, False, False], [False] * 5])
        retreating = torch.tensor([[False] * 5, [False, True, False, False, False]])
        decoded = self.run_decode(dead_mask=dead, retreating_mask=retreating)
        self.assertEqual(int(decoded["mode"][0, 0]), RECOVER_MODE)
        self.assertEqual(int(decoded["commitment"][0, 0]), 0)
        self.assertEqual(int(decoded["mode"][1, 1]), RECOVER_MODE)
        self.assertEqual(int(decoded["commitment"][1, 1]), 0)
        self.assertEqual(int(decoded["objective"][0, 0]), NO_OBJECTIVE)

    def test_autoregressive_teacher_assignment_changes_later_logits(self):
        first = {key: torch.zeros(2, 5, dtype=torch.long) for key in ("mode", "objective", "role", "lane", "commitment", "horizon")}
        second = {key: value.clone() for key, value in first.items()}
        second["role"][:, 0] = 1
        first_output = self.policy(*self.inputs, teacher_assignments=first)
        second_output = self.policy(*self.inputs, teacher_assignments=second)
        self.assertGreater(float((first_output["role"][:, 1] - second_output["role"][:, 1]).abs().sum()), 1e-7)

    def test_selected_objective_embedding_is_batch_width(self):
        objective_memory = torch.randn(2, 4, self.policy.model_width)
        selections = {
            "mode": torch.zeros(2, dtype=torch.long),
            "objective": torch.tensor([2, NO_OBJECTIVE]),
            "role": torch.zeros(2, dtype=torch.long),
            "lane": torch.zeros(2, dtype=torch.long),
            "commitment": torch.zeros(2, dtype=torch.long),
            "horizon": torch.zeros(2, dtype=torch.long),
        }
        selected = self.policy._assignment_embedding(selections, objective_memory)
        self.assertEqual(selected.shape, (2, self.policy.model_width))

    def test_masked_objective_summary_has_no_tensor_to_python_branch(self):
        source = inspect.getsource(AI42MacroPolicy._encode)
        self.assertNotRegex(source, r"if\s+objective_mask\.any(?:\(\))?")
        self.assertIn("objective_count = objective_mask.sum", source)
        self.assertIn("torch.where", source)

    def test_decode_is_deterministic_and_has_no_micro_action_surface(self):
        first = self.run_decode()
        second = self.run_decode()
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        self.assertEqual(
            set(first), {"mode", "objective", "role", "lane", "commitment", "horizon", "team_mode", "assignment_mode", "recovery_mask"}
        )
        self.assertNotIn("x", first)
        self.assertNotIn("y", first)
        self.assertNotIn("skill", first)
        self.assertNotIn("action", first)

    def test_small_config_is_small_enough_for_unit_tests(self):
        self.assertGreater(parameter_count(self.policy), 0)
        self.assertLess(parameter_count(self.policy), 500_000)


if __name__ == "__main__":
    unittest.main()
