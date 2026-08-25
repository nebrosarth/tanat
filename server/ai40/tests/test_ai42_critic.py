from __future__ import annotations

import inspect
import os
import subprocess
import sys
import unittest

import torch

from tanat_ai40.critic_ai42 import (
    AI42CentralizedCritic,
    VALUE_HEAD_NAMES,
    parameter_count,
)
from tanat_ai40.env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
)


class AI42CentralizedCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.critic = AI42CentralizedCritic(test_size=True).eval()
        self.inputs = self.make_inputs()

    @staticmethod
    def make_inputs(batch: int = 3, entity_slots: int = 7):
        torch.manual_seed(17)
        heroes = torch.randn(batch, HERO_COUNT, HERO_FEATURES)
        abilities = torch.randn(batch, HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES)
        entities = torch.randn(batch, entity_slots, ENTITY_FEATURES)
        entity_mask = torch.tensor(
            [
                [True, False, True, True, False, False, False],
                [True, True, False, True, False, False, False],
                [False, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )[:batch, :entity_slots]
        global_state = torch.randn(batch, 2, GLOBAL_FEATURES)
        macro_plans = torch.randn(batch, 2, 6)
        assignments = torch.randn(batch, HERO_COUNT, ACTION_KINDS)
        executed_actions = torch.randn(batch, HERO_COUNT, 4)
        return (
            heroes,
            abilities,
            entities,
            entity_mask,
            global_state,
            macro_plans,
            assignments,
            executed_actions,
        )

    def run_critic(self, critic=None, inputs=None):
        critic = self.critic if critic is None else critic
        inputs = self.inputs if inputs is None else inputs
        with torch.no_grad():
            return critic(*inputs)

    def test_contract_and_training_only_marker(self):
        output = self.run_critic()
        self.assertTrue(self.critic.training_only)
        self.assertTrue(self.critic.accepts_privileged_state)
        self.assertEqual(tuple(output), VALUE_HEAD_NAMES)
        for name in VALUE_HEAD_NAMES:
            self.assertEqual(output[name].shape, (3, HERO_COUNT))
            self.assertTrue(bool(torch.isfinite(output[name]).all()), name)
        self.assertGreater(parameter_count(self.critic), 0)

    def test_entity_slot_permutation_is_value_invariant(self):
        output = self.run_critic()
        permutation = torch.tensor([2, 0, 3, 1, 6, 5, 4])
        permuted = list(self.inputs)
        permuted[2] = self.inputs[2][:, permutation]
        permuted[3] = self.inputs[3][:, permutation]
        permuted_output = self.run_critic(inputs=tuple(permuted))
        for name in VALUE_HEAD_NAMES:
            torch.testing.assert_close(
                output[name],
                permuted_output[name],
                rtol=2e-5,
                atol=2e-6,
            )

    def test_joint_hero_reasoning_is_slot_equivariant(self):
        output = self.run_critic()
        # Keep the permutation visibly nontrivial while covering every slot.
        permutation = torch.tensor([2, 0, 3, 1, 6, 7, 4, 9, 8, 5])
        permuted = list(self.inputs)
        for index in (0, 1, 6, 7):
            permuted[index] = self.inputs[index][:, permutation]
        permuted_output = self.run_critic(inputs=tuple(permuted))
        for name in VALUE_HEAD_NAMES:
            torch.testing.assert_close(
                permuted_output[name],
                output[name][:, permutation],
                rtol=2e-5,
                atol=2e-6,
            )

    def test_all_masked_entities_are_finite_and_padding_is_ignored(self):
        masked = list(self.inputs)
        masked[2] = torch.full_like(masked[2], float("nan"))
        masked[3] = torch.zeros_like(masked[3])
        output = self.run_critic(inputs=tuple(masked))
        for name, value in output.items():
            self.assertTrue(bool(torch.isfinite(value).all()), name)

        changed_padding = list(self.inputs)
        changed_padding[2] = self.inputs[2].clone()
        changed_padding[2][~self.inputs[3]] = 10000
        changed_output = self.run_critic(inputs=tuple(changed_padding))
        baseline = self.run_critic()
        for name in VALUE_HEAD_NAMES:
            torch.testing.assert_close(
                baseline[name],
                changed_output[name],
                rtol=2e-5,
                atol=2e-6,
            )

    def test_critic_has_no_actor_import_or_runtime_forward_hook(self):
        module = __import__("tanat_ai40.critic_ai42", fromlist=["*"])
        source = inspect.getsource(module)
        self.assertNotIn("model_ai42_actor", source)
        self.assertNotIn("export_onnx", source)
        self.assertNotIn("inference_server", source)
        parameters = inspect.signature(AI42CentralizedCritic.forward).parameters
        self.assertNotIn("actor", parameters)
        self.assertNotIn("runtime", parameters)

    def test_direct_critic_import_does_not_load_actor_module(self):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import tanat_ai40.critic_ai42; "
                    "print('\\n'.join(sorted(name for name in sys.modules "
                    "if name.startswith('tanat_ai40.'))))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        loaded = set(probe.stdout.splitlines())
        self.assertIn("tanat_ai40.critic_ai42", loaded)
        self.assertNotIn("tanat_ai40.model_ai42_actor", loaded)
        self.assertNotIn("tanat_ai40.export_onnx", loaded)
        self.assertNotIn("tanat_ai40.inference_server", loaded)


if __name__ == "__main__":
    unittest.main()
