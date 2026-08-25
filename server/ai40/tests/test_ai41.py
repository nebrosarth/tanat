from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from tanat_ai40.model import AI40Policy
from tanat_ai40.env import (
    AI41_NAVIGATION_SCHEMA_HASH, AI41_STRATEGIC_REWARD_HASH,
    AI41_STRATEGIC_REWARD_HASH_V4,
)
from tanat_ai40.migrate_ai41_navigation import migrate_navigation_state
from tanat_ai40.migrate_ai41_strategic import migrate_checkpoint
from tanat_ai40.model_ai41 import AI41NavigationPolicy, AI41Policy, migrate_ai40_state
from tanat_ai40.train import combined_log_prob, distributions, select_parameter_logits
from tanat_ai40.train_async import ActionSampler
from tanat_ai40.evaluate_ai30 import EvaluationActor, controlled_slot_indices, greedy_actions


class AI41PolicyTest(unittest.TestCase):
    def test_evaluation_only_selects_the_controlled_team_slots(self):
        candidate = controlled_slot_indices([1, 2, 0]).tolist()
        opponent = controlled_slot_indices([1, 2, 0], candidate=False).tolist()
        self.assertEqual(candidate, list(range(5)) + list(range(15, 20)) + list(range(20, 25)))
        self.assertEqual(opponent, list(range(5, 10)) + list(range(10, 15)) + list(range(25, 30)))
        self.assertEqual(set(candidate) | set(opponent), set(range(30)))
        self.assertFalse(set(candidate) & set(opponent))

    def test_evaluation_actor_matches_eager_argmax_for_selected_slots(self):
        torch.manual_seed(29)
        policy = AI41Policy().eval()
        rows = 10
        observation = SimpleNamespace(batched_observations={
            "hero": np.random.default_rng(29).normal(size=(rows, 32)).astype(np.float32),
            "abilities": np.zeros((rows, 4, 40), dtype=np.float32),
            "entities": np.random.default_rng(30).normal(size=(rows, 96, 16)).astype(np.float32),
            "global_state": np.zeros((rows, 32), dtype=np.float32),
            "entity_mask": np.ones((rows, 96), dtype=np.uint8),
            "kind_mask": np.ones((rows, 8), dtype=np.uint8),
            "target_mask": np.ones((rows, 96), dtype=np.uint8),
            "skill_target_mask": np.ones((rows, 4, 96), dtype=np.uint8),
        })
        device = torch.device("cpu")
        full_h, full_c = policy.initial_state(rows, device)
        expected, _, _ = greedy_actions(policy, observation, full_h, full_c, device)
        actor = EvaluationActor(policy, 5, device)
        h, c = actor.initial_state()
        actual, _, _ = actor.act(observation, np.arange(5, dtype=np.intp), h, c)
        np.testing.assert_array_equal(actual, expected[:5])

    def inputs(self, batch: int = 5):
        hero = torch.randn(batch, 32)
        hero[:, 0] = torch.randint(1, 100, (batch,)) / 100
        abilities = torch.randn(batch, 4, 40)
        entities = torch.randn(batch, 96, 16)
        global_state = torch.randn(batch, 32)
        entity_mask = torch.rand(batch, 96) > 0.2
        return hero, abilities, entities, global_state, entity_mask

    def test_conditioned_head_shapes_and_selection(self):
        policy = AI41Policy()
        hero, abilities, entities, global_state, entity_mask = self.inputs()
        h, c = policy.initial_state(hero.shape[0], torch.device("cpu"))
        output = policy(hero, abilities, entities, global_state, entity_mask, h, c)
        self.assertEqual(output["kind"].shape, (5, 8))
        self.assertEqual(output["target"].shape, (5, 8, 96))
        self.assertEqual(output["direction"].shape, (5, 8, 16))
        self.assertEqual(output["distance"].shape, (5, 8, 3))
        kinds = torch.tensor([0, 1, 3, 5, 7])
        selected = select_parameter_logits(output, kinds)
        self.assertEqual(selected["target"].shape, (5, 96))
        masks = torch.ones(5, 8, dtype=torch.bool), torch.ones(5, 96, dtype=torch.bool)
        self.assertEqual(distributions(output, *masks, kinds)[1].logits.shape, (5, 96))

    def test_ai40_migration_is_behavior_preserving(self):
        torch.manual_seed(7)
        old = AI40Policy().eval()
        new = AI41Policy().eval()
        migrate_ai40_state(new, old.state_dict())
        hero, abilities, entities, global_state, entity_mask = self.inputs()
        h, c = old.initial_state(hero.shape[0], torch.device("cpu"))
        with torch.no_grad():
            old_output = old(hero, entities, global_state, entity_mask, h, c)
            new_output = new(hero, abilities, entities, global_state, entity_mask, h, c)
        for name in ("kind", "value", "h", "c"):
            torch.testing.assert_close(old_output[name], new_output[name])
        for kind in range(8):
            torch.testing.assert_close(old_output["target"], new_output["target"][:, kind])
            torch.testing.assert_close(old_output["direction"], new_output["direction"][:, kind])
            torch.testing.assert_close(old_output["distance"], new_output["distance"][:, kind])

    def test_migrated_ability_residuals_receive_gradients(self):
        old = AI40Policy()
        new = AI41Policy()
        migrate_ai40_state(new, old.state_dict())
        hero, abilities, entities, global_state, entity_mask = self.inputs(batch=8)
        h, c = new.initial_state(hero.shape[0], torch.device("cpu"))
        output = new(hero, abilities, entities, global_state, entity_mask, h, c)
        kinds = torch.tensor([3, 4, 5, 6, 3, 4, 5, 6])
        rows = torch.arange(8)
        loss = (
            output["target"][rows, kinds].square().mean()
            + output["direction"][rows, kinds].square().mean()
            + output["distance"][rows, kinds].square().mean()
        )
        loss.backward()
        self.assertGreater(float(new.ability_action.weight.grad.abs().sum()), 0)
        self.assertGreater(float(new.ability_hero.weight.grad.abs().sum()), 0)

    def test_navigation_policy_uses_grid_and_anchor_heads(self):
        policy = AI41NavigationPolicy()
        hero, abilities, entities, global_state, entity_mask = self.inputs()
        h, c = policy.initial_state(hero.shape[0], torch.device("cpu"))
        output = policy(hero, abilities, entities, global_state, entity_mask, h, c)
        self.assertEqual(output["direction"].shape, (5, 8, 81))
        self.assertEqual(output["distance"].shape, (5, 8, 15))

    def test_navigation_migration_preserves_core_and_prefers_local_grid(self):
        old = AI41Policy()
        migrated = migrate_navigation_state(old.state_dict())
        torch.testing.assert_close(
            migrated["core.weight_ih"], old.state_dict()["core.weight_ih"],
        )
        self.assertEqual(migrated["direction_head.weight"].shape[0], 81)
        self.assertEqual(migrated["distance_head.weight"].shape[0], 15)
        probabilities = migrated["distance_head.bias"].softmax(0)
        self.assertGreater(float(probabilities[0]), 0.995)
        self.assertGreater(float(probabilities[1:].sum()), 0.004)

    def test_strategic_migration_accepts_v4_and_resets_optimizer_for_reward_v5(self):
        policy = AI41NavigationPolicy()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "v4.pt"
            output = Path(directory) / "v5.pt"
            torch.save({
                "model": policy.state_dict(),
                "schema_hash": AI41_NAVIGATION_SCHEMA_HASH.hex(),
                "reward_hash": AI41_STRATEGIC_REWARD_HASH_V4.hex(),
                "update": 17,
                "hero_steps": 1234,
                "config": {"completed_matches": {"ai40": 99}},
            }, source)
            migrate_checkpoint(source, output)
            migrated = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(migrated["reward_hash"], AI41_STRATEGIC_REWARD_HASH.hex())
        self.assertEqual(migrated["config"]["completed_matches"], {})
        self.assertEqual(migrated["config"]["tanat_creep_last_hit_bonus"], 0.24)
        self.assertTrue(migrated["config"]["optimizer_reinitialized_for_reward_contract"])


class ActionSamplerTest(unittest.TestCase):
    def test_conditioned_sampling_respects_masks_and_log_probability(self):
        torch.manual_seed(17)
        batch, kinds, targets = 64, 8, 96
        output = {
            "kind": torch.randn(batch, kinds),
            "target": torch.randn(batch, kinds, targets),
            "direction": torch.randn(batch, kinds, 16),
            "distance": torch.randn(batch, kinds, 3),
        }
        kind_mask = torch.rand(batch, kinds) > 0.25
        kind_mask[:, 0] = True
        target_mask = torch.rand(batch, targets) > 0.5
        skill_target_mask = torch.rand(batch, 4, targets) > 0.6
        target_mask[:, 0] = True
        skill_target_mask[:, :, 0] = True
        sampler = ActionSampler(False)

        actions, effective_mask, log_prob = sampler(
            output, kind_mask, target_mask, skill_target_mask,
        )
        rows = torch.arange(batch)
        factors = tuple(actions[:, index] for index in range(4))
        expected = combined_log_prob(
            distributions(output, kind_mask, effective_mask, factors[0]), factors,
        )

        self.assertTrue(bool(kind_mask[rows, factors[0]].all()))
        self.assertTrue(bool(effective_mask[rows, factors[1]].all()))
        torch.testing.assert_close(log_prob, expected, rtol=1e-5, atol=1e-6)
        next_actions = sampler(output, kind_mask, target_mask, skill_target_mask)[0]
        self.assertFalse(torch.equal(actions, next_actions))

    def test_navigation_sampler_uses_anchor_only_for_move_log_probability(self):
        torch.manual_seed(23)
        batch, kinds, targets = 32, 8, 96
        output = {
            "kind": torch.randn(batch, kinds),
            "target": torch.randn(batch, kinds, targets),
            "direction": torch.randn(batch, kinds, 81),
            "distance": torch.randn(batch, kinds, 15),
        }
        kind_mask = torch.ones(batch, kinds, dtype=torch.bool)
        target_mask = torch.ones(batch, targets, dtype=torch.bool)
        skill_target_mask = torch.ones(batch, 4, targets, dtype=torch.bool)
        sampler = ActionSampler(False)
        actions, effective_mask, log_prob = sampler(
            output, kind_mask, target_mask, skill_target_mask,
        )
        factors = tuple(actions[:, index] for index in range(4))
        expected = combined_log_prob(
            distributions(output, kind_mask, effective_mask, factors[0]), factors,
        )
        torch.testing.assert_close(log_prob, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
