from __future__ import annotations

import inspect
import unittest

import torch

from tanat_ai40.env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from tanat_ai40.model_ai42_actor import (
    AI42Actor,
    CONTROL_CLASSES,
    parameter_count,
    selected_action_logits,
)


class AI42ActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.policy = AI42Actor(
            hidden_size=40,
            model_width=32,
            entity_layers=2,
            num_heads=4,
            ff_multiplier=2,
        ).eval()
        self.hero, self.abilities, self.entities, self.global_state, self.mask = self.inputs()

    @staticmethod
    def inputs(batch: int = 3, slots: int = 7):
        torch.manual_seed(17)
        hero = torch.randn(batch, HERO_FEATURES)
        hero[:, 0] = torch.tensor([0.07, 0.13, 0.23])[:batch]
        abilities = torch.randn(batch, ABILITY_COUNT, ABILITY_FEATURES)
        entities = torch.randn(batch, slots, ENTITY_FEATURES)
        global_state = torch.randn(batch, GLOBAL_FEATURES)
        mask = torch.tensor(
            [[True, False, True, True, False, False, False],
             [True, True, False, True, False, False, False],
             [False, False, False, False, False, False, False]],
            dtype=torch.bool,
        )[:batch, :slots]
        return hero, abilities, entities, global_state, mask

    def run_policy(self, policy=None, *, state=None, **inputs):
        policy = self.policy if policy is None else policy
        values = (
            self.hero if "hero" not in inputs else inputs["hero"],
            self.abilities if "abilities" not in inputs else inputs["abilities"],
            self.entities if "entities" not in inputs else inputs["entities"],
            self.global_state if "global_state" not in inputs else inputs["global_state"],
            self.mask if "entity_mask" not in inputs else inputs["entity_mask"],
        )
        if state is None:
            state = policy.initial_state(values[0].shape[0], torch.device("cpu"))
        with torch.no_grad():
            return policy(*values, *state)

    def test_output_contract_is_kind_conditioned_and_actor_only(self):
        output = self.run_policy()
        self.assertEqual(output["control"].shape, (3, CONTROL_CLASSES))
        self.assertEqual(output["kind"].shape, (3, ACTION_KINDS))
        self.assertEqual(output["target"].shape, (3, ACTION_KINDS, 7))
        self.assertEqual(output["offset"].shape, (3, ACTION_KINDS, NAVIGATION_OFFSETS))
        self.assertEqual(output["anchor"].shape, (3, ACTION_KINDS, NAVIGATION_ANCHORS))
        self.assertIs(output["offset"], output["direction"])
        self.assertIs(output["anchor"], output["distance"])
        self.assertNotIn("value", output)
        forward_parameters = inspect.signature(AI42Actor.forward).parameters
        self.assertNotIn("critic", forward_parameters)
        self.assertNotIn("full_state", forward_parameters)

    def test_structured_heads_preserve_public_probabilities_and_grid(self):
        output = self.run_policy()
        torch.testing.assert_close(
            output["control"].exp().sum(dim=-1),
            torch.ones(output["control"].shape[0]),
        )
        torch.testing.assert_close(
            output["kind"].exp().sum(dim=-1),
            torch.ones(output["kind"].shape[0]),
        )

        grid = output["offset"].reshape(3, ACTION_KINDS, 9, 9)
        interaction = (
            grid
            - grid[..., :, :1]
            - grid[..., :1, :]
            + grid[..., :1, :1]
        )
        torch.testing.assert_close(interaction, torch.zeros_like(interaction))

    def test_parameter_estimate_matches_structured_actor(self):
        from tanat_ai40.smoke_ai42_actor import _estimated_model_parameters

        expected = _estimated_model_parameters(
            hidden_size=40,
            model_width=32,
            entity_layers=2,
            ff_multiplier=2,
        )
        self.assertEqual(parameter_count(self.policy), expected)

    def test_entity_slot_permutation_only_permutates_target_logits(self):
        output = self.run_policy()
        permutation = torch.tensor([2, 0, 3, 1, 6, 5, 4])
        permuted = self.run_policy(
            entities=self.entities[:, permutation],
            entity_mask=self.mask[:, permutation],
        )
        for key in ("control", "kind", "offset", "anchor", "h", "c"):
            torch.testing.assert_close(output[key], permuted[key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            permuted["target"], output["target"][:, :, permutation], rtol=1e-5, atol=1e-6,
        )

    def test_masked_entities_cannot_change_any_output(self):
        changed = self.entities.clone()
        changed[~self.mask] = torch.randn_like(changed[~self.mask]) * 1000
        output = self.run_policy()
        changed_output = self.run_policy(entities=changed)
        for key in output:
            torch.testing.assert_close(output[key], changed_output[key], rtol=1e-5, atol=1e-6)

    def test_all_masked_observation_is_finite(self):
        output = self.run_policy(entity_mask=torch.zeros_like(self.mask))
        for key, value in output.items():
            self.assertTrue(bool(torch.isfinite(value).all()), key)

    def test_all_masked_attention_has_no_tensor_to_python_branch(self):
        forward_source = inspect.getsource(type(self.policy.entity_attention[0]).forward)
        self.assertNotIn("bool(", forward_source)
        self.assertNotRegex(forward_source, r"if\s+empty")

    def test_hero_and_ability_specialization_affect_the_actor(self):
        changed_hero = self.hero.clone()
        changed_hero[0, 0] = 0.99
        hero_output = self.run_policy(hero=changed_hero)
        self.assertGreater(
            float((hero_output["kind"] - self.run_policy()["kind"]).abs().sum()), 1e-6,
        )

        changed_abilities = self.abilities.clone()
        changed_abilities[0, 0, 0] += 10
        ability_output = self.run_policy(abilities=changed_abilities)
        self.assertGreater(
            float((ability_output["kind"] - self.run_policy()["kind"]).abs().sum()), 1e-6,
        )

    def test_each_ability_token_scores_its_own_skill_kind(self):
        with torch.no_grad():
            self.policy.ability_to_hero.weight.zero_()
            self.policy.ability_to_hero.bias.zero_()
        baseline = self.run_policy()
        changed = self.abilities.clone()
        changed[:, 0, 0] += 10
        updated = self.run_policy(abilities=changed)
        self.assertGreater(float((updated["kind"][:, 3] - baseline["kind"][:, 3]).abs().sum()), 1e-6)
        torch.testing.assert_close(updated["kind"][:, :3], baseline["kind"][:, :3])
        torch.testing.assert_close(updated["kind"][:, 7], baseline["kind"][:, 7])
        torch.testing.assert_close(
            updated["kind"][:, 4:7] - updated["kind"][:, 4:5],
            baseline["kind"][:, 4:7] - baseline["kind"][:, 4:5],
        )

    def test_recurrent_state_changes_per_hero_temporal_outputs(self):
        initial = self.policy.initial_state(3, torch.device("cpu"))
        first = self.run_policy(state=initial)
        second = self.run_policy(state=(first["h"], first["c"]))
        self.assertGreater(float((first["h"] - initial[0]).abs().sum()), 1e-6)
        self.assertGreater(float((second["kind"] - first["kind"]).abs().sum()), 1e-6)

        altered_state = (first["h"].clone(), first["c"].clone())
        altered_state[0][1] += 1
        altered = self.run_policy(state=altered_state)
        self.assertGreater(float((altered["kind"][1] - second["kind"][1]).abs().sum()), 1e-6)
        for row in (0, 2):
            torch.testing.assert_close(altered["kind"][row], second["kind"][row])
            torch.testing.assert_close(altered["h"][row], second["h"][row])
            torch.testing.assert_close(altered["c"][row], second["c"][row])

    def test_entity_count_is_representable(self):
        entities = torch.zeros(1, 4, ENTITY_FEATURES)
        mask_one = torch.tensor([[True, False, False, False]])
        mask_two = torch.tensor([[True, True, False, False]])
        hero = self.hero[:1]
        abilities = self.abilities[:1]
        global_state = self.global_state[:1]
        one = self.run_policy(
            hero=hero, abilities=abilities, entities=entities,
            global_state=global_state, entity_mask=mask_one,
        )
        two = self.run_policy(
            hero=hero, abilities=abilities, entities=entities,
            global_state=global_state, entity_mask=mask_two,
        )
        self.assertGreater(float((one["kind"] - two["kind"]).abs().sum()), 1e-6)

    def test_selected_action_logits_are_autoregressive_compatible(self):
        output = self.run_policy()
        kinds = torch.tensor([0, 3, 7])
        selected = selected_action_logits(output, kinds)
        self.assertEqual(selected["target"].shape, (3, 7))
        self.assertEqual(selected["offset"].shape, (3, NAVIGATION_OFFSETS))
        self.assertEqual(selected["anchor"].shape, (3, NAVIGATION_ANCHORS))
        torch.testing.assert_close(selected["direction"], selected["offset"])
        torch.testing.assert_close(selected["distance"], selected["anchor"])

    def test_production_default_is_in_target_parameter_range(self):
        policy = AI42Actor()
        self.assertGreaterEqual(parameter_count(policy), 2_000_000)
        self.assertLessEqual(parameter_count(policy), 6_000_000)


if __name__ == "__main__":
    unittest.main()
