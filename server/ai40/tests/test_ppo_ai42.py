from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tanat_ai40.env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
)
from tanat_ai40.model_ai42_actor import AI42Actor
from tanat_ai40.ppo_ai42 import (
    AI42ActorCritic,
    AI42PPOConfig,
    AI42Rollout,
    AI42ValueHead,
    PPO_CHECKPOINT_FORMAT,
    _action_distributions,
    _sequence_forward,
    generalized_advantage_estimate,
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)


def small_model() -> AI42ActorCritic:
    actor = AI42Actor(
        hidden_size=16, model_width=16, entity_layers=1,
        num_heads=4, ff_multiplier=2,
    )
    return AI42ActorCritic(actor, AI42ValueHead(16))


class AI42PPOTests(unittest.TestCase):
    def test_value_head_is_training_only_and_actor_contract_stays_clean(self) -> None:
        model = small_model()
        batch = 3
        output = model(
            torch.zeros(batch, HERO_FEATURES),
            torch.zeros(batch, ABILITY_COUNT, ABILITY_FEATURES),
            torch.zeros(batch, 4, ENTITY_FEATURES),
            torch.zeros(batch, GLOBAL_FEATURES),
            torch.ones(batch, 4, dtype=torch.bool),
        )
        self.assertEqual(output["value"].shape, (batch,))
        torch.testing.assert_close(output["value"], torch.zeros(batch))
        self.assertTrue(model.critic.training_only)
        self.assertFalse(model.actor.has_value_head)
        self.assertNotIn("value", model.actor(
            torch.zeros(batch, HERO_FEATURES),
            torch.zeros(batch, ABILITY_COUNT, ABILITY_FEATURES),
            torch.zeros(batch, 4, ENTITY_FEATURES),
            torch.zeros(batch, GLOBAL_FEATURES),
            torch.ones(batch, 4, dtype=torch.bool),
        ))

    def test_runtime_sampler_masks_hold_without_an_active_order(self) -> None:
        rows = 128
        control = torch.full((rows, 4), -10.0)
        control[:, 1] = 20.0  # HOLD would dominate if the runtime mask were absent.
        control[:, 0] = 0.0
        output = {
            "control": control,
            "kind": torch.zeros(rows, ACTION_KINDS),
            "target": torch.zeros(rows, ACTION_KINDS, MAX_ENTITIES),
            "offset": torch.zeros(rows, ACTION_KINDS, 81),
            "anchor": torch.zeros(rows, ACTION_KINDS, 15),
        }
        kind_mask = torch.zeros(rows, ACTION_KINDS, dtype=torch.bool)
        kind_mask[:, 1] = True
        target_mask = torch.zeros(rows, MAX_ENTITIES, dtype=torch.bool)
        skill_mask = torch.zeros(rows, 4, MAX_ENTITIES, dtype=torch.bool)
        entity_mask = torch.zeros(rows, MAX_ENTITIES, dtype=torch.bool)
        controls, actions, log_prob, entropy = _action_distributions(
            output, kind_mask, target_mask, skill_mask, entity_mask,
            torch.zeros(rows, dtype=torch.bool),
        )
        self.assertFalse(bool((controls == 1).any()))
        self.assertTrue(bool(torch.isfinite(log_prob).all()))
        self.assertTrue(bool(torch.isfinite(entropy).all()))
        self.assertEqual(actions.shape, (rows, 4))

    def test_gae_stops_at_match_boundary(self) -> None:
        rollout = self._rollout(steps=2, actors=1)
        rollout.rewards[:] = np.asarray([[1.0], [2.0]])
        rollout.old_values[:] = 0
        rollout.bootstrap_values[:] = 100
        rollout.dones[:] = np.asarray([[False], [True]])
        config = AI42PPOConfig(gamma=0.5, gae_lambda=1.0)
        advantages, returns = generalized_advantage_estimate(rollout, config)
        np.testing.assert_allclose(advantages[:, 0], [2.0, 2.0])
        np.testing.assert_allclose(returns, advantages)

    def test_recurrent_ppo_update_is_finite_and_changes_parameters(self) -> None:
        torch.manual_seed(11)
        np.random.seed(11)
        model = small_model()
        rollout = self._rollout(steps=3, actors=4)
        with torch.no_grad():
            log_prob, _, values = _sequence_forward(
                model, rollout, np.arange(rollout.actors), torch.device("cpu"),
            )
        rollout.old_log_prob[:] = log_prob.numpy()
        rollout.old_values[:] = values.numpy()
        rollout.bootstrap_values[:] = 0
        rollout.rewards[:] = np.linspace(-0.2, 0.4, rollout.rewards.size).reshape(
            rollout.rewards.shape,
        )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = ppo_update(
            model, optimizer, rollout,
            AI42PPOConfig(update_epochs=1, minibatch_actors=2, target_kl=1.0),
            torch.device("cpu"), mixed_precision=False,
        )
        self.assertTrue(np.isfinite(float(metrics["loss"])))
        self.assertTrue(any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        ))

    def test_reward_free_opening_does_not_move_actor(self) -> None:
        model = small_model()
        rollout = self._rollout(steps=2, actors=2)
        with torch.no_grad():
            log_prob, _, values = _sequence_forward(
                model, rollout, np.arange(rollout.actors), torch.device("cpu"),
            )
        rollout.old_log_prob[:] = log_prob.numpy()
        rollout.old_values[:] = values.numpy()
        actor_before = {
            name: value.detach().clone()
            for name, value in model.actor.state_dict().items()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = ppo_update(
            model, optimizer, rollout,
            AI42PPOConfig(update_epochs=1, minibatch_actors=2),
            torch.device("cpu"), mixed_precision=False,
        )
        self.assertTrue(metrics["cold_reward_free"])
        for name, value in model.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])

    def test_checkpoint_round_trip_is_weights_only_safe(self) -> None:
        model = small_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            save_ppo_checkpoint(
                path, model, optimizer, AI42PPOConfig(),
                model_kwargs={
                    "hidden_size": 16, "model_width": 16, "entity_layers": 1,
                    "num_heads": 4, "ff_multiplier": 2,
                },
                update=3, hero_steps=120, source_lineage={"kind": "test"},
                metrics={"loss": 1.0},
            )
            raw = torch.load(path, weights_only=True)
            self.assertEqual(raw["format"], PPO_CHECKPOINT_FORMAT)
            restored = small_model()
            payload = load_ppo_checkpoint(path, restored)
            self.assertEqual(payload["update"], 3)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])

    @staticmethod
    def _rollout(steps: int, actors: int) -> AI42Rollout:
        observations = {
            "hero": np.zeros((steps, actors, HERO_FEATURES), np.float32),
            "abilities": np.zeros(
                (steps, actors, ABILITY_COUNT, ABILITY_FEATURES), np.float32,
            ),
            "entities": np.zeros(
                (steps, actors, MAX_ENTITIES, ENTITY_FEATURES), np.float32,
            ),
            "global_state": np.zeros((steps, actors, GLOBAL_FEATURES), np.float32),
            "entity_mask": np.zeros((steps, actors, MAX_ENTITIES), np.bool_),
            "kind_mask": np.zeros((steps, actors, ACTION_KINDS), np.bool_),
            "target_mask": np.zeros((steps, actors, MAX_ENTITIES), np.bool_),
            "skill_target_mask": np.zeros(
                (steps, actors, 4, MAX_ENTITIES), np.bool_,
            ),
        }
        observations["entity_mask"][:, :, 0] = True
        observations["kind_mask"][:, :, 1] = True
        controls = np.zeros((steps, actors), np.int64)
        actions = np.zeros((steps, actors, 4), np.int64)
        actions[:, :, 0] = 1
        reset = np.zeros((steps, actors), np.bool_)
        reset[0] = True
        return AI42Rollout(
            observations=observations,
            controls=controls,
            actions=actions,
            active_order=np.zeros((steps, actors), np.bool_),
            reset_mask=reset,
            policy_mask=np.ones((steps, actors), np.bool_),
            rewards=np.zeros((steps, actors), np.float32),
            dones=np.zeros((steps, actors), np.bool_),
            old_log_prob=np.zeros((steps, actors), np.float32),
            old_values=np.zeros((steps, actors), np.float32),
            initial_h=np.zeros((actors, 16), np.float32),
            initial_c=np.zeros((actors, 16), np.float32),
            bootstrap_values=np.zeros(actors, np.float32),
        )


if __name__ == "__main__":
    unittest.main()
