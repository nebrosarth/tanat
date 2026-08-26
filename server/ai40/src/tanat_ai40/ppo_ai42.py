"""On-policy recurrent PPO primitives for AI-42.

The runtime actor remains actor-only.  This module adds a training-only value
head, stochastic masked action sampling, recurrent rollout storage, GAE, and
the clipped PPO update.  Rollouts are kept time-major on the host and replayed
as contiguous per-hero sequences so gradients flow through the recurrent core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .env import ACTION_KINDS, HERO_COUNT, NAVIGATION_ANCHORS, NAVIGATION_OFFSETS
from .model_ai42_actor import (
    AI42Actor,
    CONTROL_CANCEL,
    CONTROL_HOLD,
    CONTROL_ISSUE,
    CONTROL_WAIT,
)
from .train import stack_observations


RUNTIME_CONTROL_ISSUE = 0
RUNTIME_CONTROL_HOLD = 1
RUNTIME_CONTROL_IDLE = 2
PPO_CHECKPOINT_FORMAT = "AI42-ppo-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class AI42PPOConfig:
    gamma: float = math.exp(-0.2 / 1_200.0)
    gae_lambda: float = math.exp(-0.2 / 180.0) / math.exp(-0.2 / 1_200.0)
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_gradient_norm: float = 1.0
    update_epochs: int = 3
    minibatch_actors: int = 16
    target_kl: float = 0.02

    def __post_init__(self) -> None:
        probabilities = ("gamma", "gae_lambda")
        positives = (
            "clip_ratio", "value_clip", "value_loss_weight", "learning_rate",
            "max_gradient_norm", "target_kl",
        )
        for name in probabilities:
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in positives:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.entropy_weight < 0 or self.weight_decay < 0:
            raise ValueError("entropy_weight and weight_decay cannot be negative")
        if self.update_epochs < 1 or self.minibatch_actors < 1:
            raise ValueError("update_epochs and minibatch_actors must be positive")


class AI42ValueHead(nn.Module):
    """Training-only critic over the actor's recurrent belief state."""

    training_only = True

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        # A BC bootstrap has no calibrated value function.  Starting at zero
        # prevents arbitrary critic noise from becoming a fake policy signal
        # during the reward-free opening ticks before the first creep wave.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("critic hidden width does not match the actor")
        return self.network(hidden).squeeze(-1)


class AI42ActorCritic(nn.Module):
    """Joint learner module whose actor can still be exported independently."""

    def __init__(self, actor: AI42Actor, critic: AI42ValueHead | None = None) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic or AI42ValueHead(actor.hidden_size)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        output = self.actor(*args, **kwargs)
        # Keep the training-only critic independent.  Value regression must
        # not rewrite the actor representation; actor parameters are updated
        # only by the clipped policy objective and entropy regularization.
        return {**output, "value": self.critic(output["h"].detach())}


def _masked_categorical(logits: Tensor, mask: Tensor, name: str) -> torch.distributions.Categorical:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(f"{name} mask shape {tuple(mask.shape)} != logits {tuple(logits.shape)}")
    if bool((~mask.any(dim=-1)).any()):
        raise ValueError(f"{name} has a row without a valid action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def _runtime_control_logits(control_logits: Tensor) -> Tensor:
    return torch.stack(
        (
            control_logits[:, CONTROL_ISSUE],
            control_logits[:, CONTROL_HOLD],
            torch.logaddexp(
                control_logits[:, CONTROL_WAIT], control_logits[:, CONTROL_CANCEL],
            ),
        ),
        dim=-1,
    )


def _effective_target_mask(
    target_mask: Tensor,
    skill_target_mask: Tensor,
    entity_mask: Tensor,
    kinds: Tensor,
) -> Tensor:
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    skills = (kinds >= 3) & (kinds <= 6)
    selected_skill = skill_target_mask[rows, (kinds - 3).clamp(0, 3)]
    result = torch.where(skills.unsqueeze(1), selected_skill, target_mask)
    result = result & entity_mask
    empty = ~result.any(dim=1)
    if bool(empty.any()):
        result = result.clone()
        result[empty, 0] = True
    return result


def _parameter_activity(kinds: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    target = (kinds >= 2) & (kinds <= 6)
    skill = (kinds >= 3) & (kinds <= 6)
    offset = skill | ((kinds == 1) & (anchors == 0))
    anchor = kinds == 1
    return target, offset, anchor


def _action_distributions(
    output: Mapping[str, Tensor],
    kind_mask: Tensor,
    target_mask: Tensor,
    skill_target_mask: Tensor,
    entity_mask: Tensor,
    active_order: Tensor,
    *,
    controls: Tensor | None = None,
    actions: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample or replay the factorized runtime action.

    Returns ``controls, actions, log_prob, entropy``.  WAIT and CANCEL are
    intentionally marginalized into the single IDLE action seen by the game.
    """

    control_logits = _runtime_control_logits(output["control"])
    control_mask = torch.ones_like(control_logits, dtype=torch.bool)
    control_mask[:, RUNTIME_CONTROL_HOLD] = active_order.bool()
    control_dist = _masked_categorical(control_logits, control_mask, "control")
    if controls is None:
        controls = control_dist.sample()
    else:
        controls = controls.long()
    issue = controls == RUNTIME_CONTROL_ISSUE

    kind_dist = _masked_categorical(output["kind"], kind_mask.bool(), "kind")
    if actions is None:
        kinds = kind_dist.sample()
    else:
        kinds = actions[:, 0].long()

    effective_target = _effective_target_mask(
        target_mask.bool(), skill_target_mask.bool(), entity_mask.bool(), kinds,
    )
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    target_dist = _masked_categorical(
        output["target"][rows, kinds], effective_target, "target",
    )
    offset_dist = torch.distributions.Categorical(logits=output["offset"][rows, kinds])
    anchor_dist = torch.distributions.Categorical(logits=output["anchor"][rows, kinds])

    if actions is None:
        targets = target_dist.sample()
        offsets = offset_dist.sample()
        anchors = anchor_dist.sample()
    else:
        targets = actions[:, 1].long()
        offsets = actions[:, 2].long()
        anchors = actions[:, 3].long()

    target_active, offset_active, anchor_active = _parameter_activity(kinds, anchors)
    log_prob = control_dist.log_prob(controls)
    entropy = control_dist.entropy()
    issue_float = issue.float()
    parameter_log_prob = (
        kind_dist.log_prob(kinds)
        + target_dist.log_prob(targets) * target_active.float()
        + offset_dist.log_prob(offsets) * offset_active.float()
        + anchor_dist.log_prob(anchors) * anchor_active.float()
    )
    parameter_entropy = (
        kind_dist.entropy()
        + target_dist.entropy() * target_active.float()
        + offset_dist.entropy() * offset_active.float()
        + anchor_dist.entropy() * anchor_active.float()
    )
    log_prob = log_prob + parameter_log_prob * issue_float
    entropy = entropy + parameter_entropy * issue_float

    canonical = torch.stack((kinds, targets, offsets, anchors), dim=-1)
    skill = (kinds >= 3) & (kinds <= 6)
    canonical[:, 1] = torch.where(target_active, targets, 0)
    canonical[:, 2] = torch.where(offset_active, offsets, 0)
    canonical[:, 3] = torch.where(kinds == 1, anchors, 0)
    canonical = torch.where(issue.unsqueeze(1), canonical, torch.zeros_like(canonical))
    return controls, canonical, log_prob, entropy


@dataclass(slots=True)
class AI42Rollout:
    observations: dict[str, np.ndarray]
    controls: np.ndarray
    actions: np.ndarray
    active_order: np.ndarray
    reset_mask: np.ndarray
    policy_mask: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    old_log_prob: np.ndarray
    old_values: np.ndarray
    initial_h: np.ndarray
    initial_c: np.ndarray
    bootstrap_values: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.controls.shape[0])

    @property
    def actors(self) -> int:
        return int(self.controls.shape[1])


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    wire_actions: np.ndarray
    controls: np.ndarray
    actions: np.ndarray
    log_prob: np.ndarray
    values: np.ndarray
    reset_mask: np.ndarray
    policy_mask: np.ndarray
    active_order: np.ndarray
    input_h: np.ndarray
    input_c: np.ndarray


class AI42StochasticPolicy:
    """Own recurrent state while gathering on-policy AI-42 experience."""

    def __init__(
        self,
        model: AI42ActorCritic,
        actors: int,
        device: torch.device,
        *,
        controlled_mask: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.actors = int(actors)
        self.device = device
        self.h, self.c = model.actor.initial_state(actors, device)
        self.previous_dead = torch.zeros(actors, dtype=torch.bool, device=device)
        self.force_reset = torch.ones(actors, dtype=torch.bool, device=device)
        if controlled_mask is None:
            controlled_mask = np.ones(actors, dtype=np.bool_)
        controlled = np.asarray(controlled_mask, dtype=np.bool_)
        if controlled.shape != (actors,):
            raise ValueError(f"controlled_mask must have shape ({actors},)")
        self.controlled_mask = torch.as_tensor(controlled, device=device)

    def reset_workers(self, worker_indices: Sequence[int]) -> None:
        for worker in worker_indices:
            start = int(worker) * HERO_COUNT
            stop = start + HERO_COUNT
            self.h[start:stop] = 0
            self.c[start:stop] = 0
            self.previous_dead[start:stop] = False
            self.force_reset[start:stop] = True

    def _tensors(self, observations: Sequence[Any]) -> tuple[Any, dict[str, Tensor]]:
        batch = stack_observations(observations)
        values = {
            name: torch.as_tensor(
                np.ascontiguousarray(getattr(batch, name)), device=self.device,
                dtype=torch.bool if name.endswith("mask") else torch.float32,
            )
            for name in (
                "hero", "abilities", "entities", "global_state", "entity_mask",
                "kind_mask", "target_mask", "skill_target_mask",
            )
        }
        return batch, values

    @torch.no_grad()
    def sample(self, observations: Sequence[Any]) -> RolloutDecision:
        batch, values = self._tensors(observations)
        active_order = torch.as_tensor(
            np.concatenate([result.active_order for result in observations]),
            device=self.device, dtype=torch.bool,
        )
        dead = values["hero"][:, 9] >= 0.5
        reset = self.force_reset | dead | self.previous_dead
        self.h = torch.where(reset.unsqueeze(1), torch.zeros_like(self.h), self.h)
        self.c = torch.where(reset.unsqueeze(1), torch.zeros_like(self.c), self.c)
        input_h = self.h.clone()
        input_c = self.c.clone()
        output = self.model(
            values["hero"], values["abilities"], values["entities"],
            values["global_state"], values["entity_mask"], self.h, self.c,
        )
        controls, actions, log_prob, _ = _action_distributions(
            output, values["kind_mask"], values["target_mask"],
            values["skill_target_mask"], values["entity_mask"], active_order,
        )
        self.h = output["h"]
        self.c = output["c"]
        self.previous_dead = dead
        self.force_reset.zero_()
        action_np = actions.cpu().numpy().astype(np.int16, copy=False)
        control_np = controls.cpu().numpy().astype(np.int16, copy=False)
        controlled_np = self.controlled_mask.cpu().numpy()
        wire = np.concatenate((control_np[:, None], action_np), axis=1)
        wire = np.where(controlled_np[:, None], wire, np.zeros_like(wire))
        return RolloutDecision(
            wire_actions=wire.reshape(-1, HERO_COUNT, 5),
            controls=control_np,
            actions=action_np,
            log_prob=log_prob.cpu().numpy().astype(np.float32, copy=False),
            values=output["value"].cpu().numpy().astype(np.float32, copy=False),
            reset_mask=reset.cpu().numpy(),
            policy_mask=(self.controlled_mask & ~dead).cpu().numpy(),
            active_order=active_order.cpu().numpy(),
            input_h=input_h.cpu().numpy(),
            input_c=input_c.cpu().numpy(),
        )

    @torch.no_grad()
    def values(self, observations: Sequence[Any]) -> np.ndarray:
        _, values = self._tensors(observations)
        dead = values["hero"][:, 9] >= 0.5
        reset = self.force_reset | dead | self.previous_dead
        h = torch.where(reset.unsqueeze(1), torch.zeros_like(self.h), self.h)
        c = torch.where(reset.unsqueeze(1), torch.zeros_like(self.c), self.c)
        output = self.model(
            values["hero"], values["abilities"], values["entities"],
            values["global_state"], values["entity_mask"], h, c,
        )
        return output["value"].cpu().numpy().astype(np.float32, copy=False)


def collect_rollout(
    env: Any,
    policy: AI42StochasticPolicy,
    observations: list[Any],
    *,
    horizon: int,
    seed_rng: np.random.Generator,
    max_steps: int,
    controllers: Sequence[int],
    roster_factory: Any,
    opponent_runner: Any | None = None,
    opponent_slots: Sequence[tuple[int, int]] = (),
) -> tuple[AI42Rollout, list[Any], dict[str, int]]:
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    observation_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "hero", "abilities", "entities", "global_state", "entity_mask",
            "kind_mask", "target_mask", "skill_target_mask",
        )
    }
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "controls", "actions", "active_order", "reset_mask", "policy_mask",
            "rewards", "dones", "old_log_prob", "old_values",
        )
    }
    initial_h = initial_c = None
    matches = wins_1 = wins_2 = draws = invalid = 0
    for _ in range(horizon):
        batch = stack_observations(observations)
        for name in observation_rows:
            observation_rows[name].append(np.array(getattr(batch, name), copy=True))
        decision = policy.sample(observations)
        if initial_h is None:
            initial_h, initial_c = decision.input_h, decision.input_c
        wire_actions = decision.wire_actions.copy()
        if opponent_runner is not None:
            opponent_indices: list[int] = []
            for worker, side in opponent_slots:
                start = worker * HERO_COUNT + (0 if side == 1 else HERO_COUNT // 2)
                opponent_indices.extend(range(start, start + HERO_COUNT // 2))
            opponent_actions, opponent_controls, _ = opponent_runner.act(
                observations, np.asarray(opponent_indices, dtype=np.intp),
            )
            for local, (worker, side) in enumerate(opponent_slots):
                start = 0 if side == 1 else HERO_COUNT // 2
                source = slice(local * (HERO_COUNT // 2), (local + 1) * (HERO_COUNT // 2))
                wire_actions[worker, start:start + HERO_COUNT // 2, 0] = opponent_controls[source]
                wire_actions[worker, start:start + HERO_COUNT // 2, 1:] = opponent_actions[source]
        next_observations = env.step(wire_actions)
        rewards = np.concatenate([result.rewards for result in next_observations]).copy()
        dones = np.concatenate([
            np.full(HERO_COUNT, result.done, dtype=np.bool_)
            for result in next_observations
        ])
        fields["controls"].append(decision.controls)
        fields["actions"].append(decision.actions)
        fields["active_order"].append(decision.active_order)
        fields["reset_mask"].append(decision.reset_mask)
        fields["policy_mask"].append(decision.policy_mask)
        fields["rewards"].append(rewards)
        fields["dones"].append(dones)
        fields["old_log_prob"].append(decision.log_prob)
        fields["old_values"].append(decision.values)
        invalid += int(sum(int(result.invalid.sum()) for result in next_observations))

        done_indices = [index for index, result in enumerate(next_observations) if result.done]
        if done_indices:
            matches += len(done_indices)
            for index in done_indices:
                winner = next_observations[index].winner
                wins_1 += int(winner == 1)
                wins_2 += int(winner == 2)
                draws += int(winner == 0)
            seeds = [int(seed_rng.integers(1, 2**63 - 1)) for _ in done_indices]
            replacements = env.reset_indices(
                done_indices, seeds, max_steps, controllers=controllers,
                rosters=roster_factory(seed_rng, len(done_indices)),
            )
            policy.reset_workers(done_indices)
            if opponent_runner is not None:
                opponent_lookup = {
                    worker: local for local, (worker, _side) in enumerate(opponent_slots)
                }
                opponent_runner.reset_workers([
                    opponent_lookup[worker]
                    for worker in done_indices if worker in opponent_lookup
                ])
            for index, replacement in replacements.items():
                next_observations[index] = replacement
        observations = next_observations
    assert initial_h is not None and initial_c is not None
    rollout = AI42Rollout(
        observations={name: np.stack(values) for name, values in observation_rows.items()},
        controls=np.stack(fields["controls"]),
        actions=np.stack(fields["actions"]),
        active_order=np.stack(fields["active_order"]),
        reset_mask=np.stack(fields["reset_mask"]),
        policy_mask=np.stack(fields["policy_mask"]),
        rewards=np.stack(fields["rewards"]),
        dones=np.stack(fields["dones"]),
        old_log_prob=np.stack(fields["old_log_prob"]),
        old_values=np.stack(fields["old_values"]),
        initial_h=initial_h,
        initial_c=initial_c,
        bootstrap_values=policy.values(observations),
    )
    return rollout, observations, {
        "matches": matches, "side_1_wins": wins_1, "side_2_wins": wins_2,
        "draws": draws, "invalid_actions": invalid,
    }


def generalized_advantage_estimate(
    rollout: AI42Rollout, config: AI42PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rollout.rewards, dtype=np.float32)
    last = np.zeros(rollout.actors, dtype=np.float32)
    next_value = rollout.bootstrap_values
    for time in range(rollout.steps - 1, -1, -1):
        continuation = 1.0 - rollout.dones[time].astype(np.float32)
        delta = (
            rollout.rewards[time]
            + config.gamma * next_value * continuation
            - rollout.old_values[time]
        )
        last = delta + config.gamma * config.gae_lambda * continuation * last
        advantages[time] = last
        next_value = rollout.old_values[time]
    return advantages, advantages + rollout.old_values


def _sequence_forward(
    model: AI42ActorCritic,
    rollout: AI42Rollout,
    actor_indices: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    obs = {
        name: torch.as_tensor(values[:, actor_indices], device=device)
        for name, values in rollout.observations.items()
    }
    controls = torch.as_tensor(rollout.controls[:, actor_indices], device=device)
    actions = torch.as_tensor(rollout.actions[:, actor_indices], device=device)
    active_order = torch.as_tensor(rollout.active_order[:, actor_indices], device=device)
    reset_mask = torch.as_tensor(rollout.reset_mask[:, actor_indices], device=device)
    h = torch.as_tensor(rollout.initial_h[actor_indices], device=device)
    c = torch.as_tensor(rollout.initial_c[actor_indices], device=device)
    log_probs: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []
    for time in range(rollout.steps):
        reset = reset_mask[time].bool()
        h = torch.where(reset.unsqueeze(1), torch.zeros_like(h), h)
        c = torch.where(reset.unsqueeze(1), torch.zeros_like(c), c)
        output = model(
            obs["hero"][time].float(), obs["abilities"][time].float(),
            obs["entities"][time].float(), obs["global_state"][time].float(),
            obs["entity_mask"][time].bool(), h, c,
        )
        _, _, log_prob, entropy = _action_distributions(
            output, obs["kind_mask"][time].bool(), obs["target_mask"][time].bool(),
            obs["skill_target_mask"][time].bool(), obs["entity_mask"][time].bool(),
            active_order[time].bool(), controls=controls[time], actions=actions[time],
        )
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(output["value"])
        h, c = output["h"], output["c"]
    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def ppo_update(
    model: AI42ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: AI42Rollout,
    config: AI42PPOConfig,
    device: torch.device,
    *,
    mixed_precision: bool = True,
) -> dict[str, float | int | bool]:
    advantages, returns = generalized_advantage_estimate(rollout, config)
    cold_reward_free = (
        float(np.abs(rollout.rewards).max(initial=0.0)) < 1e-8
        and float(np.abs(rollout.old_values).max(initial=0.0)) < 1e-8
        and float(np.abs(rollout.bootstrap_values).max(initial=0.0)) < 1e-8
        and not bool(rollout.dones.any())
    )
    if cold_reward_free:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
            "epochs_completed": 0,
            "early_stopped": False,
            "cold_reward_free": True,
            "reward_mean": float(rollout.rewards.mean()),
            "reward_sum": float(rollout.rewards.sum()),
            "advantage_mean": 0.0,
        }
    policy_mask = rollout.policy_mask.astype(np.bool_)
    valid_advantages = advantages[policy_mask]
    if not valid_advantages.size:
        raise RuntimeError("rollout has no living policy samples")
    mean, std = float(valid_advantages.mean()), float(valid_advantages.std())
    normalized_advantages = (advantages - mean) / max(std, 1e-8)
    actor_order = np.arange(rollout.actors)
    metric_rows: list[tuple[float, ...]] = []
    early_stopped = False
    completed_epochs = 0
    for _ in range(config.update_epochs):
        np.random.shuffle(actor_order)
        for offset in range(0, rollout.actors, config.minibatch_actors):
            indices = actor_order[offset:offset + config.minibatch_actors]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                new_log_prob, entropy, values = _sequence_forward(
                    model, rollout, indices, device,
                )
            new_log_prob = new_log_prob.float()
            entropy = entropy.float()
            values = values.float()
            old_log_prob = torch.as_tensor(
                rollout.old_log_prob[:, indices], device=device,
            )
            old_values = torch.as_tensor(rollout.old_values[:, indices], device=device)
            adv = torch.as_tensor(normalized_advantages[:, indices], device=device)
            ret = torch.as_tensor(returns[:, indices], device=device)
            mask = torch.as_tensor(policy_mask[:, indices], device=device)

            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            unclipped = ratio * adv
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * adv
            policy_loss = -torch.minimum(unclipped, clipped)[mask].mean()
            clipped_value = old_values + (values - old_values).clamp(
                -config.value_clip, config.value_clip,
            )
            value_error = (values - ret).square()
            clipped_value_error = (clipped_value - ret).square()
            value_loss = 0.5 * torch.maximum(value_error, clipped_value_error)[mask].mean()
            entropy_mean = entropy[mask].mean()
            approximate_kl = ((ratio - 1.0) - log_ratio)[mask].mean()
            clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio)[mask].float().mean()
            loss = (
                policy_loss + config.value_loss_weight * value_loss
                - (0.0 if cold_reward_free else config.entropy_weight) * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("PPO gradient norm is non-finite")
            optimizer.step()
            metric_rows.append(tuple(float(value.detach().cpu()) for value in (
                loss, policy_loss, value_loss, entropy_mean, approximate_kl,
                clip_fraction, gradient_norm,
            )))
            if float(approximate_kl.detach()) > config.target_kl:
                early_stopped = True
                break
        completed_epochs += 1
        if early_stopped:
            break
    metrics = np.asarray(metric_rows, dtype=np.float64).mean(axis=0)
    return {
        "loss": float(metrics[0]),
        "policy_loss": float(metrics[1]),
        "value_loss": float(metrics[2]),
        "entropy": float(metrics[3]),
        "approximate_kl": float(metrics[4]),
        "clip_fraction": float(metrics[5]),
        "gradient_norm": float(metrics[6]),
        "epochs_completed": completed_epochs,
        "early_stopped": early_stopped,
        "cold_reward_free": cold_reward_free,
        "reward_mean": float(rollout.rewards.mean()),
        "reward_sum": float(rollout.rewards.sum()),
        "advantage_mean": float(valid_advantages.mean()),
    }


def save_ppo_checkpoint(
    path: str | os.PathLike[str],
    model: AI42ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: AI42PPOConfig,
    *,
    model_kwargs: Mapping[str, Any],
    update: int,
    hero_steps: int,
    source_lineage: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    payload = {
        "format": PPO_CHECKPOINT_FORMAT,
        "actor_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ppo_config": asdict(config),
        "model_kwargs": dict(model_kwargs),
        "update": int(update),
        "hero_steps": int(hero_steps),
        "source_lineage": dict(source_lineage),
        "metrics": dict(metrics),
        "rng_state": {
            "python": random.getstate(),
            # Keep the artifact compatible with torch.load(weights_only=True).
            "numpy": {
                "bit_generator": numpy_state[0],
                "state": torch.from_numpy(
                    np.asarray(numpy_state[1], dtype=np.uint32).copy(),
                ),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def load_ppo_checkpoint(
    path: str | os.PathLike[str],
    model: AI42ActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != PPO_CHECKPOINT_FORMAT:
        raise ValueError("not an AI-42 PPO checkpoint")
    model.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    model.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return dict(payload)


__all__ = [
    "AI42ActorCritic", "AI42PPOConfig", "AI42Rollout", "AI42StochasticPolicy",
    "AI42ValueHead", "PPO_CHECKPOINT_FORMAT", "collect_rollout",
    "generalized_advantage_estimate", "load_ppo_checkpoint", "ppo_update",
    "save_ppo_checkpoint",
]
