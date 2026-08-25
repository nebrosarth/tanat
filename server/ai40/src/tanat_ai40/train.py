from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .env import (
    AI40_ROSTER,
    AI40_SELF_PLAY_CONTROLLERS,
    AssaultVectorEnv,
    HeroAction,
    HERO_COUNT,
    REWARD_HASH,
    SCHEMA_HASH,
    self_play_rosters,
)
from .model import AI40Policy, masked_categorical


def as_tensor(array: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def select_parameter_logits(output, kinds):
    """Gather action-conditioned heads used by AI-41; AI-40 passes through."""
    if output["target"].ndim == 2:
        return output
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    selected = dict(output)
    for name in ("target", "direction", "distance"):
        selected[name] = output[name][rows, kinds]
    return selected


def distributions(output, kind_mask, target_mask, kinds=None):
    if output["target"].ndim == 3:
        if kinds is None:
            raise ValueError("action kinds are required for conditioned policy heads")
        output = select_parameter_logits(output, kinds)
    return (
        masked_categorical(output["kind"], kind_mask),
        masked_categorical(output["target"], target_mask),
        torch.distributions.Categorical(logits=output["direction"]),
        torch.distributions.Categorical(logits=output["distance"]),
    )


def policy_forward(policy, hero, entities, global_state, entity_mask, h, c, abilities=None):
    wrapped = getattr(policy, "policy", policy)
    if getattr(wrapped, "uses_abilities", False):
        if abilities is None:
            raise ValueError("AI-41 policy requires ability observations")
        return policy(hero, abilities, entities, global_state, entity_mask, h, c)
    return policy(hero, entities, global_state, entity_mask, h, c)


def combined_log_prob(dists, actions):
    kinds, targets, directions, distances = actions
    target_active = ((kinds >= 2) & (kinds <= 6)).float()
    position_active = ((kinds == 1) | ((kinds >= 3) & (kinds <= 6))).float()
    distance_active = position_active
    if dists[2].logits.shape[-1] == 81:
        position_active = (((kinds == 1) & (distances == 0)) |
                           ((kinds >= 3) & (kinds <= 6))).float()
        distance_active = (kinds == 1).float()
    return (dists[0].log_prob(kinds) + dists[1].log_prob(targets) * target_active +
            dists[2].log_prob(directions) * position_active +
            dists[3].log_prob(distances) * distance_active)


def combined_entropy(dists, kinds, distances=None):
    target_active = ((kinds >= 2) & (kinds <= 6)).float()
    position_active = ((kinds == 1) | ((kinds >= 3) & (kinds <= 6))).float()
    distance_active = position_active
    if dists[2].logits.shape[-1] == 81:
        if distances is None:
            raise ValueError("navigation entropy requires sampled anchor actions")
        position_active = (((kinds == 1) & (distances == 0)) |
                           ((kinds >= 3) & (kinds <= 6))).float()
        distance_active = (kinds == 1).float()
    return (dists[0].entropy() + dists[1].entropy() * target_active +
            dists[2].entropy() * position_active + dists[3].entropy() * distance_active)


def effective_target_mask(obs, kinds: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = as_tensor(obs.target_mask, device).bool().clone()
    skill_masks = as_tensor(obs.skill_target_mask, device).bool()
    skill_rows = (kinds >= 3) & (kinds <= 6)
    rows = torch.arange(kinds.shape[0], device=device)
    selected = skill_masks[rows, torch.clamp(kinds - 3, 0, 3)]
    mask = torch.where(skill_rows.unsqueeze(1), selected, mask)
    # Target is ignored by wait/move/self/point actions, but the factorized
    # distribution still needs one finite category. Keep this entirely on the
    # device; kinds.tolist() used to force an extra CUDA synchronization/tick.
    empty = ~mask.any(dim=1)
    mask[empty, 0] = True
    return mask


def teacher_behavior_loss(output, actions: torch.Tensor) -> torch.Tensor:
    """Balanced supervised loss for live AI-30 teacher labels.

    PPO trajectories are dominated by navigation.  Keeping a small, balanced
    expert term prevents a temporary reward advantage for "walk to an anchor"
    from erasing attacks and abilities learned during the bootstrap.
    """
    kinds = actions[:, 0].long()
    targets = actions[:, 1].long()
    directions = actions[:, 2].long()
    distances = actions[:, 3].long()
    selected = select_parameter_logits(output, kinds)
    weights = torch.tensor(
        (0.0, 0.15, 1.0, 32.0, 32.0, 32.0, 32.0, 0.0),
        device=kinds.device,
    )
    loss = nn.functional.cross_entropy(selected["kind"], kinds, weight=weights)
    attack = kinds == 2
    if attack.any():
        loss = loss + nn.functional.cross_entropy(selected["target"][attack], targets[attack])
    local_move = (kinds == 1) & (distances == 0)
    if local_move.any():
        loss = loss + nn.functional.cross_entropy(selected["direction"][local_move], directions[local_move])
    anchor_move = (kinds == 1) & (distances > 0)
    if anchor_move.any():
        loss = loss + nn.functional.cross_entropy(selected["distance"][anchor_move], distances[anchor_move])
    skill = (kinds >= 3) & (kinds <= 6)
    if skill.any():
        loss = loss + 2.0 * nn.functional.cross_entropy(selected["target"][skill], targets[skill])
        loss = loss + 2.0 * nn.functional.cross_entropy(selected["direction"][skill], directions[skill])
    return loss


def stack_observations(observations):
    class Batch:
        pass
    batch = Batch()
    batched = getattr(observations, "batched_observations", None)
    if batched is not None:
        for name, values in batched.items():
            setattr(batch, name, values)
        return batch
    for name in ("hero", "entities", "global_state", "entity_mask", "kind_mask", "target_mask", "skill_target_mask"):
        setattr(batch, name, np.concatenate([getattr(obs, name) for obs in observations], axis=0))
    if observations and observations[0].abilities is not None:
        batch.abilities = np.concatenate([obs.abilities for obs in observations], axis=0)
    # A partial vector reset invalidates the zero-copy combined response view.
    # Preserve V12 teacher labels on that slower path too; otherwise a rollout
    # changes shape after its first completed match and the next PPO update
    # receives a mixture of labelled and unlabelled rows.
    if observations and observations[0].teacher_actions is not None:
        batch.teacher_actions = np.concatenate(
            [obs.teacher_actions for obs in observations], axis=0,
        )
        batch.teacher_valid = np.concatenate(
            [obs.teacher_valid for obs in observations], axis=0,
        )
    return batch


def collect_rollout(
    env, policy, observations, h, c, horizon, device, seed_rng, max_steps,
    controllers=AI40_SELF_PLAY_CONTROLLERS,
):
    rows = []
    for _ in range(horizon):
        obs = stack_observations(observations)
        hero = as_tensor(obs.hero, device)
        entities = as_tensor(obs.entities, device)
        global_state = as_tensor(obs.global_state, device)
        entity_mask = as_tensor(obs.entity_mask, device)
        kind_mask = as_tensor(obs.kind_mask, device).bool()
        h_in, c_in = h.detach(), c.detach()
        with torch.no_grad():
            abilities = (as_tensor(obs.abilities, device) if hasattr(obs, "abilities") else None)
            output = policy_forward(
                policy, hero, entities, global_state, entity_mask, h_in, c_in, abilities,
            )
            kind_dist = masked_categorical(output["kind"], kind_mask)
            kinds = kind_dist.sample()
            target_mask = effective_target_mask(obs, kinds, device)
            dists = distributions(output, kind_mask, target_mask, kinds)
            targets = dists[1].sample()
            directions = dists[2].sample()
            distances = dists[3].sample()
            action_tensors = (kinds, targets, directions, distances)
            log_prob = combined_log_prob(dists, action_tensors)
        # One device synchronization for the complete action batch. Calling
        # int(cuda_tensor[i]) for every factor serialized hundreds of tiny
        # GPU-to-CPU reads per environment tick.
        action_values = torch.stack(action_tensors, dim=-1).cpu().numpy()
        actions = [HeroAction(*(int(value) for value in row)) for row in action_values]
        worker_actions = [actions[i:i + HERO_COUNT] for i in range(0, len(actions), HERO_COUNT)]
        next_observations = env.step(worker_actions)
        rewards = np.concatenate([result.rewards for result in next_observations])
        done = np.concatenate([np.full(HERO_COUNT, result.done, np.float32) for result in next_observations])
        rows.append({
            "hero": obs.hero.copy(), "entities": obs.entities.copy(),
            "global": obs.global_state.copy(), "entity_mask": obs.entity_mask.copy(),
            "kind_mask": obs.kind_mask.copy(), "target_mask": target_mask.cpu().numpy(),
            "h": h_in.cpu().numpy(), "c": c_in.cpu().numpy(),
            "actions": np.stack([x.cpu().numpy() for x in action_tensors], axis=-1),
            "log_prob": log_prob.cpu().numpy(), "value": output["value"].cpu().numpy(),
            "reward": rewards, "done": done,
        })
        if hasattr(obs, "abilities"):
            rows[-1]["abilities"] = obs.abilities.copy()
        h, c = output["h"].detach(), output["c"].detach()
        observations = next_observations
        done_indices = [i for i, result in enumerate(observations) if result.done]
        if done_indices:
            seeds = [int(seed_rng.integers(1, 2**63 - 1)) for _ in done_indices]
            replacements = env.reset_indices(
                done_indices,
                seeds,
                max_steps,
                controllers=controllers,
                rosters=self_play_rosters(seed_rng, len(done_indices)),
            )
            for index, replacement in replacements.items():
                observations[index] = replacement
                start, end = index * HERO_COUNT, (index + 1) * HERO_COUNT
                h[start:end] = 0
                c[start:end] = 0
    obs = stack_observations(observations)
    with torch.no_grad():
        bootstrap = policy_forward(
            policy, as_tensor(obs.hero, device), as_tensor(obs.entities, device),
            as_tensor(obs.global_state, device), as_tensor(obs.entity_mask, device), h, c,
            as_tensor(obs.abilities, device) if hasattr(obs, "abilities") else None,
        )["value"].cpu().numpy()
    return rows, observations, h, c, bootstrap


def horizon_discounts(
    tick_seconds: float = 0.2,
    discount_horizon_seconds: float = 1_200.0,
    gae_horizon_seconds: float = 180.0,
) -> tuple[float, float]:
    """Convert readable e-folding horizons to per-tick gamma and GAE lambda."""
    if tick_seconds <= 0 or gae_horizon_seconds <= 0 or discount_horizon_seconds <= 0:
        raise ValueError("tick and horizon durations must be positive")
    gamma = math.exp(-tick_seconds / discount_horizon_seconds)
    trace_discount = math.exp(-tick_seconds / gae_horizon_seconds)
    lam = trace_discount / gamma
    if not 0 < lam <= 1:
        raise ValueError("GAE horizon must not exceed the discount horizon")
    return gamma, lam


def gae(rows, bootstrap, gamma=0.99, lam=0.95):
    actors = bootstrap.shape[0]
    advantages = np.zeros((len(rows), actors), np.float32)
    last = np.zeros(actors, np.float32)
    next_value = bootstrap
    for t in reversed(range(len(rows))):
        alive = 1.0 - rows[t]["done"]
        delta = rows[t]["reward"] + gamma * next_value * alive - rows[t]["value"]
        last = delta + gamma * lam * alive * last
        advantages[t] = last
        next_value = rows[t]["value"]
    returns = advantages + np.stack([row["value"] for row in rows])
    return advantages, returns


def flatten(rows, key):
    return np.concatenate([row[key] for row in rows], axis=0)


def ppo_update(
    policy, optimizer, rows, advantages, returns, device, epochs=3,
    minibatch_size=512, return_metrics=False, mixed_precision=False,
    forward_policy=None, target_kl: float | None = None,
    teacher_loss_weight: float = 0.0,
):
    policy_mask = (flatten(rows, "policy_mask").astype(bool) if "policy_mask" in rows[0]
                   else np.ones(advantages.size, dtype=bool))
    if not policy_mask.any():
        raise RuntimeError("rollout contains no AI-40-controlled samples")
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if target_kl is not None and target_kl <= 0:
        raise ValueError("target_kl must be positive when enabled")
    if teacher_loss_weight < 0:
        raise ValueError("teacher_loss_weight cannot be negative")
    # Keep the rollout in host memory. Moving a complete multi-worker rollout
    # to CUDA makes attention activations scale with every hero-step at once;
    # a 12x256 rollout exhausted a 32 GiB GPU and committed over 45 GiB RAM.
    host = {key: flatten(rows, key) for key in
            ("hero", "entities", "global", "entity_mask", "kind_mask",
             "target_mask", "h", "c", "actions", "log_prob")}
    if "abilities" in rows[0]:
        host["abilities"] = flatten(rows, "abilities")
    teacher_indices = np.empty(0, dtype=np.int64)
    teacher_actions = None
    if (teacher_loss_weight > 0 and
            all("teacher_valid" in row and "teacher_actions" in row for row in rows)):
        teacher_indices = np.flatnonzero(flatten(rows, "teacher_valid").astype(bool))
        if teacher_indices.size:
            teacher_actions = flatten(rows, "teacher_actions").astype(np.int64, copy=False)
    selected = np.flatnonzero(policy_mask)
    advantage_host = advantages.reshape(-1)[selected]
    advantage_host = ((advantage_host - advantage_host.mean()) /
                      (advantage_host.std() + 1e-8)).astype(np.float32)
    return_host = returns.reshape(-1)[selected].astype(np.float32)
    old_value_host = flatten(rows, "value")[selected].astype(np.float32)
    return_variance = float(np.var(return_host))
    explained_variance = (1.0 - float(np.var(return_host - old_value_host)) / return_variance
                          if return_variance > 1e-12 else 0.0)
    forward_policy = forward_policy or policy
    metric_rows: list[torch.Tensor] = []
    completed_epochs = 0
    early_stopped = False
    for _ in range(epochs):
        order = np.random.permutation(len(selected))
        for offset in range(0, len(order), minibatch_size):
            local = order[offset:offset + minibatch_size]
            indices = selected[local]
            data = {
                key: as_tensor(values[indices], device)
                for key, values in host.items()
            }
            adv = as_tensor(advantage_host[local], device)
            ret = as_tensor(return_host[local], device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                raw_output = policy_forward(
                    forward_policy, data["hero"], data["entities"], data["global"],
                    data["entity_mask"], data["h"], data["c"], data.get("abilities"),
                )
            output = {
                key: value.float() if key in (
                    "kind", "target", "direction", "distance", "value",
                ) else value
                for key, value in raw_output.items()
            }
            teacher_loss = torch.zeros((), device=device)
            if teacher_actions is not None and teacher_indices.size:
                count = min(len(local), teacher_indices.size)
                teacher_local = teacher_indices[np.random.randint(teacher_indices.size, size=count)]
                teacher_data = {
                    key: as_tensor(values[teacher_local], device)
                    for key, values in host.items()
                }
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=mixed_precision and device.type == "cuda",
                ):
                    raw_teacher = policy_forward(
                        forward_policy, teacher_data["hero"], teacher_data["entities"],
                        teacher_data["global"], teacher_data["entity_mask"],
                        teacher_data["h"], teacher_data["c"], teacher_data.get("abilities"),
                    )
                teacher_output = {
                    key: value.float() if key in ("kind", "target", "direction", "distance", "value") else value
                    for key, value in raw_teacher.items()
                }
                teacher_loss = teacher_behavior_loss(
                    teacher_output, as_tensor(teacher_actions[teacher_local], device, dtype=torch.long),
                )
            actions = tuple(data["actions"][:, i].long() for i in range(4))
            dists = distributions(
                output, data["kind_mask"].bool(), data["target_mask"].bool(), actions[0],
            )
            log_prob = combined_log_prob(dists, actions)
            ratio = (log_prob - data["log_prob"]).exp()
            clipped = torch.clamp(ratio, 0.8, 1.2) * adv
            policy_loss = -torch.minimum(ratio * adv, clipped).mean()
            value_loss = nn.functional.mse_loss(output["value"], ret)
            entropy = combined_entropy(dists, actions[0], actions[3]).mean()
            log_ratio = log_prob - data["log_prob"]
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > 0.2).float().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + teacher_loss_weight * teacher_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            metric_rows.append(torch.stack((
                loss.detach(), policy_loss.detach(), value_loss.detach(), entropy.detach(),
                approximate_kl.detach(), clip_fraction.detach(), gradient_norm.detach(), teacher_loss.detach(),
            )))
            # One host read per learner minibatch is intentional here.  It is
            # much cheaper than completing an unstable PPO epoch, and protects
            # a rollout from being pushed far outside its behaviour policy.
            if target_kl is not None and float(approximate_kl.detach()) > target_kl:
                early_stopped = True
            del data, raw_output, output, dists, loss
            if early_stopped:
                break
        completed_epochs += 1
        if early_stopped:
            break
    # Avoid one blocking GPU-to-CPU synchronization per minibatch. Metrics are
    # tiny and remain on-device until the complete PPO update has finished.
    means = torch.stack(metric_rows).mean(dim=0).float().cpu().numpy()
    metrics = {
        "loss": float(means[0]), "policy_loss": float(means[1]),
        "value_loss": float(means[2]), "entropy": float(means[3]),
        "approximate_kl": float(means[4]), "clip_fraction": float(means[5]),
        "gradient_norm": float(means[6]), "explained_variance": explained_variance,
        "ppo_epochs_completed": completed_epochs,
        "ppo_early_stopped": early_stopped, "teacher_loss": float(means[7]),
    }
    return metrics if return_metrics else metrics["loss"]


def save_checkpoint(path: Path, policy, optimizer, update: int, hero_steps: int, config: dict,
                    schema_hash: bytes = SCHEMA_HASH, reward_hash: bytes = REWARD_HASH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "schema_hash": schema_hash.hex(), "reward_hash": reward_hash.hex(),
                "update": update, "hero_steps": hero_steps, "config": config}, path)


def load_checkpoint(path: Path, policy, optimizer, schema_hash: bytes = SCHEMA_HASH,
                    reward_hash: bytes = REWARD_HASH) -> tuple[int, int]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != schema_hash.hex() or saved.get("reward_hash") != reward_hash.hex():
        raise RuntimeError("checkpoint schema/reward hash does not match the current environment")
    policy.load_state_dict(saved["model"])
    if "optimizer" in saved:
        optimizer.load_state_dict(saved["optimizer"])
    return int(saved.get("update", 0)), int(saved.get("hero_steps", 0))


def train(executable: Path, steps: int, workers: int, updates: int, device: torch.device,
          output: Path, resume: Path | None = None, minibatch_size: int = 512) -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    policy = AI40Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    start_update, hero_steps = (0, 0)
    if resume is not None:
        start_update, hero_steps = load_checkpoint(resume, policy, optimizer)
    actors = workers * HERO_COUNT
    h, c = policy.initial_state(actors, device)
    seed_rng = np.random.default_rng(1)
    config = {
        "workers": workers,
        "rollout_steps": steps,
        "device": str(device),
        "training_mode": "ai40_mirror_self_play",
        "controllers": list(AI40_SELF_PLAY_CONTROLLERS),
        "minibatch_size": minibatch_size,
    }
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with AssaultVectorEnv(executable, workers) as env:
        observations = env.reset(
            range(1, workers + 1),
            max_steps=max(steps * 4, 100),
            controllers=AI40_SELF_PLAY_CONTROLLERS,
            rosters=self_play_rosters(seed_rng, workers),
        )
        for update in range(start_update + 1, start_update + updates + 1):
            rows, observations, h, c, bootstrap = collect_rollout(
                env, policy, observations, h, c, steps, device, seed_rng,
                max(steps * 4, 100), AI40_SELF_PLAY_CONTROLLERS,
            )
            advantages, returns = gae(rows, bootstrap)
            loss = ppo_update(
                policy, optimizer, rows, advantages, returns, device,
                minibatch_size=minibatch_size,
            )
            hero_steps += steps * actors
            save_checkpoint(output / "latest.pt", policy, optimizer, update, hero_steps, config)
            print(f"update={update} hero_steps={hero_steps} loss={loss:.6f}")
    checkpoint = output / "latest.pt"
    manifest = {"version": "AI-40-v3", "schema_hash": SCHEMA_HASH.hex(), "reward_hash": REWARD_HASH.hex(),
                "hidden_size": policy.hidden_size, "roster_ids": AI40_ROSTER.tolist(),
                "hero_steps": hero_steps, "updates": start_update + updates, **config}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"training complete: updates={updates} hero_steps={hero_steps} env_steps/s={steps*workers*updates/elapsed:.1f} checkpoint={checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True, help="path to assaultenv executable")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/smoke"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--minibatch-size", type=int, default=512)
    args = parser.parse_args()
    train(
        args.env, args.steps, args.workers, args.updates, torch.device(args.device),
        args.output, args.resume, args.minibatch_size,
    )


if __name__ == "__main__":
    main()
