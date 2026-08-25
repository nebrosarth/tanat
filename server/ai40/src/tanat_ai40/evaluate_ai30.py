from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .env import (
    AssaultVectorEnv,
    CONTROLLER_AI30,
    CONTROLLER_AI40,
    HERO_COUNT,
    REWARD_HASH,
    SCHEMA_HASH,
    PROTOCOL_VERSION,
    self_play_rosters,
)
from .model import AI40Policy, PolicyRunner, masked_categorical
from .train import effective_target_mask, policy_forward, select_parameter_logits, stack_observations
from .train_async import _effective_target_mask_tensors


ACTION_NAMES = ("wait", "move", "attack", "skill1", "skill2", "skill3", "skill4", "teleport")


def controlled_slot_indices(assignments: list[int], candidate: bool = True) -> np.ndarray:
    """Return the five hero rows controlled by one side in every vector slot.

    Evaluation alternates the candidate between teams.  There is no reason to
    run its recurrent policy for the five heroes controlled by AI-30 (or by the
    frozen opponent): the authoritative server ignores those actions.
    """
    workers = len(assignments)
    bases = np.arange(workers, dtype=np.intp) * HERO_COUNT
    sides = np.asarray(assignments, dtype=np.int8)
    # Finished slots remain in the fixed batch; their rows are ignored.  Give
    # them a stable, valid layout so CUDA graph shapes never change.
    sides = np.where(sides == 0, 1, sides)
    candidate_starts = bases + np.where(sides == 1, 0, HERO_COUNT // 2)
    starts = candidate_starts if candidate else bases + (HERO_COUNT // 2) - (candidate_starts - bases)
    return (starts[:, None] + np.arange(HERO_COUNT // 2, dtype=np.intp)).reshape(-1)


class EvaluationActor:
    """Persistent, deterministic GPU actor for the fixed-size evaluation batch."""

    def __init__(self, policy, batch_size: int, device: torch.device):
        self.policy = policy
        self.device = device
        self.batch_size = batch_size
        self.forward = PolicyRunner(
            policy,
            compile_model=device.type == "cuda",
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
        )
        self.inputs: dict[str, torch.Tensor] = {}
        self.actions_device = torch.empty((batch_size, 4), dtype=torch.int16, device=device)
        self.actions_host = torch.empty(
            (batch_size, 4), dtype=torch.int16,
            device="cpu",
            pin_memory=device.type == "cuda",
        )

    def initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.initial_state(self.batch_size, self.device)

    def _input(self, name: str, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        source = torch.from_numpy(np.ascontiguousarray(array))
        requested = dtype or source.dtype
        target = self.inputs.get(name)
        if target is None or target.shape != source.shape or target.dtype != requested:
            target = torch.empty(source.shape, dtype=requested, device=self.device)
            self.inputs[name] = target
        target.copy_(source, non_blocking=False)
        return target

    def act(self, observations, indices: np.ndarray, h: torch.Tensor, c: torch.Tensor):
        batch = stack_observations(observations)
        hero = self._input("hero", batch.hero[indices])
        entities = self._input("entities", batch.entities[indices])
        global_state = self._input("global_state", batch.global_state[indices])
        entity_mask = self._input("entity_mask", batch.entity_mask[indices], torch.bool)
        kind_mask = self._input("kind_mask", batch.kind_mask[indices], torch.bool)
        target_mask = self._input("target_mask", batch.target_mask[indices], torch.bool)
        skill_target_mask = self._input(
            "skill_target_mask", batch.skill_target_mask[indices], torch.bool,
        )
        abilities = self._input("abilities", batch.abilities[indices]) if hasattr(batch, "abilities") else None
        # BF16 follows the rollout actor.  Logits are widened for masking and
        # argmax so selection has the same numerical path as the learner.
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            out = policy_forward(self.forward, hero, entities, global_state, entity_mask, h, c, abilities)
            next_h, next_c = out["h"].detach().clone(), out["c"].detach().clone()
            kinds = out["kind"].float().masked_fill(~kind_mask, -1e9).argmax(dim=-1)
            selected = select_parameter_logits(out, kinds)
            effective_targets = _effective_target_mask_tensors(target_mask, skill_target_mask, kinds)
            targets = selected["target"].float().masked_fill(~effective_targets, -1e9).argmax(dim=-1)
            directions = selected["direction"].float().argmax(dim=-1)
            distances = selected["distance"].float().argmax(dim=-1)
            self.actions_device[:, 0].copy_(kinds)
            self.actions_device[:, 1].copy_(targets)
            self.actions_device[:, 2].copy_(directions)
            self.actions_device[:, 3].copy_(distances)
        # One packed transfer replaces four tensor conversions and a NumPy
        # allocation every environment tick.
        self.actions_host.copy_(self.actions_device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return self.actions_host.numpy(), next_h, next_c


def greedy_actions(policy, observations, h, c, device: torch.device):
    batch = stack_observations(observations)
    with torch.no_grad():
        out = policy_forward(
            policy, torch.as_tensor(batch.hero, device=device),
            torch.as_tensor(batch.entities, device=device),
            torch.as_tensor(batch.global_state, device=device),
            torch.as_tensor(batch.entity_mask, device=device), h, c,
            torch.as_tensor(batch.abilities, device=device)
            if hasattr(batch, "abilities") else None,
        )
        kind_mask = torch.as_tensor(batch.kind_mask, device=device).bool()
        kinds = masked_categorical(out["kind"], kind_mask).probs.argmax(-1)
        target_mask = effective_target_mask(batch, kinds, device)
        selected = select_parameter_logits(out, kinds)
        targets = selected["target"].masked_fill(~target_mask, -1e9).argmax(-1)
        directions = selected["direction"].argmax(-1)
        distances = selected["distance"].argmax(-1)
    values = torch.stack((kinds, targets, directions, distances), dim=-1).cpu().numpy()
    return values, out["h"].detach(), out["c"].detach()


def controllers_for_side(ai40_side: int) -> tuple[int, ...]:
    if ai40_side not in (1, 2):
        raise ValueError("AI-40 side must be 1 or 2")
    first = CONTROLLER_AI40 if ai40_side == 1 else CONTROLLER_AI30
    second = CONTROLLER_AI30 if ai40_side == 1 else CONTROLLER_AI40
    return (first,) * (HERO_COUNT // 2) + (second,) * (HERO_COUNT // 2)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_vs_ai30(
    checkpoint: Path,
    executable: Path,
    matches: int,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int = 10_000,
    policy_factory=AI40Policy,
    schema_hash: bytes = SCHEMA_HASH,
    reward_hash: bytes = REWARD_HASH,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict:
    if matches < 1 or workers < 1:
        raise ValueError("matches and workers must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != schema_hash.hex() or saved.get("reward_hash") != reward_hash.hex():
        raise RuntimeError("checkpoint schema/reward hash mismatch")
    policy = policy_factory().to(device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    workers = min(workers, matches)
    roster_rng = np.random.default_rng(seed)
    next_match = workers
    assignments = [1 + index % 2 for index in range(workers)]
    outcomes = Counter()
    side_outcomes = {1: Counter(), 2: Counter()}
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    invalid = total_steps = 0
    ai40_reward = 0.0
    started = time.perf_counter()
    with AssaultVectorEnv(executable, workers, protocol_version) as env:
        observations = env.reset(
            range(seed, seed + workers), max_steps,
            controller_sets=[controllers_for_side(side) for side in assignments],
            rosters=self_play_rosters(roster_rng, workers),
        )
        next_seed = seed + workers
        actor = EvaluationActor(policy, workers * (HERO_COUNT // 2), device)
        h, c = actor.initial_state()
        completed = 0
        while completed < matches:
            actor_indices = controlled_slot_indices(assignments)
            candidate_actions, h, c = actor.act(observations, actor_indices, h, c)
            action_values = np.zeros((workers, HERO_COUNT, 4), dtype=np.int16)
            for index, side in enumerate(assignments):
                if side == 0:
                    continue
                local_start = 0 if side == 1 else HERO_COUNT // 2
                source = slice(index * (HERO_COUNT // 2), (index + 1) * (HERO_COUNT // 2))
                action_values[index, local_start:local_start + HERO_COUNT // 2] = candidate_actions[source]
            results = env.step(action_values)
            reset_indices: list[int] = []
            reset_seeds: list[int] = []
            reset_controllers: list[tuple[int, ...]] = []
            for index, result in enumerate(results):
                side = assignments[index]
                if side == 0:
                    continue
                source = slice(index * (HERO_COUNT // 2), (index + 1) * (HERO_COUNT // 2))
                actions += np.bincount(candidate_actions[source, 0], minlength=len(ACTION_NAMES))
                local = slice(0, HERO_COUNT // 2) if side == 1 else slice(HERO_COUNT // 2, HERO_COUNT)
                ai40_reward += float(result.rewards[local].sum())
                invalid += int(result.invalid[local].sum())
                if not result.done:
                    continue
                completed += 1
                total_steps += result.step
                outcome = "draw" if result.winner == 0 else ("win" if result.winner == side else "loss")
                outcomes[outcome] += 1
                side_outcomes[side][outcome] += 1
                if next_match < matches:
                    replacement_side = 1 + next_match % 2
                    next_match += 1
                    assignments[index] = replacement_side
                    reset_indices.append(index)
                    reset_seeds.append(next_seed)
                    next_seed += 1
                    reset_controllers.append(controllers_for_side(replacement_side))
                else:
                    assignments[index] = 0
                hero_start = index * (HERO_COUNT // 2)
                hero_stop = hero_start + HERO_COUNT // 2
                h[hero_start:hero_stop] = 0
                c[hero_start:hero_stop] = 0
            observations = results
            if reset_indices:
                replacements = env.reset_indices(
                    reset_indices, reset_seeds, max_steps,
                    controller_sets=reset_controllers,
                    rosters=self_play_rosters(roster_rng, len(reset_indices)),
                )
                for index, replacement in replacements.items():
                    observations[index] = replacement
    elapsed = time.perf_counter() - started
    wins, losses, draws = outcomes["win"], outcomes["loss"], outcomes["draw"]
    lower, upper = wilson_interval(wins, matches)
    action_total = max(int(actions.sum()), 1)
    metrics = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_update": int(saved.get("update", 0)),
        "checkpoint_hero_steps": int(saved.get("hero_steps", 0)),
        "seed": seed, "matches": matches, "workers": workers,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / matches,
        "score_rate": (wins + 0.5 * draws) / matches,
        "draw_rate": draws / matches,
        "win_rate_ci95_low": lower, "win_rate_ci95_high": upper,
        "side_1": dict(side_outcomes[1]), "side_2": dict(side_outcomes[2]),
        "invalid_actions": invalid,
        "invalid_action_rate": invalid / action_total,
        "mean_ai40_team_reward": ai40_reward / matches,
        "mean_match_steps": total_steps / matches,
        "mean_match_minutes": total_steps * 0.2 / 60.0 / matches,
        "action_counts": {name: int(actions[index]) for index, name in enumerate(ACTION_NAMES)},
        "action_rates": {name: float(actions[index] / action_total) for index, name in enumerate(ACTION_NAMES)},
        "elapsed_seconds": elapsed,
        "matches_per_second": matches / max(elapsed, 1e-9),
    }
    return metrics


def evaluate_vs_checkpoint(
    checkpoint: Path,
    opponent_checkpoint: Path,
    executable: Path,
    matches: int,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int = 20_000,
    policy_factory=AI40Policy,
    schema_hash: bytes = SCHEMA_HASH,
    reward_hash: bytes = REWARD_HASH,
    protocol_version: int = PROTOCOL_VERSION,
    opponent_label: str | None = None,
) -> dict:
    """Deterministically evaluate a candidate against a frozen neural policy."""
    if matches < 1 or workers < 1:
        raise ValueError("matches and workers must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    opponent_saved = torch.load(opponent_checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != schema_hash.hex() or saved.get("reward_hash") != reward_hash.hex():
        raise RuntimeError("candidate checkpoint schema/reward hash mismatch")
    if opponent_saved.get("schema_hash") != schema_hash.hex():
        raise RuntimeError("opponent checkpoint schema hash mismatch")
    candidate, opponent = policy_factory().to(device), policy_factory().to(device)
    candidate.load_state_dict(saved["model"])
    opponent.load_state_dict(opponent_saved["model"])
    candidate.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    workers = min(workers, matches)
    roster_rng = np.random.default_rng(seed)
    next_match = workers
    assignments = [1 + index % 2 for index in range(workers)]
    outcomes = Counter()
    side_outcomes = {1: Counter(), 2: Counter()}
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    invalid = total_steps = 0
    candidate_reward = 0.0
    controllers = (CONTROLLER_AI40,) * HERO_COUNT
    started = time.perf_counter()
    with AssaultVectorEnv(executable, workers, protocol_version) as env:
        observations = env.reset(
            range(seed, seed + workers), max_steps,
            controller_sets=[controllers] * workers,
            rosters=self_play_rosters(roster_rng, workers),
        )
        next_seed = seed + workers
        candidate_actor = EvaluationActor(candidate, workers * (HERO_COUNT // 2), device)
        opponent_actor = EvaluationActor(opponent, workers * (HERO_COUNT // 2), device)
        candidate_h, candidate_c = candidate_actor.initial_state()
        opponent_h, opponent_c = opponent_actor.initial_state()
        completed = 0
        while completed < matches:
            candidate_indices = controlled_slot_indices(assignments, candidate=True)
            opponent_indices = controlled_slot_indices(assignments, candidate=False)
            candidate_actions, candidate_h, candidate_c = candidate_actor.act(
                observations, candidate_indices, candidate_h, candidate_c,
            )
            opponent_actions, opponent_h, opponent_c = opponent_actor.act(
                observations, opponent_indices, opponent_h, opponent_c,
            )
            action_values = np.zeros((workers, HERO_COUNT, 4), dtype=np.int16)
            for index, side in enumerate(assignments):
                if side == 0:
                    continue
                candidate_local = 0 if side == 1 else HERO_COUNT // 2
                opponent_local = HERO_COUNT // 2 - candidate_local
                source = slice(index * (HERO_COUNT // 2), (index + 1) * (HERO_COUNT // 2))
                action_values[index, candidate_local:candidate_local + HERO_COUNT // 2] = candidate_actions[source]
                action_values[index, opponent_local:opponent_local + HERO_COUNT // 2] = opponent_actions[source]
                actions += np.bincount(
                    candidate_actions[source, 0], minlength=len(ACTION_NAMES),
                )
            results = env.step(action_values)
            reset_indices: list[int] = []
            reset_seeds: list[int] = []
            for index, result in enumerate(results):
                side = assignments[index]
                if side == 0:
                    continue
                local = slice(0, HERO_COUNT // 2) if side == 1 else slice(HERO_COUNT // 2, HERO_COUNT)
                candidate_reward += float(result.rewards[local].sum())
                invalid += int(result.invalid[local].sum())
                if not result.done:
                    continue
                completed += 1
                total_steps += result.step
                outcome = "draw" if result.winner == 0 else ("win" if result.winner == side else "loss")
                outcomes[outcome] += 1
                side_outcomes[side][outcome] += 1
                if next_match < matches:
                    replacement_side = 1 + next_match % 2
                    next_match += 1
                    assignments[index] = replacement_side
                    reset_indices.append(index)
                    reset_seeds.append(next_seed)
                    next_seed += 1
                else:
                    assignments[index] = 0
                start, stop = index * (HERO_COUNT // 2), (index + 1) * (HERO_COUNT // 2)
                candidate_h[start:stop] = candidate_c[start:stop] = 0
                opponent_h[start:stop] = opponent_c[start:stop] = 0
            observations = results
            if reset_indices:
                replacements = env.reset_indices(
                    reset_indices, reset_seeds, max_steps,
                    controller_sets=[controllers] * len(reset_indices),
                    rosters=self_play_rosters(roster_rng, len(reset_indices)),
                )
                for index, replacement in replacements.items():
                    observations[index] = replacement
    elapsed = time.perf_counter() - started
    wins, losses, draws = outcomes["win"], outcomes["loss"], outcomes["draw"]
    lower, upper = wilson_interval(wins, matches)
    action_total = max(int(actions.sum()), 1)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_update": int(saved.get("update", 0)),
        "checkpoint_hero_steps": int(saved.get("hero_steps", 0)),
        "opponent": opponent_label or opponent_checkpoint.stem,
        "opponent_checkpoint": str(opponent_checkpoint.resolve()),
        "seed": seed, "matches": matches, "workers": workers,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / matches,
        "score_rate": (wins + 0.5 * draws) / matches,
        "draw_rate": draws / matches,
        "win_rate_ci95_low": lower, "win_rate_ci95_high": upper,
        "side_1": dict(side_outcomes[1]), "side_2": dict(side_outcomes[2]),
        "invalid_actions": invalid,
        "invalid_action_rate": invalid / action_total,
        "mean_ai40_team_reward": candidate_reward / matches,
        "mean_match_steps": total_steps / matches,
        "mean_match_minutes": total_steps * 0.2 / 60.0 / matches,
        "action_counts": {name: int(actions[index]) for index, name in enumerate(ACTION_NAMES)},
        "action_rates": {name: float(actions[index] / action_total) for index, name in enumerate(ACTION_NAMES)},
        "elapsed_seconds": elapsed,
        "matches_per_second": matches / max(elapsed, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_vs_ai30(
        args.checkpoint, args.env, args.matches, args.workers, args.max_steps,
        torch.device(args.device), args.seed,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
