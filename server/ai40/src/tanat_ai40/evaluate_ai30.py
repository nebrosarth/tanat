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
    self_play_rosters,
)
from .model import AI40Policy, masked_categorical
from .train import effective_target_mask, stack_observations


ACTION_NAMES = ("wait", "move", "attack", "skill1", "skill2", "skill3", "skill4", "teleport")


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
) -> dict:
    if matches < 1 or workers < 1:
        raise ValueError("matches and workers must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != SCHEMA_HASH.hex() or saved.get("reward_hash") != REWARD_HASH.hex():
        raise RuntimeError("checkpoint schema/reward hash mismatch")
    policy = AI40Policy().to(device)
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
    with AssaultVectorEnv(executable, workers) as env:
        observations = env.reset(
            range(seed, seed + workers), max_steps,
            controller_sets=[controllers_for_side(side) for side in assignments],
            rosters=self_play_rosters(roster_rng, workers),
        )
        next_seed = seed + workers
        h, c = policy.initial_state(workers * HERO_COUNT, device)
        completed = 0
        while completed < matches:
            batch = stack_observations(observations)
            with torch.no_grad():
                out = policy(
                    torch.as_tensor(batch.hero, device=device),
                    torch.as_tensor(batch.entities, device=device),
                    torch.as_tensor(batch.global_state, device=device),
                    torch.as_tensor(batch.entity_mask, device=device), h, c,
                )
                kind_mask = torch.as_tensor(batch.kind_mask, device=device).bool()
                kinds = masked_categorical(out["kind"], kind_mask).probs.argmax(-1)
                target_mask = effective_target_mask(batch, kinds, device)
                targets = out["target"].masked_fill(~target_mask, -1e9).argmax(-1)
                directions = out["direction"].argmax(-1)
                distances = out["distance"].argmax(-1)
            action_values = torch.stack((kinds, targets, directions, distances), dim=-1).cpu().numpy()
            results = env.step(action_values.reshape(workers, HERO_COUNT, 4))
            h, c = out["h"].detach(), out["c"].detach()
            reset_indices: list[int] = []
            reset_seeds: list[int] = []
            reset_controllers: list[tuple[int, ...]] = []
            for index, result in enumerate(results):
                side = assignments[index]
                if side == 0:
                    continue
                start = index * HERO_COUNT + (0 if side == 1 else HERO_COUNT // 2)
                stop = start + HERO_COUNT // 2
                actions += np.bincount(action_values.reshape(-1, 4)[start:stop, 0], minlength=len(ACTION_NAMES))
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
                hero_start, hero_stop = index * HERO_COUNT, (index + 1) * HERO_COUNT
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
