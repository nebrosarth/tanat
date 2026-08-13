from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .env import (
    AI40_SELF_PLAY_CONTROLLERS,
    AssaultVectorEnv,
    HeroAction,
    HERO_COUNT,
    REWARD_HASH,
    SCHEMA_HASH,
    self_play_rosters,
)
from .model import AI40Policy, masked_categorical
from .train import effective_target_mask, stack_observations


def evaluate(checkpoint: Path, executable: Path, matches: int, workers: int,
             max_steps: int, device: torch.device) -> None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != SCHEMA_HASH.hex() or saved.get("reward_hash") != REWARD_HASH.hex():
        raise RuntimeError("checkpoint schema/reward hash mismatch")
    policy = AI40Policy().to(device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    played = human_wins = elf_wins = draws = invalid = 0
    total_reward = np.zeros(HERO_COUNT, np.float64)
    next_seed = 10_000
    roster_rng = np.random.default_rng(10_000)
    with AssaultVectorEnv(executable, workers) as env:
        observations = env.reset(
            range(next_seed, next_seed + workers), max_steps,
            controllers=AI40_SELF_PLAY_CONTROLLERS,
            rosters=self_play_rosters(roster_rng, workers),
        )
        next_seed += workers
        h, c = policy.initial_state(workers * HERO_COUNT, device)
        while played < matches:
            batch = stack_observations(observations)
            with torch.no_grad():
                out = policy(torch.as_tensor(batch.hero, device=device),
                             torch.as_tensor(batch.entities, device=device),
                             torch.as_tensor(batch.global_state, device=device),
                             torch.as_tensor(batch.entity_mask, device=device), h, c)
                kinds = masked_categorical(out["kind"], torch.as_tensor(batch.kind_mask, device=device).bool()).probs.argmax(-1)
                target_mask = effective_target_mask(batch, kinds, device)
                targets = out["target"].masked_fill(~target_mask, -1e9).argmax(-1)
                directions = out["direction"].argmax(-1)
                distances = out["distance"].argmax(-1)
            action_values = torch.stack(
                (kinds, targets, directions, distances), dim=-1,
            ).cpu().numpy()
            results = env.step(action_values.reshape(workers, HERO_COUNT, 4))
            h, c = out["h"], out["c"]
            done_indices = []
            for index, result in enumerate(results):
                total_reward += result.rewards
                invalid += int(result.invalid.sum())
                if result.done:
                    played += 1
                    human_wins += result.winner == 1
                    elf_wins += result.winner == 2
                    draws += result.winner == 0
                    done_indices.append(index)
                    if played >= matches:
                        break
            observations = results
            if done_indices and played < matches:
                seeds = range(next_seed, next_seed + len(done_indices))
                next_seed += len(done_indices)
                replacements = env.reset_indices(
                    done_indices, seeds, max_steps,
                    controllers=AI40_SELF_PLAY_CONTROLLERS,
                    rosters=self_play_rosters(roster_rng, len(done_indices)),
                )
                for index, result in replacements.items():
                    observations[index] = result
                    h[index * HERO_COUNT:(index + 1) * HERO_COUNT] = 0
                    c[index * HERO_COUNT:(index + 1) * HERO_COUNT] = 0
    print(f"matches={played} human_wins={human_wins} elf_wins={elf_wins} draws={draws} "
          f"invalid={invalid} mean_reward={total_reward.mean()/max(played, 1):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.env, args.matches, args.workers, args.max_steps, torch.device(args.device))


if __name__ == "__main__":
    main()
