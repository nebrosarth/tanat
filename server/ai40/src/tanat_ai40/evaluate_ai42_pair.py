"""Headless paired evaluation of an AI-42 candidate against its champion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .env import (
    AI42_EVALUATION_PROTOCOL_VERSION,
    AssaultVectorEnv,
    CONTROLLER_AI40,
    HERO_COUNT,
    self_play_rosters,
)
from .evaluate_ai30 import controlled_slot_indices, wilson_interval
from .evaluate_ai42 import (
    AI42EvaluationActor,
    RUNTIME_CONTROL_ISSUE,
    load_actor,
)


def evaluate_pair(
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    config: Path,
    executable: Path,
    *,
    matches: int,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int = 84_000,
) -> dict[str, Any]:
    if matches < 1 or workers < 1 or max_steps < 1:
        raise ValueError("matches, workers, and max_steps must be positive")
    workers = min(workers, matches)
    candidate, candidate_lineage = load_actor(candidate_checkpoint, config, device)
    champion, champion_lineage = load_actor(champion_checkpoint, config, device)
    candidate_runner = AI42EvaluationActor(
        candidate, workers * (HERO_COUNT // 2), device,
    )
    champion_runner = AI42EvaluationActor(
        champion, workers * (HERO_COUNT // 2), device,
    )
    assignments = [1 + index % 2 for index in range(workers)]
    next_match = workers
    next_seed = seed + workers
    roster_rng = np.random.default_rng(seed)
    controllers = (CONTROLLER_AI40,) * HERO_COUNT
    outcomes: Counter[str] = Counter()
    side_outcomes = {1: Counter(), 2: Counter()}
    candidate_reward = champion_reward = 0.0
    candidate_invalid = champion_invalid = 0
    candidate_actions = champion_actions = 0
    total_steps = 0
    started = time.perf_counter()
    with AssaultVectorEnv(
        executable, workers, AI42_EVALUATION_PROTOCOL_VERSION,
    ) as env:
        observations = env.reset(
            range(seed, seed + workers), max_steps, controllers=controllers,
            rosters=self_play_rosters(roster_rng, workers),
        )
        completed = 0
        while completed < matches:
            candidate_indices = controlled_slot_indices(assignments, candidate=True)
            champion_indices = controlled_slot_indices(assignments, candidate=False)
            candidate_values, candidate_control, _ = candidate_runner.act(
                observations, candidate_indices,
            )
            champion_values, champion_control, _ = champion_runner.act(
                observations, champion_indices,
            )
            wire = np.zeros((workers, HERO_COUNT, 5), dtype=np.int16)
            for worker, side in enumerate(assignments):
                if side == 0:
                    continue
                source = slice(
                    worker * (HERO_COUNT // 2),
                    (worker + 1) * (HERO_COUNT // 2),
                )
                candidate_start = 0 if side == 1 else HERO_COUNT // 2
                champion_start = HERO_COUNT // 2 if side == 1 else 0
                wire[worker, candidate_start:candidate_start + 5, 0] = candidate_control[source]
                wire[worker, candidate_start:candidate_start + 5, 1:] = candidate_values[source]
                wire[worker, champion_start:champion_start + 5, 0] = champion_control[source]
                wire[worker, champion_start:champion_start + 5, 1:] = champion_values[source]
                candidate_actions += int((candidate_control[source] == RUNTIME_CONTROL_ISSUE).sum())
                champion_actions += int((champion_control[source] == RUNTIME_CONTROL_ISSUE).sum())
            results = env.step(wire)
            reset_indices: list[int] = []
            reset_seeds: list[int] = []
            for worker, result in enumerate(results):
                side = assignments[worker]
                if side == 0:
                    continue
                candidate_slice = slice(0, 5) if side == 1 else slice(5, 10)
                champion_slice = slice(5, 10) if side == 1 else slice(0, 5)
                candidate_reward += float(result.rewards[candidate_slice].sum())
                champion_reward += float(result.rewards[champion_slice].sum())
                candidate_invalid += int(result.invalid[candidate_slice].sum())
                champion_invalid += int(result.invalid[champion_slice].sum())
                if not result.done:
                    continue
                completed += 1
                total_steps += result.step
                outcome = (
                    "draw" if result.winner == 0
                    else ("win" if result.winner == side else "loss")
                )
                outcomes[outcome] += 1
                side_outcomes[side][outcome] += 1
                if next_match < matches:
                    assignments[worker] = 1 + next_match % 2
                    next_match += 1
                    reset_indices.append(worker)
                    reset_seeds.append(next_seed)
                    next_seed += 1
                else:
                    assignments[worker] = 0
                candidate_runner.reset_workers([worker])
                champion_runner.reset_workers([worker])
            observations = results
            if reset_indices:
                replacements = env.reset_indices(
                    reset_indices, reset_seeds, max_steps, controllers=controllers,
                    rosters=self_play_rosters(roster_rng, len(reset_indices)),
                )
                for worker, replacement in replacements.items():
                    observations[worker] = replacement
    wins = int(outcomes["win"])
    losses = int(outcomes["loss"])
    draws = int(outcomes["draw"])
    low, high = wilson_interval(wins, matches)
    return {
        "format": "AI42-paired-evaluation-v1",
        "candidate": str(candidate_checkpoint.resolve()),
        "champion": str(champion_checkpoint.resolve()),
        "candidate_lineage": candidate_lineage,
        "champion_lineage": champion_lineage,
        "seed": seed,
        "matches": matches,
        "workers": workers,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / matches,
        "score_rate": (wins + 0.5 * draws) / matches,
        "win_rate_ci95_low": low,
        "win_rate_ci95_high": high,
        "side_1": dict(side_outcomes[1]),
        "side_2": dict(side_outcomes[2]),
        "mean_candidate_reward": candidate_reward / matches,
        "mean_champion_reward": champion_reward / matches,
        "candidate_invalid_actions": candidate_invalid,
        "champion_invalid_actions": champion_invalid,
        "candidate_issued_actions": candidate_actions,
        "champion_issued_actions": champion_actions,
        "mean_match_minutes": total_steps * 0.2 / 60.0 / matches,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI-42 candidate against champion")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("champion", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=84_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_pair(
        args.candidate, args.champion, args.config, args.env,
        matches=args.matches, workers=args.workers, max_steps=args.max_steps,
        device=torch.device(args.device), seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
