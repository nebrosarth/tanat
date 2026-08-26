"""One bounded AI-42 train/evaluate cycle with no implicit promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate_ai42 import evaluate_vs_ai30
from .evaluate_ai42_pair import evaluate_pair
from .ppo_ai42 import AI42PPOConfig
from .train_ai42_ppo import train_self_play


def run_cycle(
    champion: Path,
    config: Path,
    env: Path,
    output: Path,
    *,
    device: torch.device,
    train_seconds: float,
    workers: int,
    rollout_steps: int,
    max_match_steps: int,
    seed: int,
    eval_matches: int,
    eval_workers: int,
    ppo_config: AI42PPOConfig,
    past_opponent_fraction: float,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = train_self_play(
        champion, config, env, output / "training", device=device,
        workers=workers, rollout_steps=rollout_steps,
        max_match_steps=max_match_steps, train_seconds=train_seconds,
        seed=seed, checkpoint_interval=10, max_updates=None,
        ppo_config=ppo_config,
        past_opponent_fraction=past_opponent_fraction,
    )
    candidate = Path(training["final_checkpoint"])
    paired = evaluate_pair(
        candidate, champion, config, env, matches=eval_matches,
        workers=eval_workers, max_steps=max_match_steps, device=device,
        seed=seed + 10_000,
    )
    candidate_anchor = evaluate_vs_ai30(
        candidate, config, env, eval_matches, eval_workers, max_match_steps,
        device, seed + 20_000,
    )
    champion_anchor = evaluate_vs_ai30(
        champion, config, env, eval_matches, eval_workers, max_match_steps,
        device, seed + 20_000,
    )
    # This is deliberately a recommendation, not a filesystem mutation.  A
    # candidate must outperform the champion head-to-head, improve shaped
    # reward, avoid action-validity regression, and not regress versus AI-30.
    checks = {
        "paired_score_above_half": paired["score_rate"] > 0.5,
        "paired_reward_improved": (
            paired["mean_candidate_reward"] > paired["mean_champion_reward"]
        ),
        "invalid_actions_not_worse": (
            paired["candidate_invalid_actions"] <= paired["champion_invalid_actions"]
        ),
        "ai30_score_not_worse": (
            candidate_anchor["score_rate"] >= champion_anchor["score_rate"]
        ),
    }
    report = {
        "format": "AI42-ppo-cycle-report-v1",
        "champion": str(champion.resolve()),
        "candidate": str(candidate.resolve()),
        "training": training,
        "paired_evaluation": paired,
        "candidate_vs_ai30": candidate_anchor,
        "champion_vs_ai30": champion_anchor,
        "promotion_checks": checks,
        "promotion_recommended": all(checks.values()),
        "promotion_applied": False,
    }
    for name, value in (
        ("paired-evaluation.json", paired),
        ("candidate-vs-ai30.json", candidate_anchor),
        ("champion-vs-ai30.json", champion_anchor),
        ("cycle-report.json", report),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AI-42 PPO train/evaluate cycle")
    parser.add_argument("champion", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-seconds", type=float, default=900.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-match-steps", type=int, default=4_500)
    parser.add_argument("--seed", type=int, default=93_000)
    parser.add_argument("--eval-matches", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--past-opponent-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--minibatch-actors", type=int, default=16)
    args = parser.parse_args()
    report = run_cycle(
        args.champion, args.config, args.env, args.output,
        device=torch.device(args.device), train_seconds=args.train_seconds,
        workers=args.workers, rollout_steps=args.rollout_steps,
        max_match_steps=args.max_match_steps, seed=args.seed,
        eval_matches=args.eval_matches, eval_workers=args.eval_workers,
        ppo_config=AI42PPOConfig(
            learning_rate=args.learning_rate,
            update_epochs=args.update_epochs,
            minibatch_actors=args.minibatch_actors,
        ),
        past_opponent_fraction=args.past_opponent_fraction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
