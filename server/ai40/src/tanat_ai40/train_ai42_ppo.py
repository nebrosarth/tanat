"""Run bounded on-policy self-play PPO updates for AI-42."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .env import (
    AI42_EVALUATION_PROTOCOL_VERSION,
    AssaultVectorEnv,
    CONTROLLER_AI40,
    HERO_COUNT,
    self_play_rosters,
)
from .evaluate_ai42 import load_actor
from .evaluate_ai42 import AI42EvaluationActor
from .model_ai42_actor import AI42Actor
from .ppo_ai42 import (
    AI42ActorCritic,
    AI42PPOConfig,
    PPO_CHECKPOINT_FORMAT,
    collect_rollout,
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_model_kwargs(config_path: Path) -> dict[str, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("AI-42 config is missing model settings")
    names = ("hidden_size", "model_width", "entity_layers", "num_heads", "ff_multiplier")
    result = {name: int(model[name]) for name in names}
    return result


def _load_training_model(
    checkpoint: Path,
    config_path: Path,
    device: torch.device,
    ppo_config: AI42PPOConfig,
) -> tuple[AI42ActorCritic, torch.optim.Optimizer, dict[str, Any], int, int]:
    model_kwargs = _read_model_kwargs(config_path)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = AI42Actor(**model_kwargs)
    model = AI42ActorCritic(actor)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ppo_config.learning_rate,
        weight_decay=ppo_config.weight_decay,
    )
    if isinstance(raw, Mapping) and raw.get("format") == PPO_CHECKPOINT_FORMAT:
        saved_kwargs = dict(raw.get("model_kwargs", {}))
        if saved_kwargs != model_kwargs:
            raise ValueError("PPO checkpoint model settings do not match the config")
        payload = load_ppo_checkpoint(checkpoint, model, optimizer, map_location=device)
        lineage = {
            "kind": "ppo_resume",
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            "parent": dict(payload.get("source_lineage", {})),
        }
        update = int(payload.get("update", 0))
        hero_steps = int(payload.get("hero_steps", 0))
    else:
        loaded_actor, bc_lineage = load_actor(checkpoint, config_path, device)
        model.actor.load_state_dict(loaded_actor.state_dict(), strict=True)
        lineage = {
            "kind": "behavior_cloning_bootstrap",
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            **bc_lineage,
        }
        update = 0
        hero_steps = 0
    model.train().requires_grad_(True)
    return model, optimizer, lineage, update, hero_steps


def train_self_play(
    checkpoint: Path,
    config_path: Path,
    env_path: Path,
    output_dir: Path,
    *,
    device: torch.device,
    workers: int,
    rollout_steps: int,
    max_match_steps: int,
    train_seconds: float,
    seed: int,
    checkpoint_interval: int,
    max_updates: int | None,
    ppo_config: AI42PPOConfig,
    past_opponent_fraction: float = 0.2,
) -> dict[str, Any]:
    if workers < 1 or rollout_steps < 1 or max_match_steps < 1:
        raise ValueError("workers, rollout_steps, and max_match_steps must be positive")
    if train_seconds <= 0:
        raise ValueError("train_seconds must be positive")
    if not 0.0 <= past_opponent_fraction < 1.0:
        raise ValueError("past_opponent_fraction must be in [0, 1)")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, optimizer, lineage, update, hero_steps = _load_training_model(
        checkpoint, config_path, device, ppo_config,
    )
    model_kwargs = _read_model_kwargs(config_path)
    frozen_opponent = None
    opponent_slots: list[tuple[int, int]] = []
    controlled_mask = np.ones((workers, HERO_COUNT), dtype=np.bool_)
    opponent_workers = min(workers - 1, int(round(workers * past_opponent_fraction)))
    if past_opponent_fraction > 0 and opponent_workers == 0 and workers > 1:
        opponent_workers = 1
    if opponent_workers:
        opponent_actor, _ = load_actor(checkpoint, config_path, device)
        frozen_opponent = AI42EvaluationActor(
            opponent_actor, opponent_workers * (HERO_COUNT // 2), device,
        )
        first_opponent_worker = workers - opponent_workers
        for local, worker in enumerate(range(first_opponent_worker, workers)):
            opponent_side = 1 + local % 2
            opponent_slots.append((worker, opponent_side))
            start = 0 if opponent_side == 1 else HERO_COUNT // 2
            controlled_mask[worker, start:start + HERO_COUNT // 2] = False
    rng = np.random.default_rng(seed)
    controllers = (CONTROLLER_AI40,) * HERO_COUNT
    seeds = [int(rng.integers(1, 2**63 - 1)) for _ in range(workers)]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    rollout_seconds = optimizer_seconds = 0.0
    match_totals = {
        "matches": 0, "side_1_wins": 0, "side_2_wins": 0,
        "draws": 0, "invalid_actions": 0,
    }
    latest_checkpoint: Path | None = None
    with AssaultVectorEnv(
        env_path, workers, AI42_EVALUATION_PROTOCOL_VERSION,
    ) as env:
        observations = env.reset(
            seeds, max_match_steps, controllers=controllers,
            rosters=self_play_rosters(rng, workers),
        )
        from .ppo_ai42 import AI42StochasticPolicy
        policy = AI42StochasticPolicy(
            model, workers * HERO_COUNT, device,
            controlled_mask=controlled_mask.reshape(-1),
        )
        while True:
            rollout_started = time.perf_counter()
            rollout, observations, outcomes = collect_rollout(
                env, policy, observations, horizon=rollout_steps, seed_rng=rng,
                max_steps=max_match_steps, controllers=controllers,
                roster_factory=self_play_rosters,
                opponent_runner=frozen_opponent,
                opponent_slots=opponent_slots,
            )
            rollout_elapsed = time.perf_counter() - rollout_started
            rollout_seconds += rollout_elapsed
            optimizer_started = time.perf_counter()
            metrics = ppo_update(
                model, optimizer, rollout, ppo_config, device,
                mixed_precision=True,
            )
            optimizer_elapsed = time.perf_counter() - optimizer_started
            optimizer_seconds += optimizer_elapsed
            update += 1
            hero_steps += rollout_steps * workers * HERO_COUNT
            for name, value in outcomes.items():
                match_totals[name] += int(value)
            record = {
                "update": update,
                "hero_steps": hero_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "rollout_seconds": rollout_elapsed,
                "optimizer_seconds": optimizer_elapsed,
                **outcomes,
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
            if checkpoint_interval > 0 and update % checkpoint_interval == 0:
                latest_checkpoint = save_ppo_checkpoint(
                    output_dir / f"update-{update:06d}.pt", model, optimizer,
                    ppo_config, model_kwargs=model_kwargs, update=update,
                    hero_steps=hero_steps, source_lineage=lineage, metrics=record,
                )
            elapsed = time.perf_counter() - started
            if elapsed >= train_seconds or (max_updates is not None and len(history) >= max_updates):
                break
    latest_checkpoint = save_ppo_checkpoint(
        output_dir / "final.pt", model, optimizer, ppo_config,
        model_kwargs=model_kwargs, update=update, hero_steps=hero_steps,
        source_lineage=lineage, metrics=history[-1],
    )
    report = {
        "format": "AI42-ppo-training-report-v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": _sha256(checkpoint),
        "final_checkpoint": str(latest_checkpoint.resolve()),
        "final_sha256": _sha256(latest_checkpoint),
        "device": str(device),
        "workers": workers,
        "rollout_steps": rollout_steps,
        "past_opponent_fraction": past_opponent_fraction,
        "past_opponent_workers": opponent_workers,
        "updates": len(history),
        "first_update": history[0]["update"],
        "last_update": history[-1]["update"],
        "hero_steps": hero_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "rollout_seconds": rollout_seconds,
        "optimizer_seconds": optimizer_seconds,
        "outcomes": match_totals,
        "last_metrics": history[-1],
        "ppo_config": ppo_config.__dict__ if hasattr(ppo_config, "__dict__") else {
            field: getattr(ppo_config, field) for field in ppo_config.__dataclass_fields__
        },
    }
    report_path = output_dir / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI-42 with bounded self-play PPO")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-match-steps", type=int, default=4_500)
    parser.add_argument("--train-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=73_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--minibatch-actors", type=int, default=16)
    parser.add_argument("--past-opponent-fraction", type=float, default=0.2)
    args = parser.parse_args()
    ppo_config = AI42PPOConfig(
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_actors=args.minibatch_actors,
    )
    report = train_self_play(
        args.checkpoint, args.config, args.env, args.output,
        device=torch.device(args.device), workers=args.workers,
        rollout_steps=args.rollout_steps, max_match_steps=args.max_match_steps,
        train_seconds=args.train_seconds, seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        max_updates=args.max_updates, ppo_config=ppo_config,
        past_opponent_fraction=args.past_opponent_fraction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
