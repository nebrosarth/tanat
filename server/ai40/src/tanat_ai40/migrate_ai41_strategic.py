from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import (
    AI41_NAVIGATION_SCHEMA_HASH,
    AI41_REWARD_HASH,
    AI41_STRATEGIC_REWARD_HASH,
    AI41_STRATEGIC_REWARD_HASH_V4,
)
from .model_ai41 import AI41NavigationPolicy


def migrate_checkpoint(source: Path, output: Path) -> None:
    """Reuse AI-41 weights under the calibrated Tanat reward with fresh PPO state."""
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != AI41_NAVIGATION_SCHEMA_HASH.hex():
        raise RuntimeError("source is not an AI-41 navigation checkpoint")
    if saved.get("reward_hash") not in (
        AI41_REWARD_HASH.hex(), AI41_STRATEGIC_REWARD_HASH_V4.hex(),
        AI41_STRATEGIC_REWARD_HASH.hex(),
    ):
        raise RuntimeError("source has an unsupported reward contract")
    policy = AI41NavigationPolicy()
    policy.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    config = dict(saved.get("config", {}))
    config.update({
        "model_version": "AI-41-v5-tanat-reward-selfplay",
        "migration_source": str(source.resolve()),
        "completed_matches": {},
        "outcomes": {},
        "optimizer_reinitialized_for_reward_contract": True,
        "shaping_time_weight": "0.6^(elapsed_seconds/600)",
        "draw_timeout_reward_post_zero_sum": -2.0,
        "tanat_creep_last_hit_bonus": 0.24,
        "standard_wave_last_hit_mean": 0.4,
        "discount_horizon_seconds": 1_200.0,
        "gae_horizon_seconds": 180.0,
    })
    migrated = {
        "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
        "schema_hash": AI41_NAVIGATION_SCHEMA_HASH.hex(),
        "reward_hash": AI41_STRATEGIC_REWARD_HASH.hex(),
        "update": int(saved.get("update", 0)),
        "hero_steps": int(saved.get("hero_steps", 0)),
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    print(f"AI-41 strategic checkpoint: {output} (weights from {source})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
