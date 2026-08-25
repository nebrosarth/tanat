from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import (
    AI41_LEGACY_SCHEMA_HASH,
    AI41_REWARD_HASH,
    AI41_SCHEMA_HASH,
    REWARD_HASH,
)


def migrate_checkpoint(source: Path, output: Path) -> None:
    """Carry AI-41-v1 weights into the versioned wrong-lane curriculum."""
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != AI41_LEGACY_SCHEMA_HASH.hex():
        raise RuntimeError("source is not an AI-41-v1 checkpoint")
    if saved.get("reward_hash") != REWARD_HASH.hex():
        raise RuntimeError("source reward schema is not AssaultRewardV2")

    migrated = dict(saved)
    model = dict(saved["model"])
    global_input = model.get("global_encoder.0.weight")
    if global_input is None or global_input.ndim != 2 or global_input.shape[1] < 14:
        raise RuntimeError("checkpoint has no compatible global encoder")
    global_input = global_input.clone()
    # These five columns were always zero in AI-41-v1. Neutral initialization
    # preserves old behavior on the first v2 rollout while leaving the weights
    # fully trainable once assignment features become nonzero.
    global_input[:, 8:13].zero_()
    model["global_encoder.0.weight"] = global_input
    config = dict(saved.get("config", {}))
    config.update({
        "model_version": "AI-41-v2-wrong-lane",
        "migration_source": str(source.resolve()),
        "wrong_lane_penalty_per_second": 0.15,
        "lane_assignment": "randomized-2-1-2",
        "lane_assignment_seconds": [360, 600],
        "lane_assignment_time_observation": False,
        "lane_corridor": 30.0,
    })
    migrated.update({
        "model": model,
        "schema_hash": AI41_SCHEMA_HASH.hex(),
        "reward_hash": AI41_REWARD_HASH.hex(),
        "config": config,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    print(
        f"AI-41 wrong-lane checkpoint: {output} "
        f"(weights/update {int(saved.get('update', 0))} preserved from {source})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
