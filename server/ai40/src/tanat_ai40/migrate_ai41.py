from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import AI41_SCHEMA_HASH, REWARD_HASH, SCHEMA_HASH
from .model_ai41 import AI41Policy, migrate_ai40_state


def migrate_checkpoint(source: Path, output: Path) -> None:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != SCHEMA_HASH.hex():
        raise RuntimeError("source is not an AssaultObservationV1/AI-40 checkpoint")
    if saved.get("reward_hash") != REWARD_HASH.hex():
        raise RuntimeError("source reward schema is incompatible")

    policy = AI41Policy()
    copied = migrate_ai40_state(policy, saved["model"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    config = dict(saved.get("config", {}))
    config.update({
        "model_version": "AI-41-v1",
        "migration_source": str(source.resolve()),
        "migration_copied_tensors": len(copied),
        "ability_features": 40,
        "action_conditioned_heads": True,
    })
    migrated = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schema_hash": AI41_SCHEMA_HASH.hex(),
        "reward_hash": REWARD_HASH.hex(),
        "update": int(saved.get("update", 0)),
        "hero_steps": int(saved.get("hero_steps", 0)),
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    print(
        f"AI-41 checkpoint: {output} (copied {len(copied)} tensors from {source})",
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
