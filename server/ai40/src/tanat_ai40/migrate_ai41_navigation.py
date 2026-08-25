from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import torch

from .env import (
    AI41_NAVIGATION_SCHEMA_HASH,
    AI41_REWARD_HASH,
    AI41_SCHEMA_HASH,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from .model_ai41 import AI41NavigationPolicy


def _legacy_factor(index: int) -> tuple[int, int] | None:
    """Map one 9x9 cell to the nearest legacy direction/distance pair."""
    x, y = index % 9 - 4, index // 9 - 4
    if x == 0 and y == 0:
        return None
    angle = math.atan2(y, x) % (2 * math.pi)
    direction = int(round(angle * 16 / (2 * math.pi))) % 16
    radius = math.hypot(x * 3, y * 3)
    distance = min(range(3), key=lambda value: abs((4, 8, 12)[value] - radius))
    return direction, distance


def migrate_navigation_state(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    policy = AI41NavigationPolicy()
    target = policy.state_dict()
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name] = value.clone()

    old_direction_weight = source["direction_head.weight"]
    old_direction_bias = source["direction_head.bias"]
    old_distance_weight = source["distance_head.weight"]
    old_distance_bias = source["distance_head.bias"]
    if old_direction_weight.shape[0] != 16 or old_distance_weight.shape[0] != 3:
        raise RuntimeError("source does not use the legacy 16x3 position heads")

    factors = [_legacy_factor(index) for index in range(NAVIGATION_OFFSETS)]
    multiplicity = Counter(factor for factor in factors if factor is not None)
    offset_weight = target["direction_head.weight"]
    offset_bias = target["direction_head.bias"]
    for index, factor in enumerate(factors):
        if factor is None:
            offset_weight[index].zero_()
            offset_bias[index] = -8
            continue
        direction, distance = factor
        offset_weight[index] = (
            old_direction_weight[direction] + old_distance_weight[distance]
        )
        offset_bias[index] = (
            old_direction_bias[direction] + old_distance_bias[distance]
            - math.log(multiplicity[factor])
        )

    # Anchor 0 means local-grid movement. Start global navigation at about 0.47%
    # aggregate probability so migration preserves play while PPO can explore it.
    anchor_weight = target["distance_head.weight"]
    anchor_bias = target["distance_head.bias"]
    anchor_weight.zero_()
    anchor_bias.fill_(-4)
    anchor_bias[0] = 4
    return target


def migrate_checkpoint(source: Path, output: Path) -> None:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != AI41_SCHEMA_HASH.hex():
        raise RuntimeError("source is not an AI-41-v2 wrong-lane checkpoint")
    if saved.get("reward_hash") != AI41_REWARD_HASH.hex():
        raise RuntimeError("source reward schema is not AssaultRewardV3")

    policy = AI41NavigationPolicy()
    policy.load_state_dict(migrate_navigation_state(saved["model"]))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    config = dict(saved.get("config", {}))
    config.update({
        "model_version": "AI-41-v3-navigation",
        "migration_source": str(source.resolve()),
        "navigation_local_grid": [9, 9],
        "navigation_local_spacing": 3.0,
        "navigation_anchors": NAVIGATION_ANCHORS,
        "navigation_anchor_layout": "local,bases,lanes3x4",
        "optimizer_reinitialized_for_action_head_surgery": True,
    })
    migrated = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schema_hash": AI41_NAVIGATION_SCHEMA_HASH.hex(),
        "reward_hash": AI41_REWARD_HASH.hex(),
        "update": int(saved.get("update", 0)),
        "hero_steps": int(saved.get("hero_steps", 0)),
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    print(
        f"AI-41 navigation checkpoint: {output} "
        f"(core/update {migrated['update']} preserved from {source})",
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
