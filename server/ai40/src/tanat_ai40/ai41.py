from __future__ import annotations

from pathlib import Path

from .env import (
    AI41_NAVIGATION_SCHEMA_HASH,
    AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
    AI41_TEACHER_PROTOCOL_VERSION,
    AI41_STRATEGIC_REWARD_HASH,
)
from .evaluate_ai30 import evaluate_vs_ai30, evaluate_vs_checkpoint
from .model_ai41 import AI41NavigationPolicy
from .train_async import train_async
from .train_campaign import build_parser, run_campaign


def train_async_ai41(*args, **kwargs):
    kwargs.update({
        "policy_factory": AI41NavigationPolicy,
        "schema_hash": AI41_NAVIGATION_SCHEMA_HASH,
        "reward_hash": AI41_STRATEGIC_REWARD_HASH,
        # Live AI-30 teacher labels regularize PPO on every update, preserving
        # the bootstrap's attack/skill behavior against navigation collapse.
        "protocol_version": AI41_TEACHER_PROTOCOL_VERSION,
        "teacher_loss_weight": 0.15,
        "model_version": "AI-41-v5-tanat-reward-selfplay",
    })
    return train_async(*args, **kwargs)


def evaluate_ai41_vs_ai30(*args, **kwargs):
    kwargs.update({
        "policy_factory": AI41NavigationPolicy,
        "schema_hash": AI41_NAVIGATION_SCHEMA_HASH,
        "reward_hash": AI41_STRATEGIC_REWARD_HASH,
        "protocol_version": AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
    })
    return evaluate_vs_ai30(*args, **kwargs)


def evaluate_ai41_vs_checkpoint(*args, **kwargs):
    kwargs.update({
        "policy_factory": AI41NavigationPolicy,
        "schema_hash": AI41_NAVIGATION_SCHEMA_HASH,
        "reward_hash": AI41_STRATEGIC_REWARD_HASH,
        "protocol_version": AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
    })
    return evaluate_vs_checkpoint(*args, **kwargs)


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        output=Path("ai40/checkpoints/ai41-tanat-reward-stable-001"),
        discount_horizon_seconds=1_200.0,
        gae_horizon_seconds=180.0,
    )
    run_campaign(
        parser.parse_args(), trainer=train_async_ai41, evaluator=evaluate_ai41_vs_ai30,
        checkpoint_evaluator=evaluate_ai41_vs_checkpoint,
    )


if __name__ == "__main__":
    main()
