"""Bootstrap AI-41 from the aggressive scripted AI-30 policy.

This is deliberately a short, supervised warm start rather than an attempt to
turn the neural policy into a byte-for-byte port of the bot.  The server exports
only decisions that have a faithful policy-space representation (move/attack),
and PPO remains responsible for skills, delayed credit and self-play tactics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .env import (
    AI41_NAVIGATION_SCHEMA_HASH,
    AI41_STRATEGIC_REWARD_HASH,
    AI41_TEACHER_PROTOCOL_VERSION,
    AI40_ROSTER,
    AssaultVectorEnv,
    CONTROLLER_AI30,
    HERO_COUNT,
    self_play_rosters,
)
from .model_ai41 import AI41NavigationPolicy, selected_action_logits
from .train import save_checkpoint


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def tensors(batched: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    fields = ("hero", "abilities", "entities", "global_state", "entity_mask")
    return {
        name: torch.as_tensor(batched[name], device=device)
        for name in fields
    }


def clone_loss(output: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """Cross entropy for AI-30 labels; movement heads follow the nav contract."""
    kinds = actions[:, 0].long()
    target = actions[:, 1].long()
    direction = actions[:, 2].long()
    distance = actions[:, 3].long()
    selected = selected_action_logits(output, kinds)
    # Dense movement is the natural majority of five-Hz expert traces, while
    # abilities are sparse but decisive. Without balancing, a clone can obtain
    # a low aggregate loss simply by walking and never casting.
    kind_weight = torch.tensor(
        (0.0, 0.15, 1.0, 32.0, 32.0, 32.0, 32.0, 0.0), device=kinds.device,
    )
    loss = F.cross_entropy(selected["kind"], kinds, weight=kind_weight)
    counts = {
        "kind": int(kinds.numel()), "attack": 0, "move_local": 0,
        "move_anchor": 0, "skill": 0,
    }
    attack = kinds == 2
    if attack.any():
        loss = loss + F.cross_entropy(selected["target"][attack], target[attack])
        counts["attack"] = int(attack.sum())
    local_move = (kinds == 1) & (distance == 0)
    if local_move.any():
        loss = loss + F.cross_entropy(selected["direction"][local_move], direction[local_move])
        counts["move_local"] = int(local_move.sum())
    anchor_move = (kinds == 1) & (distance > 0)
    if anchor_move.any():
        loss = loss + F.cross_entropy(selected["distance"][anchor_move], distance[anchor_move])
        counts["move_anchor"] = int(anchor_move.sum())
    skill = (kinds >= 3) & (kinds <= 6)
    if skill.any():
        # Self/point abilities reserve target slot 0 in the action mask; unit
        # abilities carry their visible target slot from the server trace.
        loss = loss + 2.0 * F.cross_entropy(selected["target"][skill], target[skill])
        loss = loss + 2.0 * F.cross_entropy(selected["direction"][skill], direction[skill])
        counts["skill"] = int(skill.sum())
    return loss, counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior-clone AI-30 into an AI-41 bootstrap checkpoint")
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True, help="AI-41 navigation/strategic checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="checkpoint .pt to write")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2048, help="vector simulation ticks")
    parser.add_argument("--tbptt", type=int, default=8, help="truncated recurrent unroll")
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    return parser


def run(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.steps < 1 or args.tbptt < 1:
        raise ValueError("workers, steps and tbptt must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = AI41NavigationPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != AI41_NAVIGATION_SCHEMA_HASH.hex() or saved.get("reward_hash") != AI41_STRATEGIC_REWARD_HASH.hex():
        raise RuntimeError("AI-30 cloning requires an AI-41 navigation strategic checkpoint")
    policy.load_state_dict(saved["model"])
    policy.train()
    actors = args.workers * HERO_COUNT
    h, c = policy.initial_state(actors, device)
    zero_actions = np.zeros((args.workers, HERO_COUNT, 4), dtype=np.uint16)
    controls = (CONTROLLER_AI30,) * HERO_COUNT
    total_labels = total_loss = updates = 0
    label_counts = {"kind": 0, "attack": 0, "move_local": 0, "move_anchor": 0, "skill": 0}
    started = time.perf_counter()
    report(
        f"AI-30 behavior cloning: workers={args.workers} steps={args.steps} tbptt={args.tbptt} "
        f"device={device} source={args.resume}"
    )
    with AssaultVectorEnv(args.env, args.workers, AI41_TEACHER_PROTOCOL_VERSION) as env:
        env.reset(
            range(args.seed, args.seed + args.workers), max_steps=args.max_steps,
            controllers=controls, rosters=self_play_rosters(rng, args.workers),
        )
        accumulated: torch.Tensor | None = None
        for step in range(1, args.steps + 1):
            results = env.step(zero_actions)
            batched = results.batched_observations
            assert batched is not None
            valid = torch.as_tensor(batched["teacher_valid"].astype(bool), device=device)
            data = tensors(batched, device)
            output = policy(data["hero"], data["abilities"], data["entities"], data["global_state"], data["entity_mask"], h, c)
            h, c = output["h"], output["c"]
            if valid.any():
                raw = batched["teacher_actions"]
                packed = np.stack((raw["kind"], raw["target"], raw["direction"], raw["distance"]), axis=1)
                # NumPy preserves the wire's uint16 target field. CUDA does
                # not implement boolean indexing for UInt16 tensors, whereas
                # action labels are categorical indices anyway.
                action = torch.as_tensor(packed, device=device, dtype=torch.long)[valid]
                loss, counts = clone_loss({key: value[valid] for key, value in output.items()}, action)
                accumulated = loss if accumulated is None else accumulated + loss
                total_labels += int(valid.sum())
                for key, value in counts.items():
                    label_counts[key] += value
            if step % args.tbptt == 0 or step == args.steps:
                if accumulated is not None:
                    optimizer.zero_grad(set_to_none=True)
                    accumulated.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(accumulated.detach())
                    updates += 1
                accumulated = None
                h, c = h.detach(), c.detach()
            done = [index for index, result in enumerate(results) if result.done]
            if done:
                env.reset_indices(
                    done,
                    rng.integers(1, 2**31 - 1, size=len(done)).tolist(), args.max_steps,
                    controllers=controls, rosters=self_play_rosters(rng, len(done)),
                )
                reset_actors = torch.as_tensor(
                    np.repeat(np.asarray(done) * HERO_COUNT, HERO_COUNT) +
                    np.tile(np.arange(HERO_COUNT), len(done)), device=device,
                )
                # This may happen in the middle of a truncated BPTT window.
                # Do not mutate LSTM outputs retained by earlier losses;
                # index_fill returns the fresh match state without invalidating
                # their saved autograd tensors.
                h = h.index_fill(0, reset_actors, 0.0)
                c = c.index_fill(0, reset_actors, 0.0)
            if step % max(1, args.steps // 8) == 0 or step == args.steps:
                elapsed = time.perf_counter() - started
                report(f"clone step={step}/{args.steps} labels={total_labels} updates={updates} elapsed={elapsed:.1f}s")
    config = {
        "training_mode": "ai30_behavior_clone",
        "clone_source": str(args.resume), "clone_steps": args.steps,
        "clone_workers": args.workers, "clone_tbptt": args.tbptt,
        "clone_learning_rate": args.learning_rate, "clone_labels": total_labels,
        "clone_label_counts": label_counts,
        # Cloning trajectories are not PPO campaign matches.
        "completed_matches": {"ai40": 0, "ai30": 0, "historical": 0},
    }
    save_checkpoint(args.output, policy, optimizer, int(saved.get("update", 0)), int(saved.get("hero_steps", 0)), config,
                    AI41_NAVIGATION_SCHEMA_HASH, AI41_STRATEGIC_REWARD_HASH)
    metrics = {
        "elapsed_seconds": time.perf_counter() - started,
        "updates": updates, "labels": total_labels,
        "mean_unroll_loss": total_loss / max(updates, 1), "label_counts": label_counts,
        "output": str(args.output),
    }
    args.output.with_suffix(".clone.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report(f"AI-30 clone complete: labels={total_labels} updates={updates} loss={metrics['mean_unroll_loss']:.4f} checkpoint={args.output}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
