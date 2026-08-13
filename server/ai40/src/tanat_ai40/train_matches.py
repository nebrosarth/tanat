from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .env import (
    AI40_ROSTER,
    AI40_SELF_PLAY_CONTROLLERS,
    AssaultVectorEnv,
    CONTROLLER_AI30,
    CONTROLLER_AI40,
    HERO_COUNT,
    HeroAction,
    REWARD_HASH,
    SCHEMA_HASH,
    self_play_rosters,
)
from .model import AI40Policy, masked_categorical
from .train import (
    as_tensor,
    combined_log_prob,
    distributions,
    effective_target_mask,
    gae,
    ppo_update,
    save_checkpoint,
    stack_observations,
)


@dataclass(frozen=True, slots=True)
class MatchAssignment:
    opponent: str
    controllers: tuple[int, ...]
    ai40_side: int


def build_schedule(
    ai40_matches: int, ai30_matches: int, teacher_start_index: int = 0,
) -> list[MatchAssignment]:
    """Alternate mirror and teacher matches and swap the AI-40 faction."""
    if ai40_matches < 0 or ai30_matches < 0 or ai40_matches + ai30_matches == 0:
        raise ValueError("match counts must be non-negative and have a positive sum")
    if teacher_start_index < 0:
        raise ValueError("teacher_start_index must be non-negative")
    schedule: list[MatchAssignment] = []
    mirror_left, teacher_left, teacher_index = ai40_matches, ai30_matches, teacher_start_index
    while mirror_left or teacher_left:
        if mirror_left:
            schedule.append(MatchAssignment("ai40", AI40_SELF_PLAY_CONTROLLERS, 0))
            mirror_left -= 1
        if teacher_left:
            ai40_side = 1 + teacher_index % 2
            first = CONTROLLER_AI40 if ai40_side == 1 else CONTROLLER_AI30
            second = CONTROLLER_AI30 if ai40_side == 1 else CONTROLLER_AI40
            schedule.append(MatchAssignment(
                "ai30", (first,) * (HERO_COUNT // 2) + (second,) * (HERO_COUNT // 2),
                ai40_side,
            ))
            teacher_left -= 1
            teacher_index += 1
    return schedule


def policy_actor_mask(assignments: list[MatchAssignment | None]) -> np.ndarray:
    return np.concatenate([
        np.asarray([controller == CONTROLLER_AI40 for controller in assignment.controllers], dtype=np.uint8)
        if assignment is not None else np.zeros(HERO_COUNT, dtype=np.uint8)
        for assignment in assignments
    ])


def train_matches(
    executable: Path,
    ai40_matches: int,
    ai30_matches: int,
    steps: int,
    workers: int,
    max_steps: int,
    device: torch.device,
    output: Path,
    minibatch_size: int = 2048,
    resume: Path | None = None,
    env_gomaxprocs: int = 1,
) -> None:
    completed = Counter()
    outcomes = Counter()
    hero_steps = update = 0
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved.get("schema_hash") != SCHEMA_HASH.hex() or
                saved.get("reward_hash") != REWARD_HASH.hex()):
            raise RuntimeError("checkpoint schema/reward hash mismatch")
        prior = saved.get("config", {})
        if prior.get("training_mode") != "ai40_mixed_100_matches":
            raise RuntimeError("resume checkpoint is not mixed match training")
        completed.update(prior.get("completed_matches", {}))
        outcomes.update(prior.get("outcomes", {}))
        hero_steps = int(saved.get("hero_steps", 0))
        update = int(saved.get("update", 0))
    remaining_ai40 = ai40_matches - completed["ai40"]
    remaining_ai30 = ai30_matches - completed["ai30"]
    if remaining_ai40 < 0 or remaining_ai30 < 0:
        raise RuntimeError("checkpoint has more completed matches than requested")
    if remaining_ai40 + remaining_ai30 == 0:
        print(
            f"training already complete: matches={ai40_matches + ai30_matches} "
            f"checkpoint={resume}",
            flush=True,
        )
        return
    schedule = build_schedule(remaining_ai40, remaining_ai30, completed["ai30"])
    workers = min(workers, len(schedule))
    if env_gomaxprocs < 1:
        raise ValueError("env_gomaxprocs must be positive")
    # assaultenv inherits this value. A match is predominantly serialized by
    # the authoritative instance lock, so one Go scheduler thread per worker
    # outperforms the default (all host CPUs) when many workers run together.
    os.environ["GOMAXPROCS"] = str(env_gomaxprocs)
    torch.manual_seed(1)
    np.random.seed(1)
    seed_rng = np.random.default_rng(1)
    policy = AI40Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    if saved is not None:
        policy.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
    actors = workers * HERO_COUNT
    h, c = policy.initial_state(actors, device)
    assignments: list[MatchAssignment | None] = list(schedule[:workers])
    next_assignment = workers
    config = {
        "workers": workers,
        "rollout_steps": steps,
        "max_steps": max_steps,
        "device": str(device),
        "training_mode": "ai40_mixed_100_matches",
        "ai40_mirror_matches": ai40_matches,
        "ai30_opponent_matches": ai30_matches,
        "scripted_opponent_samples_excluded": True,
        "minibatch_size": minibatch_size,
        "env_gomaxprocs": env_gomaxprocs,
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initial_rosters = self_play_rosters(seed_rng, workers)
    with AssaultVectorEnv(executable, workers) as env:
        observations = env.reset(
            range(1, workers + 1), max_steps=max_steps,
            controller_sets=[assignment.controllers for assignment in assignments if assignment],
            rosters=initial_rosters,
        )
        target_completed = ai40_matches + ai30_matches
        while sum(completed.values()) < target_completed:
            rollout_started = time.perf_counter()
            rows = []
            for _ in range(steps):
                obs = stack_observations(observations)
                hero = as_tensor(obs.hero, device)
                entities = as_tensor(obs.entities, device)
                global_state = as_tensor(obs.global_state, device)
                entity_mask = as_tensor(obs.entity_mask, device)
                kind_mask = as_tensor(obs.kind_mask, device).bool()
                h_in, c_in = h.detach(), c.detach()
                with torch.no_grad():
                    network = policy(hero, entities, global_state, entity_mask, h_in, c_in)
                    kinds = masked_categorical(network["kind"], kind_mask).sample()
                    target_mask = effective_target_mask(obs, kinds, device)
                    dists = distributions(network, kind_mask, target_mask)
                    action_tensors = (
                        kinds, dists[1].sample(), dists[2].sample(), dists[3].sample(),
                    )
                    log_prob = combined_log_prob(dists, action_tensors)
                # Copy the factorized actions in one CUDA synchronization.
                # Scalar int(cuda_tensor[index]) calls made rollout collection
                # CPU/GPU-latency-bound and left both processors underused.
                action_values = torch.stack(action_tensors, dim=-1).cpu().numpy()
                results = env.step(action_values.reshape(workers, HERO_COUNT, 4))
                rewards = np.concatenate([result.rewards for result in results])
                done = np.concatenate([
                    np.full(HERO_COUNT, result.done, np.float32) for result in results
                ])
                current_mask = policy_actor_mask(assignments)
                rows.append({
                    "hero": obs.hero.copy(), "entities": obs.entities.copy(),
                    "global": obs.global_state.copy(), "entity_mask": obs.entity_mask.copy(),
                    "kind_mask": obs.kind_mask.copy(), "target_mask": target_mask.cpu().numpy(),
                    "h": h_in.cpu().numpy(), "c": c_in.cpu().numpy(),
                    "actions": np.stack([value.cpu().numpy() for value in action_tensors], axis=-1),
                    "log_prob": log_prob.cpu().numpy(), "value": network["value"].cpu().numpy(),
                    "reward": rewards, "done": done, "policy_mask": current_mask,
                })
                hero_steps += int(current_mask.sum())
                h, c = network["h"].detach(), network["c"].detach()
                observations = results
                finished_indices = [
                    index for index, result in enumerate(results)
                    if assignments[index] is not None and result.done
                ]
                reset_indices = []
                reset_seeds = []
                reset_controllers = []
                for index in finished_indices:
                    assignment = assignments[index]
                    assert assignment is not None
                    result = results[index]
                    completed[assignment.opponent] += 1
                    if assignment.opponent == "ai30":
                        if result.winner == 0:
                            outcomes["ai30_draw"] += 1
                        elif result.winner == assignment.ai40_side:
                            outcomes["ai40_over_ai30"] += 1
                        else:
                            outcomes["ai30_over_ai40"] += 1
                    else:
                        outcomes[f"mirror_winner_{result.winner}"] += 1
                    if next_assignment < len(schedule):
                        replacement = schedule[next_assignment]
                        next_assignment += 1
                        assignments[index] = replacement
                        reset_indices.append(index)
                        reset_seeds.append(int(seed_rng.integers(1, 2**63 - 1)))
                        reset_controllers.append(replacement.controllers)
                    else:
                        assignments[index] = None
                    start, end = index * HERO_COUNT, (index + 1) * HERO_COUNT
                    h[start:end] = 0
                    c[start:end] = 0
                if reset_indices:
                    replacements = env.reset_indices(
                        reset_indices, reset_seeds, max_steps,
                        controller_sets=reset_controllers,
                        rosters=self_play_rosters(seed_rng, len(reset_indices)),
                    )
                    for index, replacement in replacements.items():
                        observations[index] = replacement
                if finished_indices:
                    print(
                        f"matches={sum(completed.values())}/{target_completed} "
                        f"mirror={completed['ai40']}/{ai40_matches} "
                        f"vs_ai30={completed['ai30']}/{ai30_matches}",
                        flush=True,
                    )
                if all(assignment is None for assignment in assignments):
                    break
            rollout_seconds = time.perf_counter() - rollout_started
            batch = stack_observations(observations)
            with torch.no_grad():
                bootstrap = policy(
                    as_tensor(batch.hero, device), as_tensor(batch.entities, device),
                    as_tensor(batch.global_state, device),
                    as_tensor(batch.entity_mask, device), h, c,
                )["value"].cpu().numpy()
            advantages, returns = gae(rows, bootstrap)
            ppo_started = time.perf_counter()
            loss = ppo_update(
                policy, optimizer, rows, advantages, returns, device,
                minibatch_size=minibatch_size,
            )
            ppo_seconds = time.perf_counter() - ppo_started
            update += 1
            config["completed_matches"] = dict(completed)
            config["outcomes"] = dict(outcomes)
            save_checkpoint(output / "latest.pt", policy, optimizer, update, hero_steps, config)
            print(
                f"update={update} hero_steps={hero_steps} loss={loss:.6f} "
                f"rollout_s={rollout_seconds:.2f} ppo_s={ppo_seconds:.2f} "
                f"env_steps_s={len(rows) * workers / max(rollout_seconds, 1e-9):.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    manifest = {
        "version": "AI-40-v3", "schema_hash": SCHEMA_HASH.hex(),
        "reward_hash": REWARD_HASH.hex(), "hidden_size": policy.hidden_size,
        "roster_ids": AI40_ROSTER.tolist(), "hero_steps": hero_steps,
        "updates": update, "completed_matches": dict(completed),
        "outcomes": dict(outcomes), **config,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(
        f"training complete: matches={sum(completed.values())} "
        f"hero_steps={hero_steps} elapsed={elapsed:.1f}s checkpoint={output / 'latest.pt'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--ai40-matches", type=int, default=50)
    parser.add_argument("--ai30-matches", type=int, default=50)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/mixed-100"))
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--env-gomaxprocs", type=int, default=1)
    args = parser.parse_args()
    train_matches(
        args.env, args.ai40_matches, args.ai30_matches, args.steps, args.workers,
        args.max_steps, torch.device(args.device), args.output,
        args.minibatch_size, args.resume, args.env_gomaxprocs,
    )


if __name__ == "__main__":
    main()
