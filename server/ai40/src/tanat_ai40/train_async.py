from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .env import (
    AI40_ROSTER,
    AssaultVectorEnv,
    CONTROLLER_AI40,
    HERO_COUNT,
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
from .train_matches import MatchAssignment, build_schedule, policy_actor_mask


@dataclass(slots=True)
class AsyncRolloutGroup:
    index: int
    env: AssaultVectorEnv
    assignments: list[MatchAssignment | None]
    observations: list
    h: torch.Tensor
    c: torch.Tensor
    rows: list[dict] = field(default_factory=list)
    pending_row: dict | None = None
    pending_h: torch.Tensor | None = None
    pending_c: torch.Tensor | None = None
    started_at: float = 0.0

    @property
    def workers(self) -> int:
        return len(self.assignments)


@dataclass(slots=True)
class RolloutBatch:
    rows: list[dict]
    advantages: np.ndarray
    returns: np.ndarray
    rollout_seconds: float
    group_times: list[float]
    hero_steps: int
    completed: dict[str, int]
    outcomes: dict[str, int]
    behavior_update: int


def partition_workers(workers: int, group_size: int) -> list[int]:
    if workers < 1 or group_size < 1:
        raise ValueError("workers and group_size must be positive")
    return [min(group_size, workers - start) for start in range(0, workers, group_size)]


def merge_group_rows(groups: list[AsyncRolloutGroup], horizon: int) -> list[dict]:
    if any(len(group.rows) != horizon for group in groups):
        raise ValueError("every async group must finish the same rollout horizon")
    merged: list[dict] = []
    for step in range(horizon):
        merged.append({
            key: np.concatenate([group.rows[step][key] for group in groups], axis=0)
            for key in groups[0].rows[step]
        })
    return merged


def prepare_step(group: AsyncRolloutGroup, policy: AI40Policy, device: torch.device) -> np.ndarray:
    obs = stack_observations(group.observations)
    hero = as_tensor(obs.hero, device)
    entities = as_tensor(obs.entities, device)
    global_state = as_tensor(obs.global_state, device)
    entity_mask = as_tensor(obs.entity_mask, device)
    kind_mask = as_tensor(obs.kind_mask, device).bool()
    h_in, c_in = group.h.detach(), group.c.detach()
    with torch.no_grad():
        network = policy(hero, entities, global_state, entity_mask, h_in, c_in)
        kinds = masked_categorical(network["kind"], kind_mask).sample()
        target_mask = effective_target_mask(obs, kinds, device)
        dists = distributions(network, kind_mask, target_mask)
        action_tensors = (kinds, dists[1].sample(), dists[2].sample(), dists[3].sample())
        log_prob = combined_log_prob(dists, action_tensors)
    actions = torch.stack(action_tensors, dim=-1).cpu().numpy()
    group.pending_row = {
        "hero": obs.hero.copy(), "entities": obs.entities.copy(),
        "global": obs.global_state.copy(), "entity_mask": obs.entity_mask.copy(),
        "kind_mask": obs.kind_mask.copy(), "target_mask": target_mask.cpu().numpy(),
        "h": h_in.cpu().numpy(), "c": c_in.cpu().numpy(),
        "actions": actions.copy(), "log_prob": log_prob.cpu().numpy(),
        "value": network["value"].cpu().numpy(),
        "policy_mask": policy_actor_mask(group.assignments),
    }
    group.pending_h, group.pending_c = network["h"].detach(), network["c"].detach()
    return actions.reshape(group.workers, HERO_COUNT, 4)


def finish_step(
    group: AsyncRolloutGroup,
    results: list,
    schedule: deque[MatchAssignment],
    completed: Counter,
    outcomes: Counter,
    seed_rng: np.random.Generator,
    max_steps: int,
) -> int:
    row = group.pending_row
    assert row is not None and group.pending_h is not None and group.pending_c is not None
    row["reward"] = np.concatenate([result.rewards for result in results])
    row["done"] = np.concatenate([
        np.full(HERO_COUNT, result.done, np.float32) for result in results
    ])
    group.rows.append(row)
    hero_steps = int(row["policy_mask"].sum())
    group.h, group.c = group.pending_h, group.pending_c
    group.pending_row = group.pending_h = group.pending_c = None
    group.observations = results
    reset_indices: list[int] = []
    reset_seeds: list[int] = []
    reset_controllers: list[tuple[int, ...]] = []
    for index, result in enumerate(results):
        assignment = group.assignments[index]
        if assignment is None or not result.done:
            continue
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
        if schedule:
            replacement = schedule.popleft()
            group.assignments[index] = replacement
            reset_indices.append(index)
            reset_seeds.append(int(seed_rng.integers(1, 2**63 - 1)))
            reset_controllers.append(replacement.controllers)
        else:
            group.assignments[index] = None
        start, end = index * HERO_COUNT, (index + 1) * HERO_COUNT
        group.h[start:end] = 0
        group.c[start:end] = 0
    if reset_indices:
        replacements = group.env.reset_indices(
            reset_indices, reset_seeds, max_steps,
            controller_sets=reset_controllers,
            rosters=self_play_rosters(seed_rng, len(reset_indices)),
        )
        for index, replacement in replacements.items():
            group.observations[index] = replacement
    return hero_steps


def bootstrap_group(group: AsyncRolloutGroup, policy: AI40Policy, device: torch.device) -> np.ndarray:
    obs = stack_observations(group.observations)
    with torch.no_grad():
        return policy(
            as_tensor(obs.hero, device), as_tensor(obs.entities, device),
            as_tensor(obs.global_state, device), as_tensor(obs.entity_mask, device),
            group.h, group.c,
        )["value"].cpu().numpy()


def collect_grouped_rollout(
    groups: list[AsyncRolloutGroup],
    policy: AI40Policy,
    device: torch.device,
    group_pool: ThreadPoolExecutor,
    schedule: deque[MatchAssignment],
    completed: Counter,
    outcomes: Counter,
    seed_rng: np.random.Generator,
    max_steps: int,
    steps: int,
    hero_steps: int,
    behavior_update: int,
    cuda_stream: torch.cuda.Stream | None = None,
) -> RolloutBatch:
    rollout_started = time.perf_counter()
    stream_context = (torch.cuda.stream(cuda_stream)
                      if cuda_stream is not None else nullcontext())
    with stream_context:
        futures: dict[Future, AsyncRolloutGroup] = {}
        for group in groups:
            group.rows.clear()
            group.started_at = time.perf_counter()
            actions = prepare_step(group, policy, device)
            futures[group_pool.submit(group.env.step, actions)] = group
        group_times: list[float] = []
        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                group = futures.pop(future)
                results = future.result()
                hero_steps += finish_step(
                    group, results, schedule, completed, outcomes, seed_rng, max_steps,
                )
                if len(group.rows) < steps and any(x is not None for x in group.assignments):
                    actions = prepare_step(group, policy, device)
                    futures[group_pool.submit(group.env.step, actions)] = group
                else:
                    while len(group.rows) < steps:
                        padding = {
                            key: np.zeros_like(value) for key, value in group.rows[-1].items()
                        }
                        group.rows.append(padding)
                    group_times.append(time.perf_counter() - group.started_at)
        rows = merge_group_rows(groups, steps)
        bootstrap = np.concatenate([
            bootstrap_group(group, policy, device) for group in groups
        ])
    rollout_seconds = time.perf_counter() - rollout_started
    advantages, returns = gae(rows, bootstrap)
    return RolloutBatch(
        rows, advantages, returns, rollout_seconds, group_times, hero_steps,
        dict(completed), dict(outcomes), behavior_update,
    )


def train_async(
    executable: Path,
    ai40_matches: int,
    ai30_matches: int,
    steps: int,
    workers: int,
    group_size: int,
    max_steps: int,
    device: torch.device,
    output: Path,
    minibatch_size: int = 2048,
    resume: Path | None = None,
    env_gomaxprocs: int = 1,
    seed: int = 1,
    pipeline: bool = True,
) -> None:
    completed, outcomes = Counter(), Counter()
    hero_steps = update = 0
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("schema_hash") != SCHEMA_HASH.hex() or saved.get("reward_hash") != REWARD_HASH.hex():
            raise RuntimeError("checkpoint schema/reward hash mismatch")
        prior = saved.get("config", {})
        completed.update(prior.get("completed_matches", {}))
        outcomes.update(prior.get("outcomes", {}))
        hero_steps, update = int(saved.get("hero_steps", 0)), int(saved.get("update", 0))
    remaining_ai40 = ai40_matches - completed["ai40"]
    remaining_ai30 = ai30_matches - completed["ai30"]
    if remaining_ai40 < 0 or remaining_ai30 < 0:
        raise RuntimeError("checkpoint has more completed matches than requested")
    if remaining_ai40 + remaining_ai30 == 0:
        print(f"training already complete: matches={ai40_matches + ai30_matches}", flush=True)
        return
    pending = deque(build_schedule(remaining_ai40, remaining_ai30, completed["ai30"]))
    workers = min(workers, len(pending))
    sizes = partition_workers(workers, group_size)
    os.environ["GOMAXPROCS"] = str(env_gomaxprocs)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_rng = np.random.default_rng(seed)
    policy = AI40Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    if saved is not None:
        policy.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
    initial_update = update
    actor_policy = policy
    if pipeline:
        actor_policy = AI40Policy().to(device)
        actor_policy.load_state_dict(policy.state_dict())
        actor_policy.eval()
        actor_policy.requires_grad_(False)
    config = {
        "workers": workers, "rollout_groups": len(sizes), "group_size": group_size,
        "rollout_steps": steps, "max_steps": max_steps, "device": str(device),
        "training_mode": "ai40_async_grouped_mixed_matches",
        "ai40_mirror_matches": ai40_matches, "ai30_opponent_matches": ai30_matches,
        "scripted_opponent_samples_excluded": True,
        "minibatch_size": minibatch_size, "env_gomaxprocs": env_gomaxprocs,
        "seed": seed, "actor_learner_pipeline": pipeline,
        "maximum_policy_lag_updates": 1 if pipeline else 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target_completed = ai40_matches + ai30_matches
    groups: list[AsyncRolloutGroup] = []
    with ExitStack() as stack:
        seed_base = seed
        for index, size in enumerate(sizes):
            env = stack.enter_context(AssaultVectorEnv(executable, size))
            assignments = [pending.popleft() for _ in range(size)]
            observations = env.reset(
                range(seed_base, seed_base + size), max_steps=max_steps,
                controller_sets=[assignment.controllers for assignment in assignments],
                rosters=self_play_rosters(seed_rng, size),
            )
            seed_base += size
            h, c = policy.initial_state(size * HERO_COUNT, device)
            groups.append(AsyncRolloutGroup(index, env, assignments, observations, h, c))
        with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="rollout-group") as pool:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-pipeline") as pipeline_pool:
                actor_stream = torch.cuda.Stream(device=device) if pipeline and device.type == "cuda" else None
                batch = collect_grouped_rollout(
                    groups, actor_policy, device, pool, pending, completed, outcomes,
                    seed_rng, max_steps, steps, hero_steps, update, actor_stream,
                )
                while True:
                    overlap_started = time.perf_counter()
                    next_future: Future | None = None
                    if pipeline and sum(completed.values()) < target_completed:
                        next_future = pipeline_pool.submit(
                            collect_grouped_rollout,
                            groups, actor_policy, device, pool, pending, completed, outcomes,
                            seed_rng, max_steps, steps, batch.hero_steps, update, actor_stream,
                        )
                    ppo_started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    ppo_metrics = ppo_update(
                        policy, optimizer, batch.rows, batch.advantages, batch.returns, device,
                        minibatch_size=minibatch_size, return_metrics=True,
                    )
                    loss = ppo_metrics["loss"]
                    ppo_seconds = time.perf_counter() - ppo_started
                    next_batch = next_future.result() if next_future is not None else None
                    if next_batch is not None:
                        actor_policy.load_state_dict(policy.state_dict())
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    cycle_seconds = time.perf_counter() - overlap_started
                    update += 1
                    hero_steps = batch.hero_steps
                    config["completed_matches"] = batch.completed
                    config["outcomes"] = batch.outcomes
                    save_checkpoint(output / "latest.pt", policy, optimizer, update, hero_steps, config)
                    policy_lag = update - 1 - batch.behavior_update
                    trainable_steps = sum(int(row["policy_mask"].sum()) for row in batch.rows)
                    update_metrics = {
                        "time": time.time(), "update": update,
                        "matches": sum(batch.completed.values()),
                        "target_matches": target_completed,
                        "mirror_matches": batch.completed.get("ai40", 0),
                        "ai30_matches": batch.completed.get("ai30", 0),
                        "hero_steps": hero_steps, "loss": loss,
                        **ppo_metrics,
                        "rollout_seconds": batch.rollout_seconds,
                        "ppo_seconds": ppo_seconds,
                        "pipeline_cycle_seconds": cycle_seconds,
                        "pipeline_startup_rollout_seconds":
                            batch.rollout_seconds if update == initial_update + 1 else 0.0,
                        "pipeline_overlap_active": next_future is not None,
                        "behavior_update": batch.behavior_update,
                        "policy_lag_updates": policy_lag,
                        "environment_steps_per_second":
                            steps * workers / max(batch.rollout_seconds, 1e-9),
                        "trainable_hero_steps_per_second":
                            trainable_steps / max(cycle_seconds, 1e-9),
                        "group_min_seconds": min(batch.group_times),
                        "group_max_seconds": max(batch.group_times),
                        "cuda_peak_allocated_bytes":
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "cuda_peak_reserved_bytes":
                            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                        "outcomes": batch.outcomes,
                    }
                    with (output / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(update_metrics, separators=(",", ":")) + "\n")
                    print(
                        f"update={update} matches={sum(batch.completed.values())}/{target_completed} "
                        f"hero_steps={hero_steps} loss={loss:.6f} "
                        f"rollout_s={batch.rollout_seconds:.2f} ppo_s={ppo_seconds:.2f} "
                        f"env_steps_s={steps * workers / max(batch.rollout_seconds, 1e-9):.1f} "
                        f"group_min_s={min(batch.group_times):.2f} "
                        f"group_max_s={max(batch.group_times):.2f} policy_lag={policy_lag}",
                        flush=True,
                    )
                    if next_future is None:
                        if sum(completed.values()) >= target_completed:
                            break
                        actor_policy = policy
                        batch = collect_grouped_rollout(
                            groups, actor_policy, device, pool, pending, completed, outcomes,
                            seed_rng, max_steps, steps, hero_steps, update, None,
                        )
                        continue
                    batch = next_batch
    elapsed = time.perf_counter() - started
    manifest = {
        "version": "AI-40-v3", "schema_hash": SCHEMA_HASH.hex(),
        "reward_hash": REWARD_HASH.hex(), "hidden_size": policy.hidden_size,
        "roster_ids": AI40_ROSTER.tolist(), "hero_steps": hero_steps,
        "updates": update, "completed_matches": dict(completed),
        "outcomes": dict(outcomes), **config,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"training complete: matches={sum(completed.values())} hero_steps={hero_steps} "
        f"elapsed={elapsed:.1f}s checkpoint={output / 'latest.pt'}", flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--ai40-matches", type=int, default=100)
    parser.add_argument("--ai30-matches", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/async"))
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--env-gomaxprocs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-pipeline", action="store_true")
    args = parser.parse_args()
    train_async(
        args.env, args.ai40_matches, args.ai30_matches, args.steps, args.workers,
        args.group_size, args.max_steps, torch.device(args.device), args.output,
        args.minibatch_size, args.resume, args.env_gomaxprocs, args.seed,
        not args.no_pipeline,
    )


if __name__ == "__main__":
    main()
