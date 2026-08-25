from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

from .env import (
    AI40_ROSTER,
    AssaultVectorEnv,
    CONTROLLER_AI40,
    HERO_COUNT,
    REWARD_HASH,
    SCHEMA_HASH,
    PROTOCOL_VERSION,
    self_play_rosters,
)
from .model import AI40Policy, PolicyRunner, masked_categorical
from .train import (
    as_tensor,
    combined_log_prob,
    distributions,
    gae,
    horizon_discounts,
    policy_forward,
    ppo_update,
    save_checkpoint,
    stack_observations,
)
from .train_matches import (
    MatchAssignment,
    build_schedule,
    historical_actor_indices,
    policy_actor_mask,
)


def report(message: str) -> None:
    """Write live PPO progress to the interactive error stream."""
    print(message, file=sys.stderr, flush=True)


def _effective_target_mask_tensors(
    target_mask: torch.Tensor,
    skill_target_mask: torch.Tensor,
    kinds: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    selected = skill_target_mask[rows, torch.clamp(kinds - 3, 0, 3)]
    skill_rows = (kinds >= 3) & (kinds <= 6)
    mask = torch.where(skill_rows.unsqueeze(1), selected, target_mask)
    empty = ~mask.any(dim=1)
    first = torch.arange(mask.shape[1], device=mask.device) == 0
    return mask | (empty.unsqueeze(1) & first.unsqueeze(0))


def _sample_actions_conditioned(
    kind_logits: torch.Tensor,
    target_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    distance_logits: torch.Tensor,
    kind_mask: torch.Tensor,
    target_mask: torch.Tensor,
    skill_target_mask: torch.Tensor,
    kind_noise: torch.Tensor,
    target_noise: torch.Tensor,
    direction_noise: torch.Tensor,
    distance_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample AI-41's action-conditioned heads using external Exp(1) noise."""
    masked_kind = kind_logits.masked_fill(~kind_mask, -1e9)
    kinds = torch.argmax(masked_kind - torch.log(kind_noise), dim=1)
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    selected_target = target_logits[rows, kinds]
    selected_direction = direction_logits[rows, kinds]
    selected_distance = distance_logits[rows, kinds]
    effective_mask = _effective_target_mask_tensors(
        target_mask, skill_target_mask, kinds,
    )
    masked_target = selected_target.masked_fill(~effective_mask, -1e9)
    targets = torch.argmax(masked_target - torch.log(target_noise), dim=1)
    directions = torch.argmax(selected_direction - torch.log(direction_noise), dim=1)
    distances = torch.argmax(selected_distance - torch.log(distance_noise), dim=1)
    target_active = ((kinds >= 2) & (kinds <= 6)).float()
    position_active = ((kinds == 1) | ((kinds >= 3) & (kinds <= 6))).float()
    distance_active = position_active
    if direction_logits.shape[-1] == 81:
        position_active = (((kinds == 1) & (distances == 0)) |
                           ((kinds >= 3) & (kinds <= 6))).float()
        distance_active = (kinds == 1).float()
    log_prob = (
        torch.log_softmax(masked_kind, dim=1)[rows, kinds]
        + torch.log_softmax(masked_target, dim=1)[rows, targets] * target_active
        + torch.log_softmax(selected_direction, dim=1)[rows, directions] * position_active
        + torch.log_softmax(selected_distance, dim=1)[rows, distances] * distance_active
    )
    actions = torch.stack((kinds, targets, directions, distances), dim=1)
    return actions, effective_mask, log_prob


def _sample_actions_unconditioned(
    kind_logits: torch.Tensor,
    target_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    distance_logits: torch.Tensor,
    kind_mask: torch.Tensor,
    target_mask: torch.Tensor,
    skill_target_mask: torch.Tensor,
    kind_noise: torch.Tensor,
    target_noise: torch.Tensor,
    direction_noise: torch.Tensor,
    distance_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample legacy AI-40 heads with the same external-noise method."""
    masked_kind = kind_logits.masked_fill(~kind_mask, -1e9)
    kinds = torch.argmax(masked_kind - torch.log(kind_noise), dim=1)
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    effective_mask = _effective_target_mask_tensors(
        target_mask, skill_target_mask, kinds,
    )
    masked_target = target_logits.masked_fill(~effective_mask, -1e9)
    targets = torch.argmax(masked_target - torch.log(target_noise), dim=1)
    directions = torch.argmax(direction_logits - torch.log(direction_noise), dim=1)
    distances = torch.argmax(distance_logits - torch.log(distance_noise), dim=1)
    target_active = ((kinds >= 2) & (kinds <= 6)).float()
    position_active = ((kinds == 1) | ((kinds >= 3) & (kinds <= 6))).float()
    distance_active = position_active
    if direction_logits.shape[-1] == 81:
        position_active = (((kinds == 1) & (distances == 0)) |
                           ((kinds >= 3) & (kinds <= 6))).float()
        distance_active = (kinds == 1).float()
    log_prob = (
        torch.log_softmax(masked_kind, dim=1)[rows, kinds]
        + torch.log_softmax(masked_target, dim=1)[rows, targets] * target_active
        + torch.log_softmax(direction_logits, dim=1)[rows, directions] * position_active
        + torch.log_softmax(distance_logits, dim=1)[rows, distances] * distance_active
    )
    actions = torch.stack((kinds, targets, directions, distances), dim=1)
    return actions, effective_mask, log_prob


class ActionSampler:
    """Fast categorical sampler whose randomness remains outside CUDA graphs."""

    def __init__(self, compile_sampler: bool = False):
        self.compile_sampler = compile_sampler and hasattr(torch, "compile")
        self.compiled: dict[bool, object] = {}
        self.noise: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def _noise_tensors(self, logits: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        key = tuple((tuple(value.shape), value.dtype, value.device) for value in logits)
        noise = self.noise.get(key)
        if noise is None:
            noise = tuple(torch.empty_like(value) for value in logits)
            self.noise[key] = noise
        # Exponential-race sampling is exactly categorical when each factor is
        # Exp(1). Updating these inputs outside the graph avoids captured RNG.
        for value in noise:
            value.exponential_()
        return noise

    def __call__(
        self,
        network: dict[str, torch.Tensor],
        kind_mask: torch.Tensor,
        target_mask: torch.Tensor,
        skill_target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioned = network["target"].ndim == 3
        if conditioned:
            selected_shapes = (
                network["kind"], network["target"][:, 0],
                network["direction"][:, 0], network["distance"][:, 0],
            )
            function = _sample_actions_conditioned
        else:
            selected_shapes = (
                network["kind"], network["target"],
                network["direction"], network["distance"],
            )
            function = _sample_actions_unconditioned
        noise = self._noise_tensors(selected_shapes)
        arguments = (
            network["kind"], network["target"], network["direction"],
            network["distance"], kind_mask, target_mask, skill_target_mask, *noise,
        )
        if self.compile_sampler:
            try:
                compiled = self.compiled.get(conditioned)
                if compiled is None:
                    compiled = torch.compile(
                        function, mode="reduce-overhead", dynamic=False, fullgraph=True,
                    )
                    self.compiled[conditioned] = compiled
                return compiled(*arguments)
            except Exception as exc:  # pragma: no cover - backend/platform specific
                warnings.warn(
                    f"compiled actor sampling failed; using eager sampling: {exc}",
                )
                self.compile_sampler = False
                self.compiled.clear()
        return function(*arguments)

    @property
    def active(self) -> bool:
        return self.compile_sampler


@dataclass(slots=True)
class AsyncRolloutGroup:
    index: int
    env: AssaultVectorEnv
    assignments: list[MatchAssignment | None]
    observations: list
    h: torch.Tensor
    c: torch.Tensor
    historical_h: torch.Tensor | None = None
    historical_c: torch.Tensor | None = None
    rows: list[dict] = field(default_factory=list)
    pending_row: dict | None = None
    pending_h: torch.Tensor | None = None
    pending_c: torch.Tensor | None = None
    pending_historical_h: torch.Tensor | None = None
    pending_historical_c: torch.Tensor | None = None
    transfer_device: torch.Tensor | None = None
    transfer_host: torch.Tensor | None = None
    # Persistent CUDA input tensors.  The Go protocol already provides each
    # observation field as a contiguous NumPy view; copying it into a stable
    # device buffer avoids eight new CUDA allocations and conversions per actor
    # tick, which otherwise dominated the very small recurrent forward pass.
    actor_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    started_at: float = 0.0
    actor_seconds: float = 0.0
    environment_seconds: float = 0.0

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
    actor_times: list[float]
    environment_times: list[float]


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


def actor_observation_tensors(
    group: AsyncRolloutGroup,
    obs,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Copy one protocol observation batch into reusable actor device buffers."""
    fields = (
        ("hero", None),
        ("entities", None),
        ("global_state", None),
        ("entity_mask", torch.bool),
        ("kind_mask", torch.bool),
        ("target_mask", torch.bool),
        ("skill_target_mask", torch.bool),
    )
    if hasattr(obs, "abilities"):
        fields += (("abilities", None),)
    buffers = group.actor_inputs
    for name, requested_dtype in fields:
        array = getattr(obs, name)
        source = torch.from_numpy(array)
        dtype = requested_dtype or source.dtype
        target = buffers.get(name)
        if target is None or target.shape != source.shape or target.dtype != dtype:
            target = torch.empty(source.shape, dtype=dtype, device=device)
            buffers[name] = target
        # copy_ performs the uint8->bool conversion for masks directly into the
        # durable tensor.  There is no intermediate GPU uint8 allocation.
        target.copy_(source, non_blocking=False)
    return buffers


def prepare_step(
    group: AsyncRolloutGroup,
    policy,
    device: torch.device,
    mixed_precision: bool = False,
    action_sampler: ActionSampler | None = None,
    historical_policies: dict[str, AI40Policy] | None = None,
) -> np.ndarray:
    obs = stack_observations(group.observations)
    inputs = actor_observation_tensors(group, obs, device)
    hero = inputs["hero"]
    entities = inputs["entities"]
    global_state = inputs["global_state"]
    entity_mask = inputs["entity_mask"]
    kind_mask = inputs["kind_mask"]
    base_target_mask = inputs["target_mask"]
    skill_target_mask = inputs["skill_target_mask"]
    abilities = inputs.get("abilities")
    h_in, c_in = group.h, group.c
    # Keep recurrent state as ordinary non-grad tensors: finish_step resets
    # individual completed matches in-place between actor calls.  inference_mode
    # would make that legal state update fail outside its context.
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=mixed_precision and device.type == "cuda",
    ):
        network = policy_forward(
            policy, hero, entities, global_state, entity_mask, h_in, c_in,
            abilities,
        )
        # A second compiled CUDA graph (the sampler) may reuse model graph
        # outputs that it does not consume. Preserve recurrent state and value
        # before entering that graph; the logits are direct sampler inputs.
        next_h = network["h"].detach().clone()
        next_c = network["c"].detach().clone()
        actor_value = network["value"].float().clone()
        distribution_network = {
            key: value.float() if key in ("kind", "target", "direction", "distance") else value
            for key, value in network.items()
        }
        if action_sampler is not None:
            actions_device, target_mask, log_prob = action_sampler(
                distribution_network, kind_mask, base_target_mask, skill_target_mask,
            )
        else:
            kinds = masked_categorical(distribution_network["kind"], kind_mask).sample()
            target_mask = _effective_target_mask_tensors(
                base_target_mask, skill_target_mask, kinds,
            )
            dists = distributions(distribution_network, kind_mask, target_mask, kinds)
            action_tensors = (kinds, dists[1].sample(), dists[2].sample(), dists[3].sample())
            log_prob = combined_log_prob(dists, action_tensors)
            actions_device = torch.stack(action_tensors, dim=-1)
        historical_groups = historical_actor_indices(group.assignments)
        if historical_groups:
            if not historical_policies or group.historical_h is None or group.historical_c is None:
                raise RuntimeError("historical match has no loaded frozen policy state")
            actions_device = actions_device.clone()
            next_historical_h = group.historical_h.clone()
            next_historical_c = group.historical_c.clone()
            for historical_id, raw_indices in historical_groups.items():
                frozen = historical_policies.get(historical_id)
                if frozen is None:
                    raise RuntimeError(f"historical policy is not loaded: {historical_id}")
                indices = torch.as_tensor(raw_indices, device=device)
                frozen_output = policy_forward(
                    frozen, hero[indices], entities[indices], global_state[indices],
                    entity_mask[indices], group.historical_h[indices],
                    group.historical_c[indices],
                    abilities[indices] if abilities is not None else None,
                )
                frozen_network = {
                    key: value.float() if key in ("kind", "target", "direction", "distance") else value
                    for key, value in frozen_output.items()
                }
                frozen_kinds = masked_categorical(
                    frozen_network["kind"], kind_mask[indices],
                ).sample()
                frozen_target_mask = _effective_target_mask_tensors(
                    base_target_mask[indices], skill_target_mask[indices], frozen_kinds,
                )
                frozen_dists = distributions(
                    frozen_network, kind_mask[indices], frozen_target_mask, frozen_kinds,
                )
                frozen_actions = torch.stack((
                    frozen_kinds, frozen_dists[1].sample(), frozen_dists[2].sample(),
                    frozen_dists[3].sample(),
                ), dim=-1)
                actions_device[indices] = frozen_actions
                # The mixed BF16 actor returns recurrent state in BF16, while
                # this shared buffer is deliberately FP32: it is reset and
                # selectively updated for only the historical-policy slots.
                # Keep that durable buffer type instead of letting indexed
                # assignment fail on the first historical self-play rollout.
                next_historical_h[indices] = frozen_output["h"].to(next_historical_h.dtype)
                next_historical_c[indices] = frozen_output["c"].to(next_historical_c.dtype)
        else:
            next_historical_h = group.historical_h
            next_historical_c = group.historical_c
    actors = actions_device.shape[0]
    actions_end = 4
    target_end = actions_end + target_mask.shape[1]
    h_end = target_end + h_in.shape[1]
    c_end = h_end + c_in.shape[1]
    log_prob_index, value_index = c_end, c_end + 1
    transfer_width = value_index + 1
    if (group.transfer_device is None or
            group.transfer_device.shape != (actors, transfer_width)):
        group.transfer_device = torch.empty(
            (actors, transfer_width), dtype=torch.float32, device=device,
        )
        group.transfer_host = torch.empty(
            (actors, transfer_width), dtype=torch.float32,
            pin_memory=device.type == "cuda",
        )
    packed_device = group.transfer_device
    packed_host = group.transfer_host
    assert packed_host is not None
    packed_device[:, :actions_end].copy_(actions_device)
    packed_device[:, actions_end:target_end].copy_(target_mask)
    packed_device[:, target_end:h_end].copy_(h_in)
    packed_device[:, h_end:c_end].copy_(c_in)
    packed_device[:, log_prob_index].copy_(log_prob)
    packed_device[:, value_index].copy_(actor_value)
    # All actor outputs cross the PCIe boundary in one asynchronous copy and
    # one synchronization. The pinned staging allocation is reused every tick.
    packed_host.copy_(packed_device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    packed = packed_host.numpy()
    actions = packed[:, :actions_end].astype(np.int16, copy=True)
    group.pending_row = {
        "hero": obs.hero.copy(), "entities": obs.entities.copy(),
        "global": obs.global_state.copy(), "entity_mask": obs.entity_mask.copy(),
        "kind_mask": obs.kind_mask.copy(),
        "target_mask": packed[:, actions_end:target_end].astype(np.uint8, copy=True),
        "h": packed[:, target_end:h_end].copy(),
        "c": packed[:, h_end:c_end].copy(),
        "actions": actions,
        "log_prob": packed[:, log_prob_index].copy(),
        "value": packed[:, value_index].copy(),
        "policy_mask": policy_actor_mask(group.assignments),
    }
    if hasattr(obs, "abilities"):
        group.pending_row["abilities"] = obs.abilities.copy()
    if hasattr(obs, "teacher_actions") and getattr(obs, "teacher_actions") is not None:
        raw_teacher = obs.teacher_actions
        group.pending_row["teacher_actions"] = np.stack(
            (raw_teacher["kind"], raw_teacher["target"], raw_teacher["direction"], raw_teacher["distance"]),
            axis=1,
        ).astype(np.int16, copy=False)
        group.pending_row["teacher_valid"] = obs.teacher_valid.copy()
    group.pending_h = next_h
    group.pending_c = next_c
    group.pending_historical_h = next_historical_h
    group.pending_historical_c = next_historical_c
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
    if group.pending_historical_h is not None:
        group.historical_h = group.pending_historical_h
        group.historical_c = group.pending_historical_c
    group.pending_row = group.pending_h = group.pending_c = None
    group.pending_historical_h = group.pending_historical_c = None
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
        elif assignment.opponent == "historical":
            if result.winner == 0:
                outcomes["historical_draw"] += 1
            elif result.winner == assignment.ai40_side:
                outcomes["ai40_over_historical"] += 1
            else:
                outcomes["historical_over_ai40"] += 1
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
        if group.historical_h is not None and group.historical_c is not None:
            group.historical_h[start:end] = 0
            group.historical_c[start:end] = 0
    if reset_indices:
        replacements = group.env.reset_indices(
            reset_indices, reset_seeds, max_steps,
            controller_sets=reset_controllers,
            rosters=self_play_rosters(seed_rng, len(reset_indices)),
        )
        for index, replacement in replacements.items():
            group.observations[index] = replacement
    return hero_steps


def bootstrap_group(
    group: AsyncRolloutGroup, policy, device: torch.device, mixed_precision: bool = False,
) -> np.ndarray:
    obs = stack_observations(group.observations)
    inputs = actor_observation_tensors(group, obs, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=mixed_precision and device.type == "cuda",
    ):
        return policy_forward(
            policy, inputs["hero"], inputs["entities"],
            inputs["global_state"], inputs["entity_mask"],
            group.h, group.c,
            inputs.get("abilities"),
        )["value"].float().cpu().numpy()


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
    mixed_precision: bool = False,
    action_sampler: ActionSampler | None = None,
    historical_policies: dict[str, AI40Policy] | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutBatch:
    rollout_started = time.perf_counter()
    stream_context = (torch.cuda.stream(cuda_stream)
                      if cuda_stream is not None else nullcontext())
    with stream_context:
        futures: dict[Future, tuple[AsyncRolloutGroup, float]] = {}
        for group in groups:
            group.rows.clear()
            group.started_at = time.perf_counter()
            group.actor_seconds = group.environment_seconds = 0.0
            actor_started = time.perf_counter()
            actions = prepare_step(
                group, policy, device, mixed_precision, action_sampler,
                historical_policies,
            )
            group.actor_seconds += time.perf_counter() - actor_started
            submitted = time.perf_counter()
            futures[group_pool.submit(group.env.step, actions)] = (group, submitted)
        group_times: list[float] = []
        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                group, submitted = futures.pop(future)
                results = future.result()
                group.environment_seconds += time.perf_counter() - submitted
                hero_steps += finish_step(
                    group, results, schedule, completed, outcomes, seed_rng, max_steps,
                )
                if len(group.rows) < steps and any(x is not None for x in group.assignments):
                    actor_started = time.perf_counter()
                    actions = prepare_step(
                        group, policy, device, mixed_precision, action_sampler,
                        historical_policies,
                    )
                    group.actor_seconds += time.perf_counter() - actor_started
                    submitted = time.perf_counter()
                    futures[group_pool.submit(group.env.step, actions)] = (group, submitted)
                else:
                    while len(group.rows) < steps:
                        padding = {
                            key: np.zeros_like(value) for key, value in group.rows[-1].items()
                        }
                        group.rows.append(padding)
                    group_times.append(time.perf_counter() - group.started_at)
        # GAE needs temporal order only within a rollout group. Keeping rows in
        # group-major order avoids copying the complete (mostly entity tensor)
        # rollout once here and a second time when PPO flattens it.
        rows = [row for group in groups for row in group.rows]
        estimates = [
            gae(
                group.rows, bootstrap_group(group, policy, device, mixed_precision),
                gamma=gamma, lam=gae_lambda,
            )
            for group in groups
        ]
    rollout_seconds = time.perf_counter() - rollout_started
    advantages = np.concatenate([value[0].reshape(-1) for value in estimates])
    returns = np.concatenate([value[1].reshape(-1) for value in estimates])
    return RolloutBatch(
        rows, advantages, returns, rollout_seconds, group_times, hero_steps,
        dict(completed), dict(outcomes), behavior_update,
        [group.actor_seconds for group in groups],
        [group.environment_seconds for group in groups],
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
    minibatch_size: int = 8192,
    resume: Path | None = None,
    env_gomaxprocs: int = 16,
    seed: int = 1,
    pipeline: bool = True,
    bf16: bool = True,
    compile_model: bool = True,
    policy_factory=AI40Policy,
    schema_hash: bytes = SCHEMA_HASH,
    reward_hash: bytes = REWARD_HASH,
    protocol_version: int = PROTOCOL_VERSION,
    model_version: str = "AI-40-v3",
    compile_actor: bool | None = None,
    historical_matches: int = 0,
    historical_opponents: dict[str, Path] | None = None,
    discount_horizon_seconds: float = 19.8998324946844,
    gae_horizon_seconds: float = 3.26032220809386,
    checkpoint_name: str = "latest.pt",
    learning_rate: float = 3e-4,
    ppo_epochs: int = 3,
    target_kl: float | None = None,
    teacher_loss_weight: float = 0.0,
) -> None:
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a plain file name")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if ppo_epochs < 1:
        raise ValueError("ppo_epochs must be positive")
    if target_kl is not None and target_kl <= 0:
        raise ValueError("target_kl must be positive when enabled")
    if teacher_loss_weight < 0:
        raise ValueError("teacher_loss_weight cannot be negative")
    completed, outcomes = Counter(), Counter()
    hero_steps = update = 0
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("schema_hash") != schema_hash.hex() or saved.get("reward_hash") != reward_hash.hex():
            raise RuntimeError("checkpoint schema/reward hash mismatch")
        prior = saved.get("config", {})
        completed.update(prior.get("completed_matches", {}))
        outcomes.update(prior.get("outcomes", {}))
        hero_steps, update = int(saved.get("hero_steps", 0)), int(saved.get("update", 0))
    remaining_ai40 = ai40_matches - completed["ai40"]
    remaining_ai30 = ai30_matches - completed["ai30"]
    remaining_historical = historical_matches - completed["historical"]
    if min(remaining_ai40, remaining_ai30, remaining_historical) < 0:
        raise RuntimeError("checkpoint has more completed matches than requested")
    if remaining_ai40 + remaining_ai30 + remaining_historical == 0:
        report(f"training already complete: matches={ai40_matches + ai30_matches + historical_matches}")
        return
    historical_opponents = historical_opponents or {}
    pending = deque(build_schedule(
        remaining_ai40, remaining_ai30, completed["ai30"], remaining_historical,
        tuple(historical_opponents),
    ))
    workers = min(workers, len(pending))
    sizes = partition_workers(workers, group_size)
    os.environ["GOMAXPROCS"] = str(env_gomaxprocs)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_rng = np.random.default_rng(seed)
    policy = policy_factory().to(device)
    navigation_actions = bool(getattr(policy, "navigation_actions", False))
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if saved is not None:
        policy.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        # A resumed campaign may deliberately use a safer schedule than its
        # bootstrap checkpoint.  Optimizer state restores its saved param-group
        # learning rate, so override it explicitly after loading.
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
    frozen_policies: dict[str, AI40Policy] = {}
    for historical_id, checkpoint in historical_opponents.items():
        frozen_saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if frozen_saved.get("schema_hash") != schema_hash.hex():
            raise RuntimeError(f"historical checkpoint schema mismatch: {checkpoint}")
        frozen = policy_factory().to(device)
        frozen.load_state_dict(frozen_saved["model"])
        frozen.eval()
        frozen.requires_grad_(False)
        frozen_policies[historical_id] = frozen
    gamma, gae_lambda = horizon_discounts(
        0.2, discount_horizon_seconds, gae_horizon_seconds,
    )
    use_bf16 = bool(
        bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    learner_forward = PolicyRunner(policy, compile_model)
    initial_update = update
    actor_policy = policy
    if pipeline:
        actor_policy = policy_factory().to(device)
        actor_policy.load_state_dict(policy.state_dict())
        actor_policy.eval()
        actor_policy.requires_grad_(False)
    # Actor batches have fixed shapes inside a campaign stage, so CUDA graphs
    # amortize well across the hundreds of rollout ticks.
    actor_compile_enabled = compile_model if compile_actor is None else compile_actor
    actor_forward = PolicyRunner(
        actor_policy,
        actor_compile_enabled and device.type == "cuda",
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
    action_sampler = ActionSampler(
        actor_compile_enabled and device.type == "cuda",
    )
    config = {
        "workers": workers, "rollout_groups": len(sizes), "group_size": group_size,
        "rollout_steps": steps, "max_steps": max_steps, "device": str(device),
        "training_mode": f"{model_version.lower()}_async_grouped_mixed_matches",
        "ai40_mirror_matches": ai40_matches, "ai30_opponent_matches": ai30_matches,
        "historical_opponent_matches": historical_matches,
        "historical_opponents": {
            key: str(value.resolve()) for key, value in historical_opponents.items()
        },
        "scripted_opponent_samples_excluded": True,
        "minibatch_size": minibatch_size, "env_gomaxprocs": env_gomaxprocs,
        "learning_rate": learning_rate, "ppo_epochs": ppo_epochs,
        "target_kl": target_kl,
        "teacher_loss_weight": teacher_loss_weight,
        "seed": seed, "actor_learner_pipeline": pipeline,
        "model_version": model_version,
        "maximum_policy_lag_updates": 1 if pipeline else 0,
        "bf16": use_bf16, "torch_compile_ppo": compile_model,
        "torch_compile_actor": actor_forward.active,
        "torch_compile_actor_sampler": action_sampler.active,
        "navigation_actions": navigation_actions,
        "discount_horizon_seconds": discount_horizon_seconds,
        "gae_horizon_seconds": gae_horizon_seconds,
        "gamma": gamma, "gae_lambda": gae_lambda,
        "checkpoint_name": checkpoint_name,
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target_completed = ai40_matches + ai30_matches + historical_matches
    groups: list[AsyncRolloutGroup] = []
    with ExitStack() as stack:
        seed_base = seed
        for index, size in enumerate(sizes):
            env = stack.enter_context(AssaultVectorEnv(executable, size, protocol_version))
            assignments = [pending.popleft() for _ in range(size)]
            observations = env.reset(
                range(seed_base, seed_base + size), max_steps=max_steps,
                controller_sets=[assignment.controllers for assignment in assignments],
                rosters=self_play_rosters(seed_rng, size),
            )
            seed_base += size
            h, c = policy.initial_state(size * HERO_COUNT, device)
            historical_h, historical_c = policy.initial_state(size * HERO_COUNT, device)
            groups.append(AsyncRolloutGroup(
                index, env, assignments, observations, h, c, historical_h, historical_c,
            ))
        with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="rollout-group") as pool:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-pipeline") as pipeline_pool:
                actor_stream = torch.cuda.Stream(device=device) if pipeline and device.type == "cuda" else None
                batch = collect_grouped_rollout(
                    groups, actor_forward, device, pool, pending, completed, outcomes,
                    seed_rng, max_steps, steps, hero_steps, update, actor_stream,
                    use_bf16, action_sampler, frozen_policies, gamma, gae_lambda,
                )
                while True:
                    overlap_started = time.perf_counter()
                    next_future: Future | None = None
                    if pipeline and sum(completed.values()) < target_completed:
                        next_future = pipeline_pool.submit(
                            collect_grouped_rollout,
                            groups, actor_forward, device, pool, pending, completed, outcomes,
                            seed_rng, max_steps, steps, batch.hero_steps, update, actor_stream,
                            use_bf16, action_sampler, frozen_policies, gamma, gae_lambda,
                        )
                    ppo_started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    ppo_metrics = ppo_update(
                        policy, optimizer, batch.rows, batch.advantages, batch.returns, device,
                        minibatch_size=minibatch_size, return_metrics=True,
                        mixed_precision=use_bf16, forward_policy=learner_forward,
                        epochs=ppo_epochs, target_kl=target_kl,
                        teacher_loss_weight=teacher_loss_weight,
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
                    save_checkpoint(
                        output / checkpoint_name, policy, optimizer, update, hero_steps, config,
                        schema_hash, reward_hash,
                    )
                    policy_lag = update - 1 - batch.behavior_update
                    trainable_steps = sum(int(row["policy_mask"].sum()) for row in batch.rows)
                    lane_active_steps = sum(int(np.count_nonzero(
                        (row["global"][:, 8] > 0.5) & row["policy_mask"],
                    )) for row in batch.rows)
                    wrong_lane_steps = sum(int(np.count_nonzero(
                        (row["global"][:, 12] > 0.5) &
                        (row["global"][:, 8] > 0.5) & row["policy_mask"],
                    )) for row in batch.rows)
                    move_steps = sum(int(np.count_nonzero(
                        (row["actions"][:, 0] == 1) & row["policy_mask"],
                    )) for row in batch.rows)
                    global_anchor_steps = (sum(int(np.count_nonzero(
                        (row["actions"][:, 0] == 1) &
                        (row["actions"][:, 3] > 0) & row["policy_mask"],
                    )) for row in batch.rows) if navigation_actions else 0)
                    update_metrics = {
                        "time": time.time(), "update": update,
                        "matches": sum(batch.completed.values()),
                        "target_matches": target_completed,
                        "mirror_matches": batch.completed.get("ai40", 0),
                        "ai30_matches": batch.completed.get("ai30", 0),
                        "historical_matches": batch.completed.get("historical", 0),
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
                        "wrong_lane_steps": wrong_lane_steps,
                        "lane_assignment_active_steps": lane_active_steps,
                        "wrong_lane_rate":
                            wrong_lane_steps / max(lane_active_steps, 1),
                        "global_anchor_steps": global_anchor_steps,
                        "move_steps": move_steps,
                        "global_anchor_move_rate":
                            global_anchor_steps / max(move_steps, 1),
                        "group_min_seconds": min(batch.group_times),
                        "group_max_seconds": max(batch.group_times),
                        "actor_inference_seconds": max(batch.actor_times),
                        "environment_step_seconds": max(batch.environment_times),
                        "cuda_peak_allocated_bytes":
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "cuda_peak_reserved_bytes":
                            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                        "outcomes": batch.outcomes,
                    }
                    with (output / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(update_metrics, separators=(",", ":")) + "\n")
                    navigation_text = (
                        f"global_nav={global_anchor_steps / max(move_steps, 1):.1%} "
                        if navigation_actions else ""
                    )
                    report(
                        f"update={update} matches={sum(batch.completed.values())}/{target_completed} "
                        f"hero_steps={hero_steps} loss={loss:.6f} "
                        f"rollout_s={batch.rollout_seconds:.2f} ppo_s={ppo_seconds:.2f} "
                        f"env_steps_s={steps * workers / max(batch.rollout_seconds, 1e-9):.1f} "
                        f"group_min_s={min(batch.group_times):.2f} "
                        f"group_max_s={max(batch.group_times):.2f} "
                        f"wrong_lane={wrong_lane_steps / max(lane_active_steps, 1):.1%} "
                        f"{navigation_text}"
                        f"policy_lag={policy_lag}",
                    )
                    if next_future is None:
                        if sum(completed.values()) >= target_completed:
                            break
                        actor_policy = policy
                        batch = collect_grouped_rollout(
                            groups, actor_forward, device, pool, pending, completed, outcomes,
                            seed_rng, max_steps, steps, hero_steps, update, None,
                            use_bf16, action_sampler, frozen_policies, gamma, gae_lambda,
                        )
                        continue
                    batch = next_batch
    elapsed = time.perf_counter() - started
    manifest = {
        "version": model_version, "schema_hash": schema_hash.hex(),
        "reward_hash": reward_hash.hex(), "hidden_size": policy.hidden_size,
        "roster_ids": AI40_ROSTER.tolist(), "hero_steps": hero_steps,
        "updates": update, "completed_matches": dict(completed),
        "outcomes": dict(outcomes), **config,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(
        f"training complete: matches={sum(completed.values())} hero_steps={hero_steps} "
        f"elapsed={elapsed:.1f}s checkpoint={output / checkpoint_name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--ai40-matches", type=int, default=100)
    parser.add_argument("--ai30-matches", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/async"))
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--env-gomaxprocs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-learner", action="store_true")
    args = parser.parse_args()
    train_async(
        args.env, args.ai40_matches, args.ai30_matches, args.steps, args.workers,
        args.group_size, args.max_steps, torch.device(args.device), args.output,
        args.minibatch_size, args.resume, args.env_gomaxprocs, args.seed,
        not args.no_pipeline, not args.no_bf16,
        compile_model=args.compile_learner and not args.no_compile,
        compile_actor=not args.no_compile,
    )


if __name__ == "__main__":
    main()
