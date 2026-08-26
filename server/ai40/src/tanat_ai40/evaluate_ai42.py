"""Deterministic headless AI-42 versus scripted AI-30 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .env import (
    AI42_EVALUATION_PROTOCOL_VERSION,
    AI42_EVALUATION_SCHEMA_HASH,
    AssaultVectorEnv,
    CONTROLLER_AI30,
    CONTROLLER_AI40,
    HERO_COUNT,
    self_play_rosters,
)
from .evaluate_ai30 import ACTION_NAMES, controlled_slot_indices, wilson_interval
from .learner_ai42 import (
    AI42LearnerConfig,
    build_learner_manifest,
    inspect_ai42_checkpoint,
)
from .model_ai42_actor import (
    AI42Actor,
    CONTROL_CANCEL,
    CONTROL_CLASSES,
    CONTROL_HOLD,
    CONTROL_ISSUE,
    CONTROL_NAMES,
    CONTROL_WAIT,
)
from .train import stack_observations
from .train_ai42_bc import _training_config_defaults, head_weights_hash
from .train_async import _effective_target_mask_tensors


RUNTIME_CONTROL_ISSUE = 0
RUNTIME_CONTROL_HOLD = 1
RUNTIME_CONTROL_IDLE = 2
RUNTIME_CONTROL_NAMES = ("issue", "hold", "idle")


def controllers_for_side(ai42_side: int) -> tuple[int, ...]:
    if ai42_side not in (1, 2):
        raise ValueError("AI-42 side must be 1 or 2")
    first = CONTROLLER_AI40 if ai42_side == 1 else CONTROLLER_AI30
    second = CONTROLLER_AI30 if ai42_side == 1 else CONTROLLER_AI40
    return (first,) * (HERO_COUNT // 2) + (second,) * (HERO_COUNT // 2)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_kwargs(defaults: Mapping[str, Any]) -> dict[str, int]:
    names = (
        "hidden_size", "model_width", "entity_layers", "num_heads",
        "ff_multiplier",
    )
    return {name: int(defaults[name]) for name in names}


def load_actor(checkpoint: Path, config_path: Path, device: torch.device) -> tuple[AI42Actor, dict[str, Any]]:
    """Fully validate one immutable BC generation and load only its actor."""

    defaults = _training_config_defaults(config_path)
    model_kwargs = _model_kwargs(defaults)
    actor = AI42Actor(**model_kwargs)
    artifact = inspect_ai42_checkpoint(checkpoint, model=actor, map_location="cpu")
    manifest = dict(artifact.manifest)
    required = {
        "dataset_hash", "class_weights", "head_weights", "head_weights_hash",
        "config_hash", "model_hash",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"AI-42 checkpoint manifest is missing {missing}")
    expected_head_weights = dict(defaults["head_weights"])
    if manifest["head_weights"] != expected_head_weights:
        raise RuntimeError("AI-42 checkpoint head weights do not match evaluation config")
    if manifest["head_weights_hash"] != head_weights_hash(expected_head_weights):
        raise RuntimeError("AI-42 checkpoint head-weight hash is invalid")
    learner_config = AI42LearnerConfig(
        learning_rate=float(defaults["learning_rate"]),
        weight_decay=float(defaults["weight_decay"]),
        class_balance_power=float(defaults["class_balance_power"]),
        offset_coordinate_loss_weight=float(defaults["offset_coordinate_loss_weight"]),
        max_gradient_norm=float(defaults["max_gradient_norm"]),
        head_weights=expected_head_weights,
        class_weights=manifest["class_weights"],
        model_kwargs=model_kwargs,
    )
    reconstructed = build_learner_manifest(actor, learner_config, str(manifest["dataset_hash"]))
    for field in ("model_hash", "config_hash", "dataset_hash"):
        if reconstructed[field] != manifest[field]:
            raise RuntimeError(f"AI-42 checkpoint {field} does not match evaluation lineage")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(saved["model_state_dict"], strict=True)
    actor.to(device).eval().requires_grad_(False)
    return actor, {
        "checkpoint_sha256": _file_sha256(checkpoint),
        "manifest_digest": saved["manifest_digest"],
        "payload_digest": artifact.payload_digest,
        "model_artifact_hash": artifact.artifact_hashes["model"],
        "dataset_hash": manifest["dataset_hash"],
        "head_weights": expected_head_weights,
        "head_weights_hash": manifest["head_weights_hash"],
        "step": artifact.step,
        "epoch": artifact.epoch,
    }


class AI42EvaluationActor:
    """Fixed-size batched greedy actor with death-aware recurrent state."""

    def __init__(self, actor: AI42Actor, batch_size: int, device: torch.device):
        if batch_size < 1:
            raise ValueError("evaluation actor batch size must be positive")
        self.actor = actor
        self.batch_size = batch_size
        self.device = device
        self.h, self.c = actor.initial_state(batch_size, device)
        self.previous_dead = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.active_order = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def reset_workers(self, worker_indices: list[int]) -> None:
        for worker in worker_indices:
            start = worker * (HERO_COUNT // 2)
            stop = start + HERO_COUNT // 2
            self.h[start:stop].zero_()
            self.c[start:stop].zero_()
            self.previous_dead[start:stop].zero_()
            self.active_order[start:stop].zero_()

    def act(
        self, observations: Any, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if all(getattr(result, "active_order", None) is not None for result in observations):
            active = np.concatenate([result.active_order for result in observations])[indices]
            self.active_order.copy_(
                torch.as_tensor(active, dtype=torch.bool, device=self.device),
            )
        batch = stack_observations(observations)

        def tensor(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            value = torch.as_tensor(np.ascontiguousarray(getattr(batch, name)[indices]), device=self.device)
            return value if dtype is None else value.to(dtype=dtype)

        hero = tensor("hero", torch.float32)
        abilities = tensor("abilities", torch.float32)
        entities = tensor("entities", torch.float32)
        global_state = tensor("global_state", torch.float32)
        entity_mask = tensor("entity_mask", torch.bool)
        kind_mask = tensor("kind_mask", torch.bool)
        target_mask = tensor("target_mask", torch.bool)
        skill_target_mask = tensor("skill_target_mask", torch.bool)
        dead = hero[:, 9] >= 0.5
        recurrent_reset = dead | self.previous_dead
        self.h = torch.where(recurrent_reset.unsqueeze(1), torch.zeros_like(self.h), self.h)
        self.c = torch.where(recurrent_reset.unsqueeze(1), torch.zeros_like(self.c), self.c)
        self.active_order = torch.where(
            recurrent_reset, torch.zeros_like(self.active_order), self.active_order,
        )
        self.previous_dead = dead

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.actor(
                hero, abilities, entities, global_state, entity_mask, self.h, self.c,
            )
            self.h = output["h"].detach()
            self.c = output["c"].detach()
            control_logits = output["control"].float()
            model_controls = control_logits.argmax(dim=1)
            runtime_logits = torch.stack((
                control_logits[:, CONTROL_ISSUE],
                control_logits[:, CONTROL_HOLD],
                torch.logsumexp(
                    control_logits[:, [CONTROL_WAIT, CONTROL_CANCEL]], dim=1,
                ),
            ), dim=1)
            # HOLD is a transition, not a standalone action. Masking it until
            # ISSUE creates a lineage prevents a class-imbalanced checkpoint
            # from entering a permanent no-op state on the opening frame.
            runtime_logits[:, RUNTIME_CONTROL_HOLD].masked_fill_(
                ~self.active_order, -1e9,
            )
            runtime_controls = runtime_logits.argmax(dim=1)
            self.active_order = torch.where(
                runtime_controls == RUNTIME_CONTROL_ISSUE,
                torch.ones_like(self.active_order),
                torch.where(
                    runtime_controls == RUNTIME_CONTROL_IDLE,
                    torch.zeros_like(self.active_order),
                    self.active_order,
                ),
            )
            kinds = output["kind"].float().masked_fill(~kind_mask, -1e9).argmax(dim=1)
            rows = torch.arange(self.batch_size, device=self.device)
            effective_targets = _effective_target_mask_tensors(
                target_mask, skill_target_mask, kinds,
            ) & entity_mask
            empty = ~effective_targets.any(dim=1)
            effective_targets = effective_targets.clone()
            effective_targets[empty, 0] = True
            targets = output["target"][rows, kinds].float().masked_fill(
                ~effective_targets, -1e9,
            ).argmax(dim=1)
            offsets = output["offset"][rows, kinds].float().argmax(dim=1)
            anchors = output["anchor"][rows, kinds].float().argmax(dim=1)
            skills = (kinds >= 3) & (kinds <= 6)
            anchors = torch.where(skills, torch.zeros_like(anchors), anchors)
            targets = torch.where((kinds >= 2) & (kinds <= 6), targets, torch.zeros_like(targets))
            offsets = torch.where((kinds == 1) | skills, offsets, torch.zeros_like(offsets))
            anchors = torch.where(kinds == 1, anchors, torch.zeros_like(anchors))
            actions = torch.stack((kinds, targets, offsets, anchors), dim=1).to(torch.int16)
            actions = torch.where(
                (runtime_controls == RUNTIME_CONTROL_ISSUE).unsqueeze(1),
                actions, torch.zeros_like(actions),
            )
        return (
            actions.cpu().numpy(), runtime_controls.cpu().numpy(),
            model_controls.cpu().numpy(),
        )


def evaluate_vs_ai30(
    checkpoint: Path,
    config: Path,
    executable: Path,
    matches: int,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int = 42_000,
    backend: str = "torch",
    onnx_path: Path | None = None,
) -> dict[str, Any]:
    if matches < 1 or workers < 1 or max_steps < 1:
        raise ValueError("matches, workers and max_steps must be positive")
    if backend not in {"torch", "onnxruntime"}:
        raise ValueError("backend must be torch or onnxruntime")
    if backend == "onnxruntime" and onnx_path is None:
        raise ValueError("onnxruntime backend requires onnx_path")
    load_started = time.perf_counter()
    actor_device = device if backend == "torch" else torch.device("cpu")
    actor, lineage = load_actor(checkpoint, config, actor_device)
    onnx_session = None
    if backend == "onnxruntime":
        from .export_ai42 import export_ai42_actor
        from .inference_ai42_onnx import create_onnx_session

        export_ai42_actor(actor, onnx_path)
        onnx_session = create_onnx_session(str(onnx_path), cuda=device.type == "cuda")
    model_load_seconds = time.perf_counter() - load_started
    workers = min(workers, matches)
    roster_rng = np.random.default_rng(seed)
    next_match = workers
    assignments = [1 + index % 2 for index in range(workers)]
    outcomes: Counter[str] = Counter()
    side_outcomes = {1: Counter(), 2: Counter()}
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    model_controls = np.zeros(CONTROL_CLASSES, dtype=np.int64)
    runtime_controls = np.zeros(len(RUNTIME_CONTROL_NAMES), dtype=np.int64)
    invalid = total_steps = 0
    candidate_reward = 0.0
    inference_seconds = 0.0
    environment_step_seconds = 0.0
    environment_reset_seconds = 0.0
    policy_batches = 0
    policy_rows = 0
    decision_margin_samples: list[np.ndarray] = []
    if backend == "torch" and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with AssaultVectorEnv(executable, workers, AI42_EVALUATION_PROTOCOL_VERSION) as env:
        reset_started = time.perf_counter()
        observations = env.reset(
            range(seed, seed + workers), max_steps,
            controller_sets=[controllers_for_side(side) for side in assignments],
            rosters=self_play_rosters(roster_rng, workers),
        )
        environment_reset_seconds += time.perf_counter() - reset_started
        next_seed = seed + workers
        if backend == "torch":
            evaluator = AI42EvaluationActor(actor, workers * (HERO_COUNT // 2), device)
        else:
            from .inference_ai42_onnx import AI42ONNXEvaluationActor

            evaluator = AI42ONNXEvaluationActor(
                onnx_session,
                workers * (HERO_COUNT // 2),
                actor.hidden_size,
            )
        completed = 0
        while completed < matches:
            indices = controlled_slot_indices(assignments)
            inference_started = time.perf_counter()
            candidate_actions, candidate_runtime, candidate_model = evaluator.act(
                observations, indices,
            )
            inference_seconds += time.perf_counter() - inference_started
            policy_batches += 1
            policy_rows += int(indices.size)
            margins = getattr(evaluator, "last_decision_margin", None)
            if margins is not None:
                decision_margin_samples.append(np.asarray(margins, dtype=np.float32).copy())
            action_values = np.zeros((workers, HERO_COUNT, 5), dtype=np.int16)
            for index, side in enumerate(assignments):
                if side == 0:
                    continue
                local_start = 0 if side == 1 else HERO_COUNT // 2
                source = slice(index * (HERO_COUNT // 2), (index + 1) * (HERO_COUNT // 2))
                action_values[index, local_start:local_start + HERO_COUNT // 2, 0] = candidate_runtime[source]
                action_values[index, local_start:local_start + HERO_COUNT // 2, 1:] = candidate_actions[source]
                issued = candidate_runtime[source] == RUNTIME_CONTROL_ISSUE
                if issued.any():
                    actions += np.bincount(
                        candidate_actions[source][issued, 0], minlength=len(ACTION_NAMES),
                    )
                runtime_controls += np.bincount(
                    candidate_runtime[source], minlength=len(RUNTIME_CONTROL_NAMES),
                )
                model_controls += np.bincount(
                    candidate_model[source], minlength=CONTROL_CLASSES,
                )
            step_started = time.perf_counter()
            results = env.step(action_values)
            environment_step_seconds += time.perf_counter() - step_started
            reset_indices: list[int] = []
            reset_seeds: list[int] = []
            reset_controllers: list[tuple[int, ...]] = []
            for index, result in enumerate(results):
                side = assignments[index]
                if side == 0:
                    continue
                local = slice(0, HERO_COUNT // 2) if side == 1 else slice(HERO_COUNT // 2, HERO_COUNT)
                candidate_reward += float(result.rewards[local].sum())
                invalid += int(result.invalid[local].sum())
                if not result.done:
                    continue
                completed += 1
                total_steps += result.step
                outcome = "draw" if result.winner == 0 else ("win" if result.winner == side else "loss")
                outcomes[outcome] += 1
                side_outcomes[side][outcome] += 1
                if next_match < matches:
                    replacement_side = 1 + next_match % 2
                    next_match += 1
                    assignments[index] = replacement_side
                    reset_indices.append(index)
                    reset_seeds.append(next_seed)
                    reset_controllers.append(controllers_for_side(replacement_side))
                    next_seed += 1
                else:
                    assignments[index] = 0
                evaluator.reset_workers([index])
            observations = results
            if reset_indices:
                reset_started = time.perf_counter()
                replacements = env.reset_indices(
                    reset_indices, reset_seeds, max_steps,
                    controller_sets=reset_controllers,
                    rosters=self_play_rosters(roster_rng, len(reset_indices)),
                )
                environment_reset_seconds += time.perf_counter() - reset_started
                for index, replacement in replacements.items():
                    observations[index] = replacement
    elapsed = time.perf_counter() - started
    wins, losses, draws = outcomes["win"], outcomes["loss"], outcomes["draw"]
    lower, upper = wilson_interval(wins, matches)
    action_total = max(int(actions.sum()), 1)
    runtime_control_total = max(int(runtime_controls.sum()), 1)
    model_control_total = max(int(model_controls.sum()), 1)
    accounted_seconds = inference_seconds + environment_step_seconds + environment_reset_seconds
    cuda_memory = None
    if backend == "torch" and device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    margin_profile = None
    if decision_margin_samples:
        margins = np.concatenate(decision_margin_samples)
        quantiles = np.quantile(margins, [0.1, 0.25, 0.5, 0.75, 0.9])
        margin_profile = {
            "count": int(margins.size),
            "min": float(margins.min()),
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "max": float(margins.max()),
        }
    return {
        "format": "AI42-headless-evaluation-v1",
        "protocol_version": AI42_EVALUATION_PROTOCOL_VERSION,
        "protocol_schema_hash": AI42_EVALUATION_SCHEMA_HASH.hex(),
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config.resolve()),
        "lineage": lineage,
        "opponent": "AI-30",
        "seed": seed,
        "matches": matches,
        "workers": workers,
        "max_steps": max_steps,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / matches,
        "score_rate": (wins + 0.5 * draws) / matches,
        "draw_rate": draws / matches,
        "win_rate_ci95_low": lower,
        "win_rate_ci95_high": upper,
        "side_1": dict(side_outcomes[1]),
        "side_2": dict(side_outcomes[2]),
        "invalid_actions": invalid,
        "invalid_action_rate": invalid / action_total,
        "mean_ai42_team_reward": candidate_reward / matches,
        "mean_match_steps": total_steps / matches,
        "mean_match_minutes": total_steps * 0.2 / 60.0 / matches,
        "action_counts": {name: int(actions[index]) for index, name in enumerate(ACTION_NAMES)},
        "action_rates": {name: float(actions[index] / action_total) for index, name in enumerate(ACTION_NAMES)},
        "combat_action_count": int(actions[2:7].sum()),
        "model_control_counts": {
            name: int(model_controls[index]) for index, name in enumerate(CONTROL_NAMES)
        },
        "model_control_rates": {
            name: float(model_controls[index] / model_control_total)
            for index, name in enumerate(CONTROL_NAMES)
        },
        "runtime_control_counts": {
            name: int(runtime_controls[index])
            for index, name in enumerate(RUNTIME_CONTROL_NAMES)
        },
        "runtime_control_rates": {
            name: float(runtime_controls[index] / runtime_control_total)
            for index, name in enumerate(RUNTIME_CONTROL_NAMES)
        },
        "control_mapping": {"issue": "issue", "hold": "hold", "wait+cancel": "idle"},
        "protocol_faithful": True,
        "elapsed_seconds": elapsed,
        "matches_per_second": matches / max(elapsed, 1e-9),
        "runtime_profile": {
            "version": 1,
            "backend": backend,
            "device": str(device),
            "model_load_seconds": model_load_seconds,
            "model_inference_seconds": inference_seconds,
            "environment_step_seconds": environment_step_seconds,
            "environment_reset_seconds": environment_reset_seconds,
            "accounted_seconds": accounted_seconds,
            "unaccounted_seconds": max(0.0, elapsed - accounted_seconds),
            "model_inference_share": inference_seconds / max(elapsed, 1e-9),
            "environment_step_share": environment_step_seconds / max(elapsed, 1e-9),
            "policy_batches": policy_batches,
            "policy_rows": policy_rows,
            "policy_rows_per_second": policy_rows / max(inference_seconds, 1e-9),
            "decision_margin": margin_profile,
            "cuda_memory": cuda_memory,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an immutable AI-42 generation against AI-30")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=40)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("torch", "onnxruntime"), default="torch")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--seed", type=int, default=42_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_vs_ai30(
        args.checkpoint, args.config, args.env, args.matches, args.workers,
        args.max_steps, torch.device(args.device), args.seed, args.backend, args.onnx,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
