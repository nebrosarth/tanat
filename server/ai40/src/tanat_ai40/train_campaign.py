from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from .evaluate_ai30 import evaluate_vs_ai30, evaluate_vs_checkpoint, wilson_interval
from .train_async import train_async


SUMMARY_FIELDS = (
    "stage", "checkpoint_update", "checkpoint_hero_steps", "train_mirror_matches",
    "train_ai30_matches", "train_historical_matches", "eval_matches", "wins", "losses", "draws", "win_rate",
    "score_rate", "win_rate_ci95_low", "win_rate_ci95_high", "side_1_win_rate",
    "side_2_win_rate", "invalid_action_rate", "mean_ai40_team_reward",
    "mean_match_minutes", "evaluation_level", "total_evaluation_matches",
    "evaluation_seconds", "promotion_accepted", "promotion_composite_score",
    "promotion_reference_score", "promotion_ai30_win_rate", "promotion_stage005_win_rate",
    "promotion_pool_win_rate", "checkpoint",
)


def report(message: str) -> None:
    """Emit live campaign progress on stderr.

    The Windows virtual-environment launcher may swallow a child process's
    stdout, while stderr remains connected to the PowerShell console.
    """
    print(message, file=sys.stderr, flush=True)


def read_checkpoint_progress(checkpoint: Path | None) -> tuple[int, int]:
    if checkpoint is None:
        return 0, 0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("config", {})
    completed = config.get("completed_matches", {})
    return int(completed.get("ai40", 0)), int(completed.get("ai30", 0))


def read_checkpoint_completed(checkpoint: Path | None) -> dict[str, int]:
    if checkpoint is None:
        return {"ai40": 0, "ai30": 0, "historical": 0}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    completed = saved.get("config", {}).get("completed_matches", {})
    return {key: int(completed.get(key, 0)) for key in ("ai40", "ai30", "historical")}


def aggregate_evaluations(values: list[dict], label: str) -> dict:
    if not values:
        return {"label": label, "available": False, "matches": 0}
    matches = sum(int(item["matches"]) for item in values)
    wins = sum(int(item["wins"]) for item in values)
    losses = sum(int(item["losses"]) for item in values)
    draws = sum(int(item["draws"]) for item in values)
    low, high = wilson_interval(wins, matches)
    return {
        "label": label, "available": True, "matches": matches,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / matches,
        "score_rate": (wins + 0.5 * draws) / matches,
        "win_rate_ci95_low": low, "win_rate_ci95_high": high,
        "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in values),
        "opponents": values,
    }


def promotion_suite_score(suite: dict) -> float:
    available = [
        item for item in suite["categories"].values() if item.get("available", True)
    ]
    return sum(float(item["score_rate"]) for item in available) / len(available)


def promotion_is_safe(candidate: dict, reference: dict, tolerance: float,
                      max_category_regression: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if promotion_suite_score(candidate) + tolerance < promotion_suite_score(reference):
        reasons.append("composite score regressed")
    for name, current in candidate["categories"].items():
        previous = reference["categories"].get(name)
        if not current.get("available", True) or not previous or not previous.get("available", True):
            continue
        if float(current["score_rate"]) + max_category_regression < float(previous["score_rate"]):
            reasons.append(f"{name} score regressed")
    return not reasons, reasons


def promotion_category_regressed(candidate: dict, reference: dict,
                                 max_category_regression: float) -> bool:
    """Return whether one fully evaluated category is a hard promotion reject."""
    return (
        candidate.get("available", True) and reference.get("available", True) and
        float(candidate["score_rate"]) + max_category_regression <
        float(reference["score_rate"])
    )


def side_win_rate(side: dict) -> float:
    total = sum(int(side.get(key, 0)) for key in ("win", "loss", "draw"))
    return float(side.get("win", 0)) / max(total, 1)


def summary_row(stage: int, mirror: int, ai30: int, metrics: dict, checkpoint: Path,
                historical: int = 0) -> dict:
    row = {
        "stage": stage,
        "checkpoint_update": metrics["checkpoint_update"],
        "checkpoint_hero_steps": metrics["checkpoint_hero_steps"],
        "train_mirror_matches": mirror,
        "train_ai30_matches": ai30,
        "train_historical_matches": historical,
        "eval_matches": metrics["matches"],
        "wins": metrics["wins"], "losses": metrics["losses"], "draws": metrics["draws"],
        "win_rate": metrics["win_rate"], "score_rate": metrics["score_rate"],
        "win_rate_ci95_low": metrics["win_rate_ci95_low"],
        "win_rate_ci95_high": metrics["win_rate_ci95_high"],
        "side_1_win_rate": side_win_rate(metrics["side_1"]),
        "side_2_win_rate": side_win_rate(metrics["side_2"]),
        "invalid_action_rate": metrics["invalid_action_rate"],
        "mean_ai40_team_reward": metrics["mean_ai40_team_reward"],
        "mean_match_minutes": metrics["mean_match_minutes"],
        "evaluation_level": metrics.get("adaptive_evaluation_level", "fixed"),
        "total_evaluation_matches": metrics.get("total_evaluation_matches", metrics["matches"]),
        "evaluation_seconds": metrics.get("total_evaluation_seconds", metrics["elapsed_seconds"]),
        "checkpoint": str(checkpoint.resolve()),
    }
    promotion = metrics.get("promotion")
    if promotion:
        candidate = promotion["candidate"]
        reference = promotion["reference"]
        categories = candidate["categories"]
        row.update({
            "promotion_accepted": promotion["accepted"],
            "promotion_composite_score": candidate["composite_score_rate"],
            "promotion_reference_score": reference["composite_score_rate"],
            "promotion_ai30_win_rate": categories["ai30"].get("win_rate"),
            "promotion_stage005_win_rate": categories["stage005"].get("win_rate"),
            "promotion_pool_win_rate": categories["historical_pool"].get("win_rate"),
        })
    return row


def write_summary(output: Path, row: dict) -> None:
    jsonl_path = output / "metrics.jsonl"
    rows: dict[int, dict] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                saved = json.loads(line)
                rows[int(saved["stage"])] = saved
    rows[int(row["stage"])] = row
    ordered = [rows[index] for index in sorted(rows)]
    csv_temp = output / "metrics.csv.tmp"
    with csv_temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    csv_temp.replace(output / "metrics.csv")
    jsonl_temp = output / "metrics.jsonl.tmp"
    jsonl_temp.write_text("".join(
        json.dumps(saved, ensure_ascii=False, separators=(",", ":")) + "\n"
        for saved in ordered
    ), encoding="utf-8")
    jsonl_temp.replace(jsonl_path)


def best_score(metrics: dict) -> tuple[float, float, int]:
    return (
        float(metrics["win_rate_ci95_low"]),
        float(metrics["score_rate"]),
        int(metrics["checkpoint_hero_steps"]),
    )


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluation_plan(args: argparse.Namespace) -> tuple[tuple[str, int, float | None], ...]:
    return (
        ("quick", args.eval_matches, args.eval_medium_win_rate),
        ("medium", args.eval_medium_matches, args.eval_final_win_rate),
        ("confirmation", args.eval_final_matches, None),
    )


def target_is_reached(metrics: dict | None, args: argparse.Namespace) -> bool:
    return bool(
        metrics and
        float(metrics["win_rate"]) >= args.stop_win_rate and
        float(metrics["win_rate_ci95_low"]) >= args.stop_ci_low
    )


def synchronize_target_state(state: dict, args: argparse.Namespace) -> bool:
    """Apply changed stop thresholds to an existing campaign state.

    Older campaign files did not persist their thresholds.  Without this
    reconciliation, a campaign that met an earlier, lower target could never
    resume after the user raised the target.
    """
    target_changed = (
        state.get("stop_win_rate") is None or
        state.get("stop_ci_low") is None or
        float(state["stop_win_rate"]) != args.stop_win_rate or
        float(state["stop_ci_low"]) != args.stop_ci_low
    )
    if not target_changed:
        return False
    state["stop_win_rate"] = args.stop_win_rate
    state["stop_ci_low"] = args.stop_ci_low
    if target_is_reached(state.get("last_evaluation"), args):
        state["target_reached"] = True
        state["target_reached_stage"] = int(state["completed_stages"])
    else:
        state.pop("target_reached", None)
        state.pop("target_reached_stage", None)
    return True


def synchronize_evaluation_thresholds(state: dict, args: argparse.Namespace) -> bool:
    """Apply user-selected adaptive-evaluation gates to a resumed campaign.

    These gates only decide whether a checkpoint receives a larger evaluation
    sample; every evaluation record stores its actual level and match count.
    Keeping a history makes a changed gate visible when comparing stages.
    """
    names = ("eval_medium_win_rate", "eval_final_win_rate")
    previous = {name: float(state[name]) for name in names}
    current = {name: float(getattr(args, name)) for name in names}
    if previous == current:
        return False
    state.update(current)
    state.setdefault("evaluation_threshold_history", []).append({
        "applied_before_stage": int(state["completed_stages"]) + 1,
        "at": time.time(),
        "previous": previous,
        "current": current,
    })
    return True


def adaptive_evaluate(
    checkpoint: Path,
    executable: Path,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    evaluator=evaluate_vs_ai30,
) -> dict:
    passes: list[dict] = []
    metrics = None
    for index, (level, matches, escalate_at) in enumerate(evaluation_plan(args)):
        pass_seed = seed + index * 100_000
        report(f"evaluation level={level} matches={matches} seed={pass_seed}")
        metrics = evaluator(
            checkpoint, executable, matches, min(workers, matches), max_steps,
            device, pass_seed,
        )
        report(
            f"evaluation level={level} result: "
            f"{metrics['wins']}-{metrics['losses']}-{metrics['draws']} "
            f"winrate={metrics['win_rate']:.1%} score={metrics['score_rate']:.1%} "
            f"CI95=[{metrics['win_rate_ci95_low']:.1%}, "
            f"{metrics['win_rate_ci95_high']:.1%}] "
            f"elapsed={metrics['elapsed_seconds']:.1f}s",
        )
        passes.append({
            "level": level, "matches": matches, "seed": pass_seed,
            "wins": metrics["wins"], "losses": metrics["losses"],
            "draws": metrics["draws"], "win_rate": metrics["win_rate"],
            "score_rate": metrics["score_rate"],
            "win_rate_ci95_low": metrics["win_rate_ci95_low"],
            "win_rate_ci95_high": metrics["win_rate_ci95_high"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        })
        if escalate_at is None or metrics["win_rate"] < escalate_at:
            break
    assert metrics is not None
    metrics["adaptive_evaluation_level"] = passes[-1]["level"]
    metrics["evaluation_passes"] = passes
    metrics["total_evaluation_matches"] = sum(item["matches"] for item in passes)
    metrics["total_evaluation_seconds"] = sum(item["elapsed_seconds"] for item in passes)
    return metrics


def skipped_promotion_category(label: str, reason: str) -> dict:
    return {
        "label": label, "available": False, "matches": 0,
        "skipped": True, "skip_reason": reason,
    }


def promotion_suite(checkpoint: Path, seed: int, categories: dict,
                    complete: bool = True) -> dict:
    suite = {
        "checkpoint": str(checkpoint.resolve()), "seed": seed,
        "categories": categories, "complete": complete,
    }
    suite["composite_score_rate"] = promotion_suite_score(suite)
    return suite


def evaluate_promotion_ai30(
    checkpoint: Path,
    executable: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    role: str,
    ai30_evaluator=evaluate_vs_ai30,
) -> dict:
    matches = args.promotion_eval_matches
    report(f"promotion {role} evaluation vs AI-30 matches={matches}")
    metrics = ai30_evaluator(
        checkpoint, executable, matches, min(args.eval_workers, matches),
        args.max_steps, device, seed,
    )
    metrics["label"] = "ai30"
    report(
        f"promotion {role} AI-30 WR={metrics['win_rate']:.1%} score={metrics['score_rate']:.1%} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s"
    )
    return metrics


def evaluate_promotion_checkpoint(
    checkpoint: Path,
    opponent: Path,
    executable: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    label: str,
    role: str,
    checkpoint_evaluator=evaluate_vs_checkpoint,
) -> dict:
    matches = args.promotion_eval_matches
    report(f"promotion {role} evaluation vs {label} matches={matches}")
    metrics = checkpoint_evaluator(
        checkpoint, opponent, executable, matches, min(args.eval_workers, matches),
        args.max_steps, device, seed, opponent_label=label,
    )
    metrics["label"] = label
    report(
        f"promotion {role} {label} WR={metrics['win_rate']:.1%} score={metrics['score_rate']:.1%} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s"
    )
    return metrics


def evaluate_promotion_pool(
    checkpoint: Path,
    anchor: Path,
    historical_pool: list[Path],
    executable: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    role: str,
    checkpoint_evaluator=evaluate_vs_checkpoint,
) -> dict:
    matches = args.promotion_eval_matches
    past = [path for path in historical_pool if path.resolve() != anchor.resolve()]
    pool_values: list[dict] = []
    if past:
        base, remainder = divmod(matches, len(past))
        for index, opponent in enumerate(past):
            count = base + (1 if index < remainder else 0)
            if count == 0:
                break
            pool_values.append(checkpoint_evaluator(
                checkpoint, opponent, executable, count,
                min(args.eval_workers, count), args.max_steps, device,
                seed + index * 10_000, opponent_label=opponent.stem,
            ))
    pool = aggregate_evaluations(pool_values, "historical_pool")
    if pool.get("available"):
        report(
            f"promotion {role} historical-pool WR={pool['win_rate']:.1%} score={pool['score_rate']:.1%} "
            f"elapsed={pool['elapsed_seconds']:.1f}s"
        )
    else:
        report(f"promotion {role} historical-pool: no promoted past models yet")
    return pool


def evaluate_promotion_pair(
    candidate_checkpoint: Path,
    reference_checkpoint: Path,
    anchor: Path,
    historical_pool: list[Path],
    executable: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    ai30_evaluator=evaluate_vs_ai30,
    checkpoint_evaluator=evaluate_vs_checkpoint,
) -> tuple[dict, dict, list[str]]:
    """Evaluate matched promotion categories, stopping on a hard regression."""
    candidate_categories: dict[str, dict] = {}
    reference_categories: dict[str, dict] = {}
    for name, evaluator_seed, opponent, label in (
        ("ai30", seed, None, "AI-30"),
        ("stage005", seed + 100_000, anchor, "frozen stage-005"),
    ):
        if name == "ai30":
            candidate_value = evaluate_promotion_ai30(
                candidate_checkpoint, executable, device, args, evaluator_seed, "candidate",
                ai30_evaluator,
            )
            reference_value = evaluate_promotion_ai30(
                reference_checkpoint, executable, device, args, evaluator_seed, "promoted",
                ai30_evaluator,
            )
        else:
            candidate_value = evaluate_promotion_checkpoint(
                candidate_checkpoint, opponent, executable, device, args, evaluator_seed,
                label, "candidate", checkpoint_evaluator,
            )
            reference_value = evaluate_promotion_checkpoint(
                reference_checkpoint, opponent, executable, device, args, evaluator_seed,
                label, "promoted", checkpoint_evaluator,
            )
        candidate_value["label"] = name
        reference_value["label"] = name
        candidate_categories[name] = candidate_value
        reference_categories[name] = reference_value
        if promotion_category_regressed(
            candidate_value, reference_value, args.promotion_max_category_regression,
        ):
            reason = f"{name} score regressed"
            for skipped_name in ("ai30", "stage005", "historical_pool"):
                if skipped_name not in candidate_categories:
                    candidate_categories[skipped_name] = skipped_promotion_category(skipped_name, reason)
                    reference_categories[skipped_name] = skipped_promotion_category(skipped_name, reason)
            return (
                promotion_suite(candidate_checkpoint, seed, candidate_categories, complete=False),
                promotion_suite(reference_checkpoint, seed, reference_categories, complete=False),
                [reason],
            )

    pool_seed = seed + 200_000
    candidate_categories["historical_pool"] = evaluate_promotion_pool(
        candidate_checkpoint, anchor, historical_pool, executable, device, args,
        pool_seed, "candidate", checkpoint_evaluator,
    )
    reference_categories["historical_pool"] = evaluate_promotion_pool(
        reference_checkpoint, anchor, historical_pool, executable, device, args,
        pool_seed, "promoted", checkpoint_evaluator,
    )
    if promotion_category_regressed(
        candidate_categories["historical_pool"], reference_categories["historical_pool"],
        args.promotion_max_category_regression,
    ):
        reason = "historical_pool score regressed"
        return (
            promotion_suite(candidate_checkpoint, seed, candidate_categories),
            promotion_suite(reference_checkpoint, seed, reference_categories),
            [reason],
        )
    return (
        promotion_suite(candidate_checkpoint, seed, candidate_categories),
        promotion_suite(reference_checkpoint, seed, reference_categories),
        [],
    )


def run_campaign(args: argparse.Namespace, trainer=train_async,
                 evaluator=evaluate_vs_ai30,
                 checkpoint_evaluator=evaluate_vs_checkpoint) -> None:
    if min(args.stages, args.mirror_per_stage, args.ai30_per_stage, args.eval_matches,
           args.eval_medium_matches, args.eval_final_matches,
           args.workers, args.group_size) < 1:
        raise ValueError("stage, match and worker counts must be positive")
    if not (args.eval_matches <= args.eval_medium_matches <= args.eval_final_matches):
        raise ValueError("adaptive evaluation match counts must be non-decreasing")
    if not (0.0 <= args.eval_medium_win_rate <= args.eval_final_win_rate <= 1.0):
        raise ValueError("adaptive evaluation thresholds must be ordered probabilities")
    if args.historical_per_stage < 0 or args.promotion_eval_matches < 1:
        raise ValueError("historical match count must be non-negative and promotion evaluation positive")
    if args.historical_pool_size < 2:
        raise ValueError("historical pool size must be at least 2 (anchor plus past model)")
    if args.learning_rate <= 0 or args.ppo_epochs < 1:
        raise ValueError("learning rate and PPO epochs must be positive")
    if args.target_kl < 0:
        raise ValueError("target KL must be non-negative (zero disables early stopping)")
    if not (0 <= args.promotion_tolerance <= 1 and
            0 <= args.promotion_max_category_regression <= 1):
        raise ValueError("promotion tolerances must be probabilities")
    promotion_enabled = args.historical_anchor is not None
    anchor = args.historical_anchor.resolve() if promotion_enabled else None
    if anchor is not None and not anchor.is_file():
        raise RuntimeError(f"historical anchor checkpoint is missing: {anchor}")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    stages_dir, evaluations_dir = output / "checkpoints", output / "evaluations"
    stages_dir.mkdir(exist_ok=True)
    evaluations_dir.mkdir(exist_ok=True)
    latest = output / "latest.pt"
    promoted_copy = output / "promoted.pt"
    state_path = output / "campaign.json"
    state = load_state(state_path)
    if state is None:
        initial = args.resume.resolve() if args.resume else None
        base_mirror, base_ai30 = read_checkpoint_progress(initial)
        state = {
            "version": 2 if promotion_enabled else 1,
            "created_at": time.time(), "completed_stages": 0,
            "base_mirror_matches": base_mirror, "base_ai30_matches": base_ai30,
            "initial_checkpoint": str(initial) if initial else None,
            "mirror_per_stage": args.mirror_per_stage,
            "ai30_per_stage": args.ai30_per_stage,
            "eval_matches": args.eval_matches,
            "eval_medium_matches": args.eval_medium_matches,
            "eval_final_matches": args.eval_final_matches,
            "eval_medium_win_rate": args.eval_medium_win_rate,
            "eval_final_win_rate": args.eval_final_win_rate,
            "stop_win_rate": args.stop_win_rate,
            "stop_ci_low": args.stop_ci_low,
            "actor_learner_pipeline": not args.no_pipeline,
            "learning_rate": args.learning_rate,
            "ppo_epochs": args.ppo_epochs,
            "target_kl": args.target_kl,
            "max_steps": args.max_steps,
            "train_seed": args.train_seed,
            "eval_seed": args.eval_seed,
            "best": None,
            "historical_per_stage": args.historical_per_stage,
            "historical_pool": [str(anchor)] if anchor is not None else [],
            "historical_anchor": str(anchor) if anchor is not None else None,
            "promoted_checkpoint": str(initial) if initial else None,
        }
        if promotion_enabled:
            if initial is None:
                raise RuntimeError("historical promotion campaign requires an initial checkpoint")
            shutil.copy2(initial, promoted_copy)
            shutil.copy2(initial, latest)
        save_state(state_path, state)
    else:
        if (int(state["mirror_per_stage"]) != args.mirror_per_stage or
                int(state["ai30_per_stage"]) != args.ai30_per_stage):
            raise RuntimeError("do not change per-stage match counts in an existing campaign")
        for name in (
            "eval_matches", "eval_medium_matches", "eval_final_matches",
            "max_steps", "train_seed", "eval_seed",
            "ppo_epochs",
        ):
            if int(state[name]) != int(getattr(args, name)):
                raise RuntimeError(f"do not change {name} in an existing campaign")
        for name in ("learning_rate", "target_kl"):
            if float(state.get(name, 0.0)) != float(getattr(args, name)):
                raise RuntimeError(f"do not change {name} in an existing campaign")
        if int(state.get("historical_per_stage", 0)) != args.historical_per_stage:
            raise RuntimeError("do not change historical_per_stage in an existing campaign")
        if state.get("historical_anchor") != (str(anchor) if anchor is not None else None):
            raise RuntimeError("do not change historical_anchor in an existing campaign")
        if bool(state["actor_learner_pipeline"]) == bool(args.no_pipeline):
            raise RuntimeError("do not change actor-learner pipeline mode in an existing campaign")
        requested = str(args.resume.resolve()) if args.resume else None
        if not latest.exists():
            if promotion_enabled and promoted_copy.exists():
                shutil.copy2(promoted_copy, latest)
            elif int(state["completed_stages"]) > 0:
                raise RuntimeError("campaign latest.pt is missing; restore it before resuming")
            elif requested != state.get("initial_checkpoint"):
                raise RuntimeError("campaign initial checkpoint changed before its first saved update")
        if promotion_enabled and not promoted_copy.exists():
            promoted_source = state.get("promoted_checkpoint") or state.get("initial_checkpoint")
            if not promoted_source or not Path(promoted_source).is_file():
                raise RuntimeError("campaign promoted checkpoint is missing")
            shutil.copy2(promoted_source, promoted_copy)

    evaluation_thresholds_changed = synchronize_evaluation_thresholds(state, args)
    target_changed = synchronize_target_state(state, args)
    if evaluation_thresholds_changed or target_changed:
        save_state(state_path, state)
    if evaluation_thresholds_changed:
        report(
            f"adaptive evaluation thresholds updated: quick={args.eval_medium_win_rate:.1%} "
            f"medium={args.eval_final_win_rate:.1%}",
        )
    if target_changed:
        report(
            f"campaign target updated: winrate={args.stop_win_rate:.1%} "
            f"CI-low={args.stop_ci_low:.1%}; "
            f"reached={bool(state.get('target_reached'))}",
        )

    if state.get("target_reached") and not args.continue_after_target:
        report(
            f"campaign target already reached at stage={state['target_reached_stage']}; "
            "pass --continue-after-target to keep training",
        )
        return
    start_stage = int(state["completed_stages"]) + 1
    if start_stage > args.stages:
        report(f"campaign already complete: stages={state['completed_stages']} output={output}")
        return
    for stage in range(start_stage, args.stages + 1):
        resume = (promoted_copy if promotion_enabled else
                  (latest if latest.exists() else (args.resume.resolve() if args.resume else None)))
        if promotion_enabled:
            progress = read_checkpoint_completed(resume)
            target_mirror = progress["ai40"] + args.mirror_per_stage
            target_ai30 = progress["ai30"] + args.ai30_per_stage
            target_historical = progress["historical"] + args.historical_per_stage
        else:
            target_mirror = int(state["base_mirror_matches"]) + stage * args.mirror_per_stage
            target_ai30 = int(state["base_ai30_matches"]) + stage * args.ai30_per_stage
            target_historical = 0
        pool_paths = [Path(value) for value in state.get("historical_pool", [])]
        historical_opponents = {
            f"pool-{index:03d}": path for index, path in enumerate(pool_paths)
        }
        stage_seed = args.train_seed + stage * 1_000_003
        report(
            f"stage={stage}/{args.stages} training target: mirror={target_mirror} "
            f"vs_ai30={target_ai30} historical={target_historical} resume={resume}",
        )
        trainer(
            args.env, target_mirror, target_ai30, args.steps, args.workers,
            args.group_size, args.max_steps, torch.device(args.device), output,
            args.minibatch_size, resume, args.env_gomaxprocs, stage_seed,
            not args.no_pipeline, not args.no_bf16,
            compile_model=args.compile_learner and not args.no_compile,
            compile_actor=not args.no_compile,
            historical_matches=target_historical,
            historical_opponents=historical_opponents,
            discount_horizon_seconds=args.discount_horizon_seconds,
            gae_horizon_seconds=args.gae_horizon_seconds,
            checkpoint_name="candidate.pt" if promotion_enabled else "latest.pt",
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
            target_kl=args.target_kl if args.target_kl > 0 else None,
        )
        candidate_checkpoint = output / ("candidate.pt" if promotion_enabled else "latest.pt")
        frozen = stages_dir / f"stage-{stage:03d}.pt"
        shutil.copy2(candidate_checkpoint, frozen)
        evaluation_path = evaluations_dir / f"stage-{stage:03d}.json"
        eval_seed = args.eval_seed + stage * 1_000_000
        report(
            f"stage={stage}/{args.stages} adaptive evaluation seed_base={eval_seed}",
        )
        metrics = adaptive_evaluate(
            frozen, args.env, args.eval_workers, args.max_steps,
            torch.device(args.device), eval_seed, args, evaluator,
        )
        promoted = True
        promotion_reasons: list[str] = []
        if promotion_enabled:
            promotion_seed = args.promotion_eval_seed + stage * 1_000_000
            candidate_suite, reference_suite, promotion_reasons = evaluate_promotion_pair(
                frozen, promoted_copy, anchor, pool_paths, args.env, torch.device(args.device),
                args, promotion_seed, evaluator, checkpoint_evaluator,
            )
            if promotion_reasons:
                promoted = False
            else:
                promoted, promotion_reasons = promotion_is_safe(
                    candidate_suite, reference_suite, args.promotion_tolerance,
                    args.promotion_max_category_regression,
                )
            metrics["promotion"] = {
                "accepted": promoted, "reasons": promotion_reasons,
                "candidate": candidate_suite, "reference": reference_suite,
            }
            if promoted:
                previous_promoted = state.get("promoted_checkpoint")
                if (previous_promoted and int(state["completed_stages"]) > 0 and
                        Path(previous_promoted).resolve() != anchor):
                    pool_paths.append(Path(previous_promoted))
                unique_pool: list[Path] = []
                for path in pool_paths:
                    if path.resolve() not in {value.resolve() for value in unique_pool}:
                        unique_pool.append(path)
                if len(unique_pool) > args.historical_pool_size:
                    unique_pool = [anchor] + [
                        path for path in unique_pool if path.resolve() != anchor
                    ][-(args.historical_pool_size - 1):]
                state["historical_pool"] = [str(path.resolve()) for path in unique_pool]
                shutil.copy2(frozen, promoted_copy)
                shutil.copy2(frozen, latest)
                state["promoted_checkpoint"] = str(frozen.resolve())
                report(
                    f"stage={stage} promoted: composite "
                    f"{reference_suite['composite_score_rate']:.1%} -> "
                    f"{candidate_suite['composite_score_rate']:.1%}",
                )
            else:
                report(f"stage={stage} rejected: {', '.join(promotion_reasons)}")
        evaluation_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        row = summary_row(
            stage, target_mirror, target_ai30, metrics, frozen, target_historical,
        )
        write_summary(output, row)
        candidate = best_score(metrics)
        previous = tuple(state["best"]["score"]) if state.get("best") else None
        if promoted and (previous is None or candidate > previous):
            shutil.copy2(frozen, output / "best.pt")
            shutil.copy2(evaluation_path, output / "best-evaluation.json")
            state["best"] = {"stage": stage, "score": list(candidate)}
        state["completed_stages"] = stage
        state["latest_checkpoint"] = state.get("promoted_checkpoint", str(frozen.resolve()))
        state["last_evaluation"] = metrics
        save_state(state_path, state)
        report(
            f"stage={stage} result: {metrics['wins']}-{metrics['losses']}-{metrics['draws']} "
            f"winrate={metrics['win_rate']:.1%} score={metrics['score_rate']:.1%} "
            f"CI95=[{metrics['win_rate_ci95_low']:.1%}, {metrics['win_rate_ci95_high']:.1%}] "
            f"eval={metrics['adaptive_evaluation_level']}:{metrics['matches']} "
            f"promoted={promoted} best_stage={state['best']['stage'] if state.get('best') else 'none'}",
        )
        if promoted and target_is_reached(metrics, args):
            state["target_reached"] = True
            state["target_reached_stage"] = stage
            save_state(state_path, state)
            report(
                f"target reached at stage {stage}: winrate and confidence thresholds passed",
            )
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/campaign"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--mirror-per-stage", type=int, default=100)
    parser.add_argument("--ai30-per-stage", type=int, default=100)
    parser.add_argument("--historical-per-stage", type=int, default=0)
    parser.add_argument("--historical-anchor", type=Path)
    parser.add_argument("--historical-pool-size", type=int, default=8)
    parser.add_argument("--eval-matches", type=int, default=50)
    parser.add_argument("--eval-medium-matches", type=int, default=200)
    parser.add_argument("--eval-final-matches", type=int, default=500)
    parser.add_argument("--eval-medium-win-rate", type=float, default=0.40)
    parser.add_argument("--eval-final-win-rate", type=float, default=0.55)
    parser.add_argument("--eval-workers", type=int, default=64)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--target-kl", type=float, default=0.0,
                        help="stop a PPO update once approximate KL exceeds this value (0 disables)")
    parser.add_argument("--env-gomaxprocs", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-seed", type=int, default=1_000_000)
    parser.add_argument("--eval-seed", type=int, default=100_000_000)
    parser.add_argument("--promotion-eval-matches", type=int, default=50)
    parser.add_argument("--promotion-eval-seed", type=int, default=900_000_000)
    parser.add_argument("--promotion-tolerance", type=float, default=0.0)
    parser.add_argument("--promotion-max-category-regression", type=float, default=0.05)
    parser.add_argument("--discount-horizon-seconds", type=float, default=19.8998324946844)
    parser.add_argument("--gae-horizon-seconds", type=float, default=3.26032220809386)
    parser.add_argument("--stop-win-rate", type=float, default=0.60)
    parser.add_argument("--stop-ci-low", type=float, default=0.50)
    parser.add_argument("--continue-after-target", action="store_true")
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-learner", action="store_true",
        help="compile PPO learner too (high per-stage compilation cost)",
    )
    return parser


def main() -> None:
    run_campaign(build_parser().parse_args())


if __name__ == "__main__":
    main()
