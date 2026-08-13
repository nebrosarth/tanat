from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import torch

from .evaluate_ai30 import evaluate_vs_ai30
from .train_async import train_async


SUMMARY_FIELDS = (
    "stage", "checkpoint_update", "checkpoint_hero_steps", "train_mirror_matches",
    "train_ai30_matches", "eval_matches", "wins", "losses", "draws", "win_rate",
    "score_rate", "win_rate_ci95_low", "win_rate_ci95_high", "side_1_win_rate",
    "side_2_win_rate", "invalid_action_rate", "mean_ai40_team_reward",
    "mean_match_minutes", "evaluation_level", "total_evaluation_matches",
    "evaluation_seconds", "checkpoint",
)


def read_checkpoint_progress(checkpoint: Path | None) -> tuple[int, int]:
    if checkpoint is None:
        return 0, 0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("config", {})
    completed = config.get("completed_matches", {})
    return int(completed.get("ai40", 0)), int(completed.get("ai30", 0))


def side_win_rate(side: dict) -> float:
    total = sum(int(side.get(key, 0)) for key in ("win", "loss", "draw"))
    return float(side.get("win", 0)) / max(total, 1)


def summary_row(stage: int, mirror: int, ai30: int, metrics: dict, checkpoint: Path) -> dict:
    return {
        "stage": stage,
        "checkpoint_update": metrics["checkpoint_update"],
        "checkpoint_hero_steps": metrics["checkpoint_hero_steps"],
        "train_mirror_matches": mirror,
        "train_ai30_matches": ai30,
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


def adaptive_evaluate(
    checkpoint: Path,
    executable: Path,
    workers: int,
    max_steps: int,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    passes: list[dict] = []
    metrics = None
    for index, (level, matches, escalate_at) in enumerate(evaluation_plan(args)):
        pass_seed = seed + index * 100_000
        print(
            f"evaluation level={level} matches={matches} seed={pass_seed}", flush=True,
        )
        metrics = evaluate_vs_ai30(
            checkpoint, executable, matches, min(workers, matches), max_steps,
            device, pass_seed,
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


def run_campaign(args: argparse.Namespace) -> None:
    if min(args.stages, args.mirror_per_stage, args.ai30_per_stage, args.eval_matches,
           args.eval_medium_matches, args.eval_final_matches,
           args.workers, args.group_size) < 1:
        raise ValueError("stage, match and worker counts must be positive")
    if not (args.eval_matches <= args.eval_medium_matches <= args.eval_final_matches):
        raise ValueError("adaptive evaluation match counts must be non-decreasing")
    if not (0.0 <= args.eval_medium_win_rate <= args.eval_final_win_rate <= 1.0):
        raise ValueError("adaptive evaluation thresholds must be ordered probabilities")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    stages_dir, evaluations_dir = output / "checkpoints", output / "evaluations"
    stages_dir.mkdir(exist_ok=True)
    evaluations_dir.mkdir(exist_ok=True)
    latest = output / "latest.pt"
    state_path = output / "campaign.json"
    state = load_state(state_path)
    if state is None:
        initial = args.resume.resolve() if args.resume else None
        base_mirror, base_ai30 = read_checkpoint_progress(initial)
        state = {
            "version": 1, "created_at": time.time(), "completed_stages": 0,
            "base_mirror_matches": base_mirror, "base_ai30_matches": base_ai30,
            "initial_checkpoint": str(initial) if initial else None,
            "mirror_per_stage": args.mirror_per_stage,
            "ai30_per_stage": args.ai30_per_stage,
            "eval_matches": args.eval_matches,
            "eval_medium_matches": args.eval_medium_matches,
            "eval_final_matches": args.eval_final_matches,
            "eval_medium_win_rate": args.eval_medium_win_rate,
            "eval_final_win_rate": args.eval_final_win_rate,
            "actor_learner_pipeline": not args.no_pipeline,
            "max_steps": args.max_steps,
            "train_seed": args.train_seed,
            "eval_seed": args.eval_seed,
            "best": None,
        }
        save_state(state_path, state)
    else:
        if (int(state["mirror_per_stage"]) != args.mirror_per_stage or
                int(state["ai30_per_stage"]) != args.ai30_per_stage):
            raise RuntimeError("do not change per-stage match counts in an existing campaign")
        for name in (
            "eval_matches", "eval_medium_matches", "eval_final_matches",
            "max_steps", "train_seed", "eval_seed",
        ):
            if int(state[name]) != int(getattr(args, name)):
                raise RuntimeError(f"do not change {name} in an existing campaign")
        for name in ("eval_medium_win_rate", "eval_final_win_rate"):
            if float(state[name]) != float(getattr(args, name)):
                raise RuntimeError(f"do not change {name} in an existing campaign")
        if bool(state["actor_learner_pipeline"]) == bool(args.no_pipeline):
            raise RuntimeError("do not change actor-learner pipeline mode in an existing campaign")
        requested = str(args.resume.resolve()) if args.resume else None
        if not latest.exists():
            if int(state["completed_stages"]) > 0:
                raise RuntimeError("campaign latest.pt is missing; restore it before resuming")
            if requested != state.get("initial_checkpoint"):
                raise RuntimeError("campaign initial checkpoint changed before its first saved update")

    if state.get("target_reached") and not args.continue_after_target:
        print(
            f"campaign target already reached at stage={state['target_reached_stage']}; "
            "pass --continue-after-target to keep training", flush=True,
        )
        return
    start_stage = int(state["completed_stages"]) + 1
    if start_stage > args.stages:
        print(f"campaign already complete: stages={state['completed_stages']} output={output}", flush=True)
        return
    for stage in range(start_stage, args.stages + 1):
        target_mirror = int(state["base_mirror_matches"]) + stage * args.mirror_per_stage
        target_ai30 = int(state["base_ai30_matches"]) + stage * args.ai30_per_stage
        resume = latest if latest.exists() else (args.resume.resolve() if args.resume else None)
        stage_seed = args.train_seed + stage * 1_000_003
        print(
            f"stage={stage}/{args.stages} training target: mirror={target_mirror} "
            f"vs_ai30={target_ai30} resume={resume}", flush=True,
        )
        train_async(
            args.env, target_mirror, target_ai30, args.steps, args.workers,
            args.group_size, args.max_steps, torch.device(args.device), output,
            args.minibatch_size, resume, args.env_gomaxprocs, stage_seed,
            not args.no_pipeline,
        )
        frozen = stages_dir / f"stage-{stage:03d}.pt"
        shutil.copy2(latest, frozen)
        evaluation_path = evaluations_dir / f"stage-{stage:03d}.json"
        eval_seed = args.eval_seed + stage * 1_000_000
        print(
            f"stage={stage}/{args.stages} adaptive evaluation seed_base={eval_seed}",
            flush=True,
        )
        metrics = adaptive_evaluate(
            frozen, args.env, args.eval_workers, args.max_steps,
            torch.device(args.device), eval_seed, args,
        )
        evaluation_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        row = summary_row(stage, target_mirror, target_ai30, metrics, frozen)
        write_summary(output, row)
        candidate = best_score(metrics)
        previous = tuple(state["best"]["score"]) if state.get("best") else None
        if previous is None or candidate > previous:
            shutil.copy2(frozen, output / "best.pt")
            shutil.copy2(evaluation_path, output / "best-evaluation.json")
            state["best"] = {"stage": stage, "score": list(candidate)}
        state["completed_stages"] = stage
        state["latest_checkpoint"] = str(frozen.resolve())
        state["last_evaluation"] = metrics
        save_state(state_path, state)
        print(
            f"stage={stage} result: {metrics['wins']}-{metrics['losses']}-{metrics['draws']} "
            f"winrate={metrics['win_rate']:.1%} score={metrics['score_rate']:.1%} "
            f"CI95=[{metrics['win_rate_ci95_low']:.1%}, {metrics['win_rate_ci95_high']:.1%}] "
            f"eval={metrics['adaptive_evaluation_level']}:{metrics['matches']} "
            f"best_stage={state['best']['stage']}", flush=True,
        )
        if (metrics["win_rate"] >= args.stop_win_rate and
                metrics["win_rate_ci95_low"] >= args.stop_ci_low):
            state["target_reached"] = True
            state["target_reached_stage"] = stage
            save_state(state_path, state)
            print(
                f"target reached at stage {stage}: winrate and confidence thresholds passed",
                flush=True,
            )
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ai40/checkpoints/campaign"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--mirror-per-stage", type=int, default=100)
    parser.add_argument("--ai30-per-stage", type=int, default=100)
    parser.add_argument("--eval-matches", type=int, default=50)
    parser.add_argument("--eval-medium-matches", type=int, default=200)
    parser.add_argument("--eval-final-matches", type=int, default=500)
    parser.add_argument("--eval-medium-win-rate", type=float, default=0.40)
    parser.add_argument("--eval-final-win-rate", type=float, default=0.55)
    parser.add_argument("--eval-workers", type=int, default=64)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4_500)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--env-gomaxprocs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-seed", type=int, default=1_000_000)
    parser.add_argument("--eval-seed", type=int, default=100_000_000)
    parser.add_argument("--stop-win-rate", type=float, default=0.60)
    parser.add_argument("--stop-ci-low", type=float, default=0.50)
    parser.add_argument("--continue-after-target", action="store_true")
    parser.add_argument("--no-pipeline", action="store_true")
    run_campaign(parser.parse_args())


if __name__ == "__main__":
    main()
