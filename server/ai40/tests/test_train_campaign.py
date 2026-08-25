from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from tanat_ai40.train_campaign import (
    evaluate_promotion_pair,
    promotion_is_safe,
    synchronize_evaluation_thresholds,
    synchronize_target_state,
)


class CampaignTargetStateTest(unittest.TestCase):
    def test_raising_target_clears_legacy_target_reached_flag(self):
        state = {
            "completed_stages": 5,
            "target_reached": True,
            "target_reached_stage": 5,
            "last_evaluation": {
                "win_rate": 0.668,
                "win_rate_ci95_low": 0.625,
            },
        }
        args = SimpleNamespace(stop_win_rate=0.70, stop_ci_low=0.50)

        self.assertTrue(synchronize_target_state(state, args))
        self.assertFalse(state.get("target_reached", False))
        self.assertNotIn("target_reached_stage", state)
        self.assertEqual(state["stop_win_rate"], 0.70)
        self.assertEqual(state["stop_ci_low"], 0.50)

    def test_changed_quick_gate_is_recorded_for_the_next_stage(self):
        state = {
            "completed_stages": 11,
            "eval_medium_win_rate": 0.40,
            "eval_final_win_rate": 0.55,
        }
        args = SimpleNamespace(eval_medium_win_rate=0.70, eval_final_win_rate=0.99)

        self.assertTrue(synchronize_evaluation_thresholds(state, args))
        self.assertEqual(state["eval_medium_win_rate"], 0.70)
        self.assertEqual(state["eval_final_win_rate"], 0.99)
        self.assertEqual(state["evaluation_threshold_history"][0]["applied_before_stage"], 12)

    def test_promotion_rejects_composite_or_large_category_regression(self):
        reference = {"categories": {
            "ai30": {"score_rate": 0.60},
            "stage005": {"score_rate": 0.55},
            "historical_pool": {"available": False},
        }}
        improved = {"categories": {
            "ai30": {"score_rate": 0.62},
            "stage005": {"score_rate": 0.56},
            "historical_pool": {"available": False},
        }}
        accepted, reasons = promotion_is_safe(improved, reference, 0.0, 0.05)
        self.assertTrue(accepted, reasons)
        regressed = {"categories": {
            "ai30": {"score_rate": 0.65},
            "stage005": {"score_rate": 0.45},
            "historical_pool": {"available": False},
        }}
        accepted, reasons = promotion_is_safe(regressed, reference, 0.0, 0.05)
        self.assertFalse(accepted)
        self.assertIn("stage005 score regressed", reasons)

    def test_promotion_pair_skips_remaining_categories_after_hard_ai30_regression(self):
        args = SimpleNamespace(
            promotion_eval_matches=50,
            eval_workers=50,
            max_steps=4500,
            promotion_max_category_regression=0.05,
        )
        calls: list[str] = []

        def ai30_evaluator(checkpoint, *_args):
            calls.append(checkpoint.name)
            score = 0.50 if checkpoint.name == "candidate.pt" else 0.70
            return {
                "matches": 50, "wins": int(score * 50), "losses": 0,
                "draws": 50 - int(score * 50), "win_rate": score,
                "score_rate": score, "elapsed_seconds": 1.0,
            }

        def checkpoint_evaluator(*_args, **_kwargs):
            self.fail("stage-005 and historical pool must be skipped after AI-30 rejection")

        candidate, reference, reasons = evaluate_promotion_pair(
            Path("candidate.pt"), Path("promoted.pt"), Path("anchor.pt"),
            [Path("anchor.pt"), Path("past.pt")], Path("assaultenv.exe"),
            "cpu", args, 900_000_000, ai30_evaluator, checkpoint_evaluator,
        )
        self.assertEqual(calls, ["candidate.pt", "promoted.pt"])
        self.assertEqual(reasons, ["ai30 score regressed"])
        self.assertFalse(candidate["complete"])
        self.assertTrue(candidate["categories"]["stage005"]["skipped"])
        self.assertTrue(reference["categories"]["historical_pool"]["skipped"])


if __name__ == "__main__":
    unittest.main()
