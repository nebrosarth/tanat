from __future__ import annotations

import unittest

from tanat_ai40.evaluate_ai42_bc_checkpoints import build_parser, curve_summary


def _point(step: int, loss: float, end_to_end: float) -> dict:
    return {
        "path": f"step-{step}.pt",
        "step": step,
        "epoch": 0,
        "validation": {
            "loss": loss,
            "end_to_end_accuracy": end_to_end,
        },
    }


class CheckpointCurveTests(unittest.TestCase):
    def test_curve_is_sorted_and_selects_lowest_loss(self) -> None:
        result = curve_summary(
            [_point(300, 2.2, 0.05), _point(100, 3.0, 0.0), _point(200, 2.0, 0.1)],
            1e-4,
        )

        self.assertEqual(result["ordered_steps"], [100, 200, 300])
        self.assertFalse(result["loss_non_increasing"])
        self.assertFalse(result["end_to_end_non_decreasing"])
        self.assertEqual(result["best"]["step"], 200)
        self.assertAlmostEqual(result["loss_regressions"][0]["delta"], 0.2)
        self.assertAlmostEqual(result["end_to_end_regressions"][0]["delta"], -0.05)

    def test_epsilon_ignores_numerical_noise_and_tie_prefers_later_step(self) -> None:
        result = curve_summary(
            [_point(100, 2.0, 0.1), _point(200, 2.0, 0.09995)],
            1e-4,
        )

        self.assertTrue(result["loss_non_increasing"])
        self.assertTrue(result["end_to_end_non_decreasing"])
        self.assertEqual(result["best"]["step"], 200)

    def test_rejects_invalid_epsilon(self) -> None:
        with self.assertRaisesRegex(Exception, "epsilon"):
            curve_summary([_point(100, 2.0, 0.1)], -1.0)

    def test_patience_reports_when_best_checkpoint_is_stale(self) -> None:
        result = curve_summary(
            [_point(100, 1.0, 0.1), _point(200, 1.1, 0.1), _point(300, 1.2, 0.1)],
            1e-4,
            patience=2,
        )

        self.assertEqual(result["checkpoints_since_best"], 2)
        self.assertTrue(result["early_stop_ready"])

    def test_evaluation_batch_size_is_explicit(self) -> None:
        args = build_parser().parse_args([
            "--config", "config.json", "--dataset", "dataset", "--profile", "profile.json",
            "--checkpoint", "step.pt", "--output", "curve.json", "--evaluation-batch-size", "64",
        ])

        self.assertEqual(args.evaluation_batch_size, 64)


if __name__ == "__main__":
    unittest.main()
