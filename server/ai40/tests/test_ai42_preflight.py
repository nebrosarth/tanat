from __future__ import annotations

import unittest

from tanat_ai40.preflight_ai42 import run_preflight


class AI42PreflightTests(unittest.TestCase):
    def test_cpu_preflight_checks_backward_without_updating_parameters(self) -> None:
        report = run_preflight(
            device_name="cpu",
            batch_size=2,
            iterations=1,
            hidden_size=32,
            model_width=32,
            entity_layers=1,
            num_heads=4,
        )

        self.assertTrue(report.ok)
        self.assertTrue(report.backward_checked)
        self.assertTrue(report.finite_outputs)
        self.assertTrue(report.finite_gradients)
        self.assertTrue(report.parameters_unchanged)
        self.assertEqual(report.peak_allocated_bytes, 0)
        self.assertGreater(report.forward_steps_per_second, 0)

    def test_preflight_rejects_non_positive_work(self) -> None:
        for batch_size, iterations in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(batch_size=batch_size, iterations=iterations):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    run_preflight(
                        device_name="cpu",
                        batch_size=batch_size,
                        iterations=iterations,
                        backward=False,
                        hidden_size=16,
                        model_width=16,
                        entity_layers=1,
                        num_heads=4,
                    )


if __name__ == "__main__":
    unittest.main()
