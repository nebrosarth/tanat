from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from tanat_ai40 import smoke_ai42_actor as smoke
from tanat_ai40.model_ai42_actor import AI42Actor
from tanat_ai40.smoke_ai42_actor import run_smoke


class AI42ActorSmokeTests(unittest.TestCase):
    @staticmethod
    def _small_config() -> dict[str, object]:
        return {
            "device_name": "cpu",
            "batch_size": 1,
            "iterations": 1,
            "backward": False,
            "hidden_size": 16,
            "model_width": 16,
            "entity_layers": 1,
            "num_heads": 4,
        }

    def test_cpu_smoke_checks_backward_without_updating_parameters(self) -> None:
        report = run_smoke(
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

    def test_smoke_accepts_safe_boundaries(self) -> None:
        cases = (
            {"batch_size": smoke.MAX_BATCH_SIZE},
            {"iterations": smoke.MAX_ITERATIONS},
            {"hidden_size": 1, "model_width": 1, "num_heads": 1},
            {"model_width": 64, "num_heads": smoke.MAX_NUM_HEADS},
            {"entity_layers": smoke.MAX_ENTITY_LAYERS},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                config = self._small_config()
                config.update(overrides)
                report = run_smoke(**config)
                self.assertTrue(report.ok)

    def test_smoke_rejects_oversize_values_before_actor_allocation(self) -> None:
        cases = (
            ("batch_size", smoke.MAX_BATCH_SIZE + 1),
            ("iterations", smoke.MAX_ITERATIONS + 1),
            ("hidden_size", smoke.MAX_HIDDEN_SIZE + 1),
            ("model_width", smoke.MAX_MODEL_WIDTH + 1),
            ("entity_layers", smoke.MAX_ENTITY_LAYERS + 1),
            ("num_heads", smoke.MAX_NUM_HEADS + 1),
        )
        for name, value in cases:
            with self.subTest(name=name):
                config = self._small_config()
                config[name] = value
                with patch.object(smoke, "AI42Actor") as actor:
                    with self.assertRaisesRegex(ValueError, "exceeds maximum"):
                        run_smoke(**config)
                    actor.assert_not_called()

    def test_smoke_rejects_invalid_types_and_nonfinite_numbers(self) -> None:
        cases = (
            ("batch_size", True),
            ("iterations", 1.0),
            ("hidden_size", float("nan")),
            ("model_width", float("inf")),
            ("entity_layers", "1"),
            ("num_heads", False),
        )
        for name, value in cases:
            with self.subTest(name=name):
                config = self._small_config()
                config[name] = value
                with self.assertRaises(ValueError):
                    run_smoke(**config)

        config = self._small_config()
        config["backward"] = 1
        with self.assertRaisesRegex(ValueError, "backward must be a bool"):
            run_smoke(**config)

    def test_smoke_rejects_divisibility_and_resource_limits_before_allocation(self) -> None:
        cases = (
            (
                {"model_width": 31, "num_heads": 4},
                "model_width must be divisible",
            ),
            (
                {
                    "hidden_size": smoke.MAX_HIDDEN_SIZE,
                    "model_width": smoke.MAX_MODEL_WIDTH,
                    "num_heads": 1,
                },
                "estimated model parameters",
            ),
            (
                {
                    "batch_size": smoke.MAX_BATCH_SIZE,
                    "iterations": smoke.MAX_ITERATIONS,
                    "hidden_size": 128,
                    "model_width": 128,
                    "entity_layers": 4,
                    "num_heads": 4,
                },
                "estimated model work",
            ),
            (
                {
                    "batch_size": smoke.MAX_BATCH_SIZE,
                    "iterations": 1,
                    "hidden_size": 384,
                    "model_width": 384,
                    "entity_layers": 4,
                    "num_heads": 8,
                    "backward": True,
                },
                "estimated model memory",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                config = self._small_config()
                config.update(overrides)
                with patch.object(smoke, "AI42Actor") as actor:
                    with self.assertRaisesRegex(ValueError, message):
                        run_smoke(**config)
                    actor.assert_not_called()

    def test_smoke_checks_shape_for_every_actor_output(self) -> None:
        real_forward = AI42Actor.forward
        output_names = (
            "control", "kind", "target", "offset", "anchor",
            "direction", "distance", "h", "c",
        )
        for name in output_names:
            with self.subTest(name=name):
                def malformed_forward(self, *args, output_name=name, **kwargs):
                    output = real_forward(self, *args, **kwargs)
                    malformed = dict(output)
                    malformed[output_name] = output[output_name][..., :-1]
                    return malformed

                config = self._small_config()
                with patch.object(AI42Actor, "forward", new=malformed_forward):
                    with self.assertRaisesRegex(
                        RuntimeError, f"actor output {name} has shape",
                    ):
                        run_smoke(**config)

    def test_smoke_reports_nonfinite_direction_and_distance(self) -> None:
        real_forward = AI42Actor.forward

        def nonfinite_forward(self, *args, **kwargs):
            output = real_forward(self, *args, **kwargs)
            result = dict(output)
            result["direction"] = output["direction"].clone()
            result["distance"] = output["distance"].clone()
            result["direction"][0, 0, 0] = torch.nan
            result["distance"][0, 0, 0] = torch.inf
            return result

        config = self._small_config()
        with patch.object(AI42Actor, "forward", new=nonfinite_forward):
            report = run_smoke(**config)

        self.assertFalse(report.ok)
        self.assertFalse(report.finite_outputs)
        self.assertTrue(report.parameters_unchanged)

    def test_smoke_rejects_non_positive_work(self) -> None:
        for batch_size, iterations in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(batch_size=batch_size, iterations=iterations):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    run_smoke(
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
