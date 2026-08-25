from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AI42BCV2StaticTests(unittest.TestCase):
    def test_lineage_configs_freeze_half_power(self) -> None:
        for name in ("ai42_bc_preflight.json", "ai42_bc_training.json", "ai42_bc_training_q3.json"):
            payload = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["learner"]["class_balance_power"], 0.5)

    def test_q3_lineage_config_keeps_q2_settings_except_authorized_change(self) -> None:
        q2 = json.loads((ROOT / "config" / "ai42_bc_training.json").read_text(encoding="utf-8"))
        q3 = json.loads((ROOT / "config" / "ai42_bc_training_q3.json").read_text(encoding="utf-8"))
        self.assertEqual(q3["model"], q2["model"])
        self.assertEqual(q3["recurrent_batch"], q2["recurrent_batch"])
        self.assertEqual(q3["training"], q2["training"])
        self.assertEqual(q3["learner"]["weight_decay"], q2["learner"]["weight_decay"])
        self.assertEqual(q3["learner"]["class_balance_power"], q2["learner"]["class_balance_power"])
        self.assertEqual(q3["learner"]["max_gradient_norm"], q2["learner"]["max_gradient_norm"])
        self.assertEqual(q3["learner"]["learning_rate"], 0.0001)
        self.assertEqual(
            q3["learner"]["class_weight_overrides"]["control"],
            [0.76592687, 0.68894406, 0.07211628, 2.47301279],
        )

    def test_override_provenance_is_present_in_production_source(self) -> None:
        trainer = (ROOT / "src" / "tanat_ai40" / "train_ai42_bc.py").read_text(encoding="utf-8")
        for field in (
            "class_weight_overrides",
            "class_weight_overrides_hash",
            "class_weights",
            "class_weight_provenance",
        ):
            self.assertIn(field, trainer)

    def test_production_modules_expose_required_boundaries(self) -> None:
        learner = (ROOT / "src" / "tanat_ai40" / "learner_ai42.py").read_text(encoding="utf-8")
        profile = (ROOT / "src" / "tanat_ai40" / "bc_profile_ai42.py").read_text(encoding="utf-8")
        metrics = (ROOT / "src" / "tanat_ai40" / "bc_metrics_ai42.py").read_text(encoding="utf-8")
        trainer = (ROOT / "src" / "tanat_ai40" / "train_ai42_bc.py").read_text(encoding="utf-8")
        self.assertIn("def prepare_ai42_supervision", learner)
        self.assertIn("class AI42ClassBalanceProfile", profile)
        self.assertIn("class AI42MetricAccumulator", metrics)
        self.assertIn("def load_ai42_model_warm_start", learner)
        self.assertIn("def _validate_gate_metric_schema", trainer)
        self.assertIn('"metrics_complete": False', trainer)
        self.assertNotIn("baseline.head_accuracies.get", trainer)
        self.assertIn("profile_path.is_file()", trainer)

    def test_modified_python_sources_parse_without_importing_torch(self) -> None:
        for path in (
            ROOT / "src" / "tanat_ai40" / "learner_ai42.py",
            ROOT / "src" / "tanat_ai40" / "bc_profile_ai42.py",
            ROOT / "src" / "tanat_ai40" / "bc_metrics_ai42.py",
            ROOT / "src" / "tanat_ai40" / "train_ai42_bc.py",
            ROOT / "tests" / "test_ai42_bc_v2.py",
        ):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
