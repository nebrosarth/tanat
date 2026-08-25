from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AI42BCV2StaticTests(unittest.TestCase):
    def test_lineage_configs_freeze_half_power(self) -> None:
        for name in ("ai42_bc_preflight.json", "ai42_bc_training.json"):
            payload = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["learner"]["class_balance_power"], 0.5)

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
