from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _bash_executable():
    candidates = (
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")


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

    def test_bc_preflight_wrappers_harden_control_flag_boundary(self) -> None:
        for name in ("run-ai42-bc-preflight.ps1", "run-ai42-bc-preflight.sh"):
            wrapper = (ROOT.parent / name).read_text(encoding="utf-8")
            self.assertIn("./cmd/ai42preflight", wrapper)
            self.assertIn("torch_probe_worker_ai42", wrapper)
            self.assertNotIn("preflight_torch_ai42", wrapper)
            for flag in ("--torch-python", "--worker-command", "--worker-arg", "--worker-timeout"):
                self.assertIn(flag, wrapper)
            self.assertIn("Duplicate", wrapper)
            self.assertIn("5m", wrapper)
            self.assertNotIn("TANAT_AI42_PREFLIGHT_TIMEOUT", wrapper)
            self.assertNotIn("--worker-command=", wrapper)
            self.assertNotIn("--worker-arg=", wrapper)
            self.assertIn("--worker-timeout", wrapper)
            self.assertNotIn("train_ai42_bc", wrapper)

    def test_bc_preflight_wrappers_use_fixed_worker_flags_only(self) -> None:
        powershell = (ROOT.parent / "run-ai42-bc-preflight.ps1").read_text(encoding="utf-8")
        powershell_invocation = powershell[powershell.index("$nativeArgs =") :]
        self.assertIn('"--torch-python", $python', powershell_invocation)
        self.assertIn('"--worker-timeout", $workerTimeout', powershell_invocation)
        self.assertNotIn("--worker-command", powershell_invocation)
        self.assertNotIn("--worker-arg", powershell_invocation)

        bash = (ROOT.parent / "run-ai42-bc-preflight.sh").read_text(encoding="utf-8")
        bash_invocation = bash[bash.index("native_args+=(") :]
        self.assertIn('--torch-python "$PYTHON_BIN"', bash_invocation)
        self.assertIn('--worker-timeout "$WORKER_TIMEOUT"', bash_invocation)
        self.assertNotIn("--worker-command", bash_invocation)
        self.assertNotIn("--worker-arg", bash_invocation)

    @unittest.skipUnless(_bash_executable(), "bash is required for shell syntax validation")
    def test_bash_wrapper_syntax(self) -> None:
        bash = _bash_executable()
        subprocess.run(
            [bash, "-n", str(ROOT.parent / "run-ai42-bc-preflight.sh")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(
        shutil.which("pwsh") or shutil.which("powershell"),
        "PowerShell is required for wrapper syntax validation",
    )
    def test_powershell_wrapper_syntax(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        path = str(ROOT.parent / "run-ai42-bc-preflight.ps1").replace("'", "''")
        command = (
            "$tokens = $null; $errors = $null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', "
            "[ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_preflight_docs_pin_profile_hash_and_report_scope(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--profile-hash <lower-case-profile-sha256>", readme)
        self.assertIn("`--report` must\nbe located under `--output`", readme)
        self.assertIn("exact selected\ninterpreter as `--torch-python`", readme)
        self.assertIn("full durable-data verification and report publication", readme)

    def test_python_entrypoints_keep_smoke_separate_from_native_preflight(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("tanat-ai42-actor-smoke", pyproject)
        self.assertNotIn("tanat-ai42-preflight", pyproject)
        self.assertNotIn("tanat-ai42-bc-preflight", pyproject)
        self.assertIn("tanat_ai40.smoke_ai42_actor:main", pyproject)


if __name__ == "__main__":
    unittest.main()
