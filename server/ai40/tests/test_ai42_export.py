from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

try:
    import onnx  # noqa: F401
    import onnxruntime as ort
except (ImportError, OSError):
    ort = None

from tanat_ai40.critic_ai42 import AI42CentralizedCritic
from tanat_ai40.env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
)
from tanat_ai40.model_ai42_actor import AI42Actor
from tanat_ai40.export_ai42 import (
    ACTOR_INPUT_NAMES,
    ACTOR_OUTPUT_NAMES,
    AI42ActorONNXWrapper,
    ONNXInterfaceError,
    compare_pytorch_onnx,
    export_ai42_actor,
    main as export_main,
    validate_onnx_interface,
)
from tanat_ai40.runtime_ai42 import AI42Manifest


def manifest(seed: str = "export") -> AI42Manifest:
    def digest(name: str) -> str:
        return hashlib.sha256(f"{name}-{seed}".encode()).hexdigest()

    return AI42Manifest(
        model_hash=digest("model"),
        config_hash=digest("config"),
        checkpoint_hash=digest("checkpoint"),
        observation_hash=digest("observation"),
        action_hash=digest("action"),
        trajectory_hash=digest("trajectory"),
    )


class _Value:
    def __init__(self, name: str):
        self.name = name


class _FakeSession:
    def __init__(self, wrapper: AI42ActorONNXWrapper):
        self.wrapper = wrapper

    def get_inputs(self):
        return [_Value(name) for name in ACTOR_INPUT_NAMES]

    def get_outputs(self):
        return [_Value(name) for name in ACTOR_OUTPUT_NAMES]

    def run(self, _output_names, feed):
        values = tuple(torch.from_numpy(feed[name]) for name in ACTOR_INPUT_NAMES)
        with torch.no_grad():
            return [value.detach().cpu().numpy() for value in self.wrapper(*values)]


class AI42ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.actor = AI42Actor(
            hidden_size=8,
            model_width=8,
            entity_layers=1,
            num_heads=2,
            ff_multiplier=1,
            timing_bins=2,
        ).eval()

    def inputs(self, batch: int = 2) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(17)
        return (
            torch.randn(batch, HERO_FEATURES),
            torch.randn(batch, ABILITY_COUNT, ABILITY_FEATURES),
            torch.randn(batch, MAX_ENTITIES, ENTITY_FEATURES),
            torch.randn(batch, GLOBAL_FEATURES),
            torch.ones(batch, MAX_ENTITIES, dtype=torch.bool),
            *self.actor.initial_state(batch, "cpu"),
        )

    def test_wrapper_has_explicit_actor_only_interface(self) -> None:
        wrapper = AI42ActorONNXWrapper(self.actor).eval()
        output = wrapper(*self.inputs(1))
        self.assertEqual(len(output), len(ACTOR_OUTPUT_NAMES))
        self.assertEqual([*ACTOR_INPUT_NAMES], [
            "hero", "abilities", "entities", "global_state", "entity_mask", "h", "c",
        ])
        self.assertEqual([*ACTOR_OUTPUT_NAMES], [
            "control", "kind", "target", "offset", "anchor", "timing", "timing_aux",
            "next_h", "next_c",
        ])
        self.assertEqual(tuple(output[0].shape), (1, 4))
        self.assertEqual(tuple(output[1].shape), (1, 8))
        self.assertEqual(tuple(output[-1].shape), (1, 8))

    def test_critic_cannot_be_exported(self) -> None:
        with self.assertRaises(TypeError):
            AI42ActorONNXWrapper(AI42CentralizedCritic(test_size=True))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            export_ai42_actor(AI42CentralizedCritic(test_size=True), "unused.onnx")  # type: ignore[arg-type]

    def test_export_passes_named_dynamic_batch_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "actor.onnx"
            with mock.patch("torch.onnx.export") as export:
                result = export_ai42_actor(self.actor, output, example_inputs=self.inputs(2))
            self.assertEqual(result.output_path, output)
            kwargs = export.call_args.kwargs
            self.assertEqual(kwargs["input_names"], list(ACTOR_INPUT_NAMES))
            self.assertEqual(kwargs["output_names"], list(ACTOR_OUTPUT_NAMES))
            self.assertEqual(kwargs["dynamic_axes"]["hero"], {0: "batch"})
            self.assertEqual(kwargs["dynamic_axes"]["next_h"], {0: "batch"})
            self.assertEqual(kwargs["opset_version"], 18)

    def test_manifest_sidecar_is_strict_and_export_is_atomic_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "actor.onnx"
            sidecar = root / "actor.manifest.json"
            output.write_bytes(b"existing-model")
            sidecar.write_text("existing-manifest", encoding="utf-8")

            def fail_export(_wrapper, _inputs, temporary_output, **_kwargs):
                Path(temporary_output).write_bytes(b"partial-model")
                raise RuntimeError("export failed")

            with mock.patch("torch.onnx.export", side_effect=fail_export):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    export_ai42_actor(
                        self.actor,
                        output,
                        example_inputs=self.inputs(1),
                        manifest=manifest(),
                        manifest_path=sidecar,
                    )
            self.assertEqual(output.read_bytes(), b"existing-model")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "existing-manifest")
            self.assertEqual(sorted(path.name for path in root.iterdir()), [
                "actor.manifest.json", "actor.onnx",
            ])

    def test_pair_replace_rolls_back_first_success_when_second_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "actor.onnx"
            sidecar = root / "actor.manifest.json"
            old_model = b"old-model-byte-for-byte"
            old_manifest = b"old-manifest-byte-for-byte"
            output.write_bytes(old_model)
            sidecar.write_bytes(old_manifest)

            def write_export(_wrapper, _inputs, temporary_output, **_kwargs):
                Path(temporary_output).write_bytes(b"new-model")

            real_replace = os.replace
            replace_calls = 0

            def fail_second_replace(source, destination):
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 2:
                    raise PermissionError("manifest target is locked")
                return real_replace(source, destination)

            with (
                mock.patch("torch.onnx.export", side_effect=write_export),
                mock.patch(
                    "tanat_ai40.export_ai42.os.replace",
                    side_effect=fail_second_replace,
                ),
            ):
                with self.assertRaisesRegex(PermissionError, "manifest target is locked"):
                    export_ai42_actor(
                        self.actor,
                        output,
                        example_inputs=self.inputs(1),
                        manifest=manifest(),
                        manifest_path=sidecar,
                    )

            self.assertEqual(replace_calls, 4)
            self.assertEqual(output.read_bytes(), old_model)
            self.assertEqual(sidecar.read_bytes(), old_manifest)
            self.assertEqual(sorted(path.name for path in root.iterdir()), [
                "actor.manifest.json", "actor.onnx",
            ])

    def test_manifest_output_requires_complete_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "actor.onnx"
            sidecar = root / "actor.json"
            with mock.patch("torch.onnx.export") as export:
                with self.assertRaisesRegex(ValueError, "requires a valid"):
                    export_ai42_actor(self.actor, output, manifest_path=sidecar)
                incomplete = manifest().to_dict()
                incomplete.pop("trajectory_hash")
                with self.assertRaisesRegex(ValueError, "missing"):
                    export_ai42_actor(self.actor, output, manifest=incomplete)
                invalid = manifest().to_dict()
                invalid["model_hash"] = "A" * 64
                with self.assertRaisesRegex(ValueError, "64 lowercase"):
                    export_ai42_actor(self.actor, output, manifest=invalid)
            export.assert_not_called()
            with self.assertRaisesRegex(ValueError, "requires a valid"):
                export_main([
                    "--output", str(output),
                    "--manifest-output", str(sidecar),
                ])
            invalid_file = root / "invalid-manifest.json"
            invalid_file.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                export_main([
                    "--output", str(output),
                    "--manifest", str(invalid_file),
                    "--manifest-output", str(sidecar),
                ])

    def test_valid_manifest_sidecar_uses_canonical_validated_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "actor.onnx"
            sidecar = root / "actor.json"
            expected = manifest()
            with mock.patch("torch.onnx.export"):
                export_ai42_actor(
                    self.actor,
                    output,
                    example_inputs=self.inputs(1),
                    manifest=expected.to_dict(),
                    manifest_path=sidecar,
                )
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest"], expected.to_dict())
            self.assertEqual(payload["interface"]["dynamic_axes"]["hero"], {"0": "batch"})

    def test_parity_helper_checks_outputs_and_recurrent_transition(self) -> None:
        wrapper = AI42ActorONNXWrapper(self.actor).eval()
        report = compare_pytorch_onnx(self.actor, _FakeSession(wrapper), self.inputs())
        self.assertTrue(report.passed)
        self.assertTrue(report.recurrent_transition_passed)
        self.assertEqual(report.max_abs_error, 0.0)

    @unittest.skipUnless(
        ort is not None,
        "onnx and onnxruntime are required for real export parity",
    )
    def test_real_onnx_export_dynamic_batch_and_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "actor.onnx"
            export_ai42_actor(self.actor, output, example_inputs=self.inputs(1))
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            validate_onnx_interface(session)
            for batch in (1, 3):
                report = compare_pytorch_onnx(
                    self.actor,
                    session,
                    self.inputs(batch),
                    rtol=2e-4,
                    atol=2e-5,
                )
                self.assertTrue(report.passed, report.to_dict())
                self.assertTrue(report.recurrent_transition_passed)

    def test_interface_validation_rejects_non_actor_names(self) -> None:
        class BadSession(_FakeSession):
            def get_outputs(self):
                return [_Value("value")]

        with self.assertRaises(ONNXInterfaceError):
            validate_onnx_interface(BadSession(AI42ActorONNXWrapper(self.actor)))


if __name__ == "__main__":
    unittest.main()
