"""Actor-only ONNX export and PyTorch/ONNX parity helpers for AI-42.

The interface in this module is intentionally explicit.  It contains no
training outputs and accepts only the observation and recurrent tensors that
the actor consumes.  ONNX Runtime is optional; parity tests can provide a
small fake session implementing ``run``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import os
import shutil
import tempfile

import numpy as np
import torch
from torch import nn

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
)
from .model_ai42_actor import AI42Actor
from .runtime_ai42 import (
    ACTOR_INPUT_NAMES,
    ACTOR_OUTPUT_NAMES,
    AI42Manifest,
    CheckpointCompatibilityError,
    load_compatible_checkpoint,
)


@dataclass(frozen=True, slots=True)
class AI42ONNXInterface:
    input_names: tuple[str, ...] = ACTOR_INPUT_NAMES
    output_names: tuple[str, ...] = ACTOR_OUTPUT_NAMES
    opset_version: int = 18

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            name: {0: "batch"}
            for name in self.input_names + self.output_names
        }


INTERFACE = AI42ONNXInterface()


class AI42ActorONNXWrapper(nn.Module):
    """Flatten the actor mapping into a stable, named ONNX tuple."""

    def __init__(self, actor: AI42Actor):
        super().__init__()
        _require_actor_only(actor)
        self.actor = actor

    def forward(
        self,
        hero: torch.Tensor,
        abilities: torch.Tensor,
        entities: torch.Tensor,
        global_state: torch.Tensor,
        entity_mask: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output = self.actor(hero, abilities, entities, global_state, entity_mask, h, c)
        return (
            output["kind"],
            output["target"],
            output["offset"],
            output["anchor"],
            output["timing"],
            output["timing_aux"],
            output["h"],
            output["c"],
        )


ActorONNXWrapper = AI42ActorONNXWrapper


@dataclass(frozen=True, slots=True)
class AI42ExportResult:
    output_path: Path
    interface: AI42ONNXInterface
    manifest_path: Path | None = None


class ONNXInterfaceError(ValueError):
    pass


class ONNXParityError(AssertionError):
    def __init__(self, report: "ONNXParityReport"):
        self.report = report
        super().__init__(report.reason or "PyTorch and ONNX actor outputs differ")


@dataclass(frozen=True, slots=True)
class ONNXParityReport:
    passed: bool
    per_output: Mapping[str, Mapping[str, float | bool | str]]
    recurrent_transition_passed: bool
    reason: str | None = None

    @property
    def max_abs_error(self) -> float:
        return max(
            (float(item.get("max_abs", 0.0)) for item in self.per_output.values()),
            default=0.0,
        )

    @property
    def max_relative_error(self) -> float:
        return max(
            (float(item.get("max_rel", 0.0)) for item in self.per_output.values()),
            default=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "per_output": {key: dict(value) for key, value in self.per_output.items()},
            "recurrent_transition_passed": self.recurrent_transition_passed,
            "reason": self.reason,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
        }


def _require_actor_only(actor: Any) -> None:
    if not isinstance(actor, AI42Actor):
        raise TypeError("AI-42 export accepts only an AI42Actor instance")
    if getattr(actor, "has_value_head", True):
        raise TypeError("AI-42 export accepts only an actor without value outputs")


def _default_inputs(actor: AI42Actor, batch: int = 1) -> tuple[torch.Tensor, ...]:
    if batch < 1:
        raise ValueError("batch must be positive")
    try:
        parameter = next(actor.parameters())
        device, dtype = parameter.device, parameter.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    hero = torch.zeros((batch, HERO_FEATURES), device=device, dtype=dtype)
    abilities = torch.zeros((batch, ABILITY_COUNT, ABILITY_FEATURES), device=device, dtype=dtype)
    entities = torch.zeros((batch, MAX_ENTITIES, ENTITY_FEATURES), device=device, dtype=dtype)
    global_state = torch.zeros((batch, GLOBAL_FEATURES), device=device, dtype=dtype)
    entity_mask = torch.ones((batch, MAX_ENTITIES), device=device, dtype=torch.bool)
    h, c = actor.initial_state(batch, device=device, dtype=dtype)
    return hero, abilities, entities, global_state, entity_mask, h, c


def _input_tuple(inputs: Mapping[str, Any] | Sequence[Any]) -> tuple[torch.Tensor, ...]:
    if isinstance(inputs, Mapping):
        missing = set(ACTOR_INPUT_NAMES) - set(inputs)
        extra = set(inputs) - set(ACTOR_INPUT_NAMES)
        if missing or extra:
            raise ONNXInterfaceError(f"input schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        values = tuple(inputs[name] for name in ACTOR_INPUT_NAMES)
    else:
        if len(inputs) != len(ACTOR_INPUT_NAMES):
            raise ONNXInterfaceError(f"expected {len(ACTOR_INPUT_NAMES)} ONNX inputs")
        values = tuple(inputs)
    return tuple(value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values)


def _validated_manifest(
    manifest: AI42Manifest | Mapping[str, Any] | None,
) -> AI42Manifest | None:
    if manifest is None:
        return None
    if isinstance(manifest, AI42Manifest):
        # Re-decode to retain strictness even for unusually constructed
        # instances that bypassed the dataclass initializer.
        return AI42Manifest.from_dict(manifest.to_dict())
    if isinstance(manifest, Mapping):
        return AI42Manifest.from_dict(manifest)
    raise TypeError("manifest must be an AI42Manifest or a complete strict mapping")


def _temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=f".tmp{destination.suffix}",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _replace_transactionally(artifacts: Sequence[tuple[Path, Path]]) -> None:
    """Replace an artifact set and roll the entire set back on any failure."""

    backups: list[tuple[Path, Path | None]] = []
    try:
        # Backups are byte copies rather than moves, so all old targets remain
        # available until commit starts. Each backup lives beside its target.
        for _staged, target in artifacts:
            backup: Path | None = None
            if target.exists():
                backup = _temporary_path(target)
                shutil.copyfile(target, backup)
            backups.append((target, backup))

        try:
            for staged, target in artifacts:
                os.replace(staged, target)
        except Exception as original_error:
            restoration_errors: list[str] = []
            for target, backup in backups:
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                except Exception as restore_error:  # pragma: no cover - catastrophic filesystem failure
                    restoration_errors.append(f"{target}: {restore_error}")
            for message in restoration_errors:
                original_error.add_note(f"AI-42 artifact rollback failed for {message}")
            raise
    finally:
        for _target, backup in backups:
            if backup is not None:
                backup.unlink(missing_ok=True)


def export_ai42_actor(
    actor: AI42Actor,
    output_path: str | Path,
    *,
    example_inputs: Mapping[str, Any] | Sequence[Any] | None = None,
    opset_version: int = 18,
    manifest: AI42Manifest | Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> AI42ExportResult:
    """Export only the AI42Actor with dynamic batch axes."""

    _require_actor_only(actor)
    if opset_version < 18:
        raise ValueError("AI-42 ONNX export requires opset 18 or newer")
    validated_manifest = _validated_manifest(manifest)
    if manifest_path is not None and validated_manifest is None:
        raise ValueError("manifest_path requires a valid AI42Manifest")
    output = Path(output_path)
    sidecar = Path(manifest_path) if manifest_path is not None else None
    if sidecar is not None and output.resolve() == sidecar.resolve():
        raise ValueError("ONNX output and manifest output must be different paths")
    inputs = _default_inputs(actor) if example_inputs is None else _input_tuple(example_inputs)
    if inputs[0].shape[0] < 1:
        raise ValueError("example input batch must be positive")
    interface = AI42ONNXInterface(opset_version=opset_version)
    wrapper = AI42ActorONNXWrapper(actor).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = None
    if validated_manifest is not None:
        payload = {
            "manifest": validated_manifest.to_dict(),
            "interface": {
                "inputs": list(interface.input_names),
                "outputs": list(interface.output_names),
                "dynamic_axes": interface.dynamic_axes,
                "opset_version": opset_version,
            },
        }
    temporary_output = _temporary_path(output)
    temporary_sidecar = _temporary_path(sidecar) if sidecar is not None else None
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                str(temporary_output),
                input_names=list(interface.input_names),
                output_names=list(interface.output_names),
                dynamic_axes=interface.dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
        if temporary_sidecar is not None:
            temporary_sidecar.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
        artifacts = [(temporary_output, output)]
        if temporary_sidecar is not None and sidecar is not None:
            artifacts.append((temporary_sidecar, sidecar))
        _replace_transactionally(artifacts)
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)
    return AI42ExportResult(output, interface, sidecar)


export_actor_onnx = export_ai42_actor
export_ai42_onnx = export_ai42_actor
export_ai42_actor_onnx = export_ai42_actor


def validate_onnx_interface(session: Any) -> None:
    """Reject an ONNX session whose names are not the actor-only contract."""

    if not hasattr(session, "get_inputs") or not hasattr(session, "get_outputs"):
        raise ONNXInterfaceError("session must expose get_inputs/get_outputs")
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if input_names != list(ACTOR_INPUT_NAMES):
        raise ONNXInterfaceError(f"ONNX input names {input_names} != {list(ACTOR_INPUT_NAMES)}")
    if output_names != list(ACTOR_OUTPUT_NAMES):
        raise ONNXInterfaceError(f"ONNX output names {output_names} != {list(ACTOR_OUTPUT_NAMES)}")


def _feed_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(ACTOR_INPUT_NAMES, inputs)
    }


def compare_pytorch_onnx(
    actor: AI42Actor,
    session: Any,
    inputs: Mapping[str, Any] | Sequence[Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ONNXParityReport:
    """Compare all actor outputs, including the recurrent transition."""

    _require_actor_only(actor)
    try:
        # A real session is validated by names.  Lightweight parity fixtures
        # are intentionally allowed to implement only ``run``.
        if hasattr(session, "get_inputs") or hasattr(session, "get_outputs"):
            validate_onnx_interface(session)
        values = _input_tuple(inputs)
        with torch.no_grad():
            torch_output = AI42ActorONNXWrapper(actor)(*values)
        ort_output = session.run(None, _feed_inputs(values))
        if len(ort_output) != len(ACTOR_OUTPUT_NAMES):
            raise ONNXInterfaceError(f"ONNX returned {len(ort_output)} outputs")
        report: dict[str, dict[str, float | bool | str]] = {}
        for name, expected, actual in zip(ACTOR_OUTPUT_NAMES, torch_output, ort_output):
            expected_np = expected.detach().cpu().numpy()
            actual_np = np.asarray(actual)
            if expected_np.shape != actual_np.shape:
                report[name] = {
                    "passed": False,
                    "max_abs": float("inf"),
                    "max_rel": float("inf"),
                    "reason": f"shape {actual_np.shape} != {expected_np.shape}",
                }
                continue
            if not np.isfinite(actual_np).all():
                report[name] = {
                    "passed": False,
                    "max_abs": float("inf"),
                    "max_rel": float("inf"),
                    "reason": "non-finite ONNX output",
                }
                continue
            difference = np.abs(expected_np - actual_np)
            denominator = np.maximum(np.abs(expected_np), np.abs(actual_np))
            relative = difference / np.maximum(denominator, np.finfo(np.float32).tiny)
            report[name] = {
                "passed": bool(np.allclose(expected_np, actual_np, rtol=rtol, atol=atol)),
                "max_abs": float(difference.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
            }
        passed = all(bool(item["passed"]) for item in report.values())
        recurrent = bool(report["next_h"]["passed"] and report["next_c"]["passed"])
        reason = None if passed else "one or more actor outputs exceeded parity tolerance"
        return ONNXParityReport(passed, report, recurrent, reason)
    except (ONNXInterfaceError, RuntimeError, ValueError) as exc:
        return ONNXParityReport(False, {}, False, str(exc))


parity_check = compare_pytorch_onnx
compare_pytorch_to_onnx = compare_pytorch_onnx


def assert_onnx_parity(
    actor: AI42Actor,
    session: Any,
    inputs: Mapping[str, Any] | Sequence[Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ONNXParityReport:
    report = compare_pytorch_onnx(actor, session, inputs, rtol=rtol, atol=atol)
    if not report.passed:
        raise ONNXParityError(report)
    return report


def _manifest_from_path(path: Path) -> AI42Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "manifest" in payload:
        payload = payload["manifest"]
    return AI42Manifest.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an actor-only AI-42 ONNX model")
    parser.add_argument("checkpoint", type=Path, nargs="?", help="exact AI-42 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="expected manifest JSON")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--entity-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--timing-bins", type=int, default=4)
    parser.add_argument("--opset", type=int, default=18)
    return parser


def main(argv: Sequence[str] | None = None) -> AI42ExportResult:
    args = build_parser().parse_args(argv)
    if args.manifest_output is not None and args.manifest is None:
        raise ValueError("--manifest-output requires a valid --manifest")
    expected = _manifest_from_path(args.manifest) if args.manifest else None
    actor = AI42Actor(
        hidden_size=args.hidden_size,
        model_width=args.model_width,
        entity_layers=args.entity_layers,
        num_heads=args.num_heads,
        timing_bins=args.timing_bins,
    )
    actor.eval()
    if args.checkpoint is not None:
        if expected is None:
            raise CheckpointCompatibilityError(
                # This report is never loaded; it gives the CLI a clear error
                # without inventing a compatibility policy.
                checkpoint_compatibility_report_for_cli(),
            )
        load_compatible_checkpoint(actor, args.checkpoint, expected)
    result = export_ai42_actor(
        actor,
        args.output,
        opset_version=args.opset,
        manifest=expected,
        manifest_path=args.manifest_output,
    )
    print(result.output_path)
    return result


def checkpoint_compatibility_report_for_cli():
    """Construct a typed report for the CLI's missing-manifest diagnostic."""

    from .runtime_ai42 import CheckpointCompatibilityReport

    return CheckpointCompatibilityReport(
        False,
        mismatched=("manifest:missing_expected_manifest",),
        reason="--manifest is required when exporting a checkpoint",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ACTOR_INPUT_NAMES",
    "ACTOR_OUTPUT_NAMES",
    "AI42ActorONNXWrapper",
    "AI42ExportResult",
    "AI42ONNXInterface",
    "ActorONNXWrapper",
    "INTERFACE",
    "ONNXInterfaceError",
    "ONNXParityError",
    "ONNXParityReport",
    "assert_onnx_parity",
    "compare_pytorch_onnx",
    "compare_pytorch_to_onnx",
    "export_actor_onnx",
    "export_ai42_actor",
    "export_ai42_actor_onnx",
    "export_ai42_onnx",
    "main",
    "parity_check",
    "validate_onnx_interface",
]
