"""Bounded AI-42 actor Torch smoke test that never performs an optimizer update.

The command is intentionally separate from every training entry point. It is
a diagnostic smoke test, not the production preflight; Go owns that boundary.
It validates the installed PyTorch runtime and the real actor configuration,
measures inference/backward resource use, checks finite outputs/gradients, and
proves that the model parameters remain byte-identical throughout the run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    AI42_PROTOCOL_VERSION,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)
from .model_ai42_actor import (
    AI42Actor,
    CONTROL_CLASSES,
    CONTROL_CONTINUATION_CLASSES,
    KIND_GROUP_CLASSES,
    NAVIGATION_GRID_SIZE,
    parameter_count,
)
from .trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH


# These limits apply to the standalone diagnostic command.  They are kept
# explicit here because this entry point can be called directly from Python,
# bypassing argparse's integer conversion and any external preflight.
MAX_BATCH_SIZE = 64
MAX_ITERATIONS = 100
MAX_HIDDEN_SIZE = 4096
MAX_MODEL_WIDTH = 4096
MAX_ENTITY_LAYERS = 64
MAX_NUM_HEADS = 64
MAX_MODEL_PARAMETERS = 16_000_000
MAX_MODEL_WORK = 8_000_000_000
MAX_MODEL_MEMORY_BYTES = 512 * 1024 * 1024

_MODEL_PARAMETER_BYTES = 4
_MODEL_MEMORY_COPIES = 8


@dataclass(frozen=True, slots=True)
class SmokeReport:
    ok: bool
    protocol_version: int
    schema_hash: str
    reward_hash: str
    trajectory_hash: str
    python: str
    torch: str
    device: str
    device_name: str
    parameter_count: int
    batch_size: int
    iterations: int
    forward_steps_per_second: float
    peak_allocated_bytes: int
    backward_checked: bool
    finite_outputs: bool
    finite_gradients: bool
    parameters_unchanged: bool


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _tensor_hash(actor: AI42Actor) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _inputs(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    hero = torch.randn(batch, HERO_FEATURES, generator=generator, device=device)
    hero[:, 0] = torch.arange(batch, device=device).remainder(100) / 100.0
    abilities = torch.randn(
        batch, ABILITY_COUNT, ABILITY_FEATURES, generator=generator, device=device,
    )
    entities = torch.randn(
        batch, MAX_ENTITIES, ENTITY_FEATURES, generator=generator, device=device,
    )
    global_state = torch.randn(
        batch, GLOBAL_FEATURES, generator=generator, device=device,
    )
    entity_mask = torch.ones(batch, MAX_ENTITIES, dtype=torch.bool, device=device)
    if MAX_ENTITIES > 1:
        entity_mask[:, -1] = False
    return {
        "hero": hero,
        "abilities": abilities,
        "entities": entities,
        "global_state": global_state,
        "entity_mask": entity_mask,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bounded_integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if value > maximum:
        raise ValueError(f"{name} exceeds maximum {maximum}")
    return value


def _estimated_model_parameters(
    *,
    hidden_size: int,
    model_width: int,
    entity_layers: int,
    ff_multiplier: int,
) -> int:
    """Estimate AI42Actor parameters without constructing tensors."""

    def linear(inputs: int, outputs: int) -> int:
        return inputs * outputs + outputs

    def projection(inputs: int, outputs: int) -> int:
        return linear(inputs, outputs) + linear(outputs, outputs)

    total = sum(projection(inputs, model_width) for inputs in (
        HERO_FEATURES, ENTITY_FEATURES, GLOBAL_FEATURES, ABILITY_FEATURES,
    ))
    total += 128 * model_width
    total += ABILITY_COUNT * model_width
    total += 2 * linear(model_width, model_width)

    # MultiheadAttention, two LayerNorms, and the feed-forward sub-block.
    attention = (
        4 * model_width * model_width + 4 * model_width
        + 4 * model_width
        + linear(model_width, model_width * ff_multiplier)
        + linear(model_width * ff_multiplier, model_width)
    )
    total += entity_layers * attention
    total += projection(model_width, model_width)
    total += projection(1, model_width)
    total += 16 * hidden_size * model_width + 4 * hidden_size * hidden_size + 8 * hidden_size
    total += projection(hidden_size, model_width)
    total += linear(hidden_size, 2)
    total += linear(hidden_size, CONTROL_CONTINUATION_CLASSES)
    total += linear(hidden_size, KIND_GROUP_CLASSES)
    total += linear(model_width, 1)
    total += ACTION_KINDS * model_width
    total += 2 * linear(model_width, model_width)
    total += 2 * linear(model_width, NAVIGATION_GRID_SIZE)
    total += linear(model_width, NAVIGATION_ANCHORS)
    return total


def _estimated_work(
    parameter_count_estimate: int,
    batch_size: int,
    iterations: int,
    backward: bool,
) -> int:
    # Include warm-up and the optional backward pass.  The backward pass is
    # conservatively charged as another parameter-sized traversal.
    passes = iterations + 1 + int(backward)
    traversal_multiplier = 2 if backward else 1
    return parameter_count_estimate * batch_size * passes * traversal_multiplier


def _estimated_memory_bytes(
    parameter_count_estimate: int,
    *,
    batch_size: int,
    hidden_size: int,
    model_width: int,
    entity_layers: int,
    backward: bool,
) -> int:
    """Estimate model, input, and activation memory before allocation."""

    model_bytes = parameter_count_estimate * _MODEL_PARAMETER_BYTES * _MODEL_MEMORY_COPIES
    input_elements = batch_size * (
        HERO_FEATURES
        + ABILITY_COUNT * ABILITY_FEATURES
        + MAX_ENTITIES * ENTITY_FEATURES
        + GLOBAL_FEATURES
        + MAX_ENTITIES
    )
    # Bound the intermediate token/state footprint conservatively.  The
    # fixed extra factor covers recurrent, attention, and output-head values.
    activation_elements = batch_size * (
        (MAX_ENTITIES + ABILITY_COUNT + 4) * model_width * (entity_layers + 8)
        + hidden_size
    )
    activation_copies = 4 if backward else 2
    return model_bytes + (input_elements + activation_elements) * _MODEL_PARAMETER_BYTES * activation_copies


def _validate_smoke_config(
    *,
    device_name: object,
    batch_size: object,
    iterations: object,
    backward: object,
    hidden_size: object,
    model_width: object,
    entity_layers: object,
    num_heads: object,
) -> None:
    if not isinstance(device_name, str):
        raise ValueError("device_name must be a string")
    if not isinstance(backward, bool):
        raise ValueError("backward must be a bool")

    batch_size = _bounded_integer(batch_size, "batch_size", MAX_BATCH_SIZE)
    iterations = _bounded_integer(iterations, "iterations", MAX_ITERATIONS)
    hidden_size = _bounded_integer(hidden_size, "hidden_size", MAX_HIDDEN_SIZE)
    model_width = _bounded_integer(model_width, "model_width", MAX_MODEL_WIDTH)
    entity_layers = _bounded_integer(entity_layers, "entity_layers", MAX_ENTITY_LAYERS)
    num_heads = _bounded_integer(num_heads, "num_heads", MAX_NUM_HEADS)

    if model_width % num_heads:
        raise ValueError("model_width must be divisible by num_heads")

    estimated_parameters = _estimated_model_parameters(
        hidden_size=hidden_size,
        model_width=model_width,
        entity_layers=entity_layers,
        ff_multiplier=4,
    )
    if estimated_parameters > MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"estimated model parameters {estimated_parameters} exceed maximum {MAX_MODEL_PARAMETERS}"
        )

    estimated_work = _estimated_work(
        estimated_parameters, batch_size, iterations, backward,
    )
    if estimated_work > MAX_MODEL_WORK:
        raise ValueError(
            f"estimated model work {estimated_work} exceeds maximum {MAX_MODEL_WORK}"
        )

    estimated_memory = _estimated_memory_bytes(
        estimated_parameters,
        batch_size=batch_size,
        hidden_size=hidden_size,
        model_width=model_width,
        entity_layers=entity_layers,
        backward=backward,
    )
    if estimated_memory > MAX_MODEL_MEMORY_BYTES:
        raise ValueError(
            f"estimated model memory {estimated_memory} exceeds maximum {MAX_MODEL_MEMORY_BYTES}"
        )


def _check_actor_outputs(
    output: object,
    *,
    batch_size: int,
    hidden_size: int,
) -> bool:
    """Validate the complete actor contract and return its finite status."""

    expected = {
        "control": (batch_size, CONTROL_CLASSES),
        "kind": (batch_size, ACTION_KINDS),
        "target": (batch_size, ACTION_KINDS, MAX_ENTITIES),
        "offset": (batch_size, ACTION_KINDS, NAVIGATION_OFFSETS),
        "anchor": (batch_size, ACTION_KINDS, NAVIGATION_ANCHORS),
        "direction": (batch_size, ACTION_KINDS, NAVIGATION_OFFSETS),
        "distance": (batch_size, ACTION_KINDS, NAVIGATION_ANCHORS),
        "h": (batch_size, hidden_size),
        "c": (batch_size, hidden_size),
    }
    if not isinstance(output, Mapping):
        raise RuntimeError("actor output must be a mapping of tensors")
    if set(output) != set(expected):
        raise RuntimeError(
            f"actor output keys {sorted(output)} do not match {sorted(expected)}"
        )

    finite = True
    for key, shape in expected.items():
        value = output[key]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"actor output {key} is not a Tensor")
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"actor output {key} has shape {tuple(value.shape)}, want {shape}"
            )
        finite = bool(torch.isfinite(value).all().item()) and finite
    return finite


def run_smoke(
    *,
    device_name: str = "auto",
    batch_size: int = 10,
    iterations: int = 20,
    backward: bool = True,
    hidden_size: int = 192,
    model_width: int = 192,
    entity_layers: int = 2,
    num_heads: int = 6,
) -> SmokeReport:
    _validate_smoke_config(
        device_name=device_name,
        batch_size=batch_size,
        iterations=iterations,
        backward=backward,
        hidden_size=hidden_size,
        model_width=model_width,
        entity_layers=entity_layers,
        num_heads=num_heads,
    )
    device = _resolve_device(device_name)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    actor = AI42Actor(
        hidden_size=hidden_size,
        model_width=model_width,
        entity_layers=entity_layers,
        num_heads=num_heads,
    ).to(device)
    actor.eval()
    inputs = _inputs(batch_size, device)
    before = _tensor_hash(actor)

    with torch.no_grad():
        warm = actor(**inputs)
    _synchronize(device)
    finite_outputs = _check_actor_outputs(
        warm,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )

    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = actor(**inputs)
            finite_outputs = _check_actor_outputs(
                output,
                batch_size=batch_size,
                hidden_size=hidden_size,
            ) and finite_outputs
    _synchronize(device)
    elapsed = time.perf_counter() - started

    finite_gradients = True
    if backward:
        actor.train()
        actor.zero_grad(set_to_none=True)
        output = actor(**inputs)
        finite_outputs = _check_actor_outputs(
            output,
            batch_size=batch_size,
            hidden_size=hidden_size,
        ) and finite_outputs
        loss = (
            output["kind"].square().mean()
            + output["target"].square().mean()
            + output["offset"].square().mean()
            + output["anchor"].square().mean()
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("synthetic actor loss is non-finite")
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in actor.parameters()
        )
        actor.zero_grad(set_to_none=True)
        _synchronize(device)

    after = _tensor_hash(actor)
    unchanged = before == after
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    actual_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )
    ok = finite_outputs and finite_gradients and unchanged
    return SmokeReport(
        ok=ok,
        protocol_version=AI42_PROTOCOL_VERSION,
        schema_hash=AI42_SCHEMA_HASH.hex(),
        reward_hash=AI42_REWARD_HASH.hex(),
        trajectory_hash=AI42_TRAJECTORY_SCHEMA_HASH,
        python=platform.python_version(),
        torch=torch.__version__,
        device=str(device),
        device_name=actual_name,
        parameter_count=parameter_count(actor),
        batch_size=batch_size,
        iterations=iterations,
        forward_steps_per_second=(batch_size * iterations) / max(elapsed, 1e-12),
        peak_allocated_bytes=int(peak),
        backward_checked=backward,
        finite_outputs=finite_outputs,
        finite_gradients=finite_gradients,
        parameters_unchanged=unchanged,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--entity-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_smoke(
        device_name=args.device,
        batch_size=args.batch_size,
        iterations=args.iterations,
        backward=not args.no_backward,
        hidden_size=args.hidden_size,
        model_width=args.model_width,
        entity_layers=args.entity_layers,
        num_heads=args.num_heads,
    )
    payload = json.dumps(asdict(report), sort_keys=True, indent=2)
    print(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
