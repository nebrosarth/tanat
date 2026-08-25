"""Fail-closed AI-42 actor preflight that never performs an optimizer update.

The command is intentionally separate from every training entry point.  It
validates the installed PyTorch runtime and the real actor configuration,
measures inference/backward resource use, checks finite outputs/gradients, and
proves that the model parameters remain byte-identical throughout the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
)
from .model_ai42_actor import AI42Actor, parameter_count
from .trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH


@dataclass(frozen=True, slots=True)
class PreflightReport:
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


def run_preflight(
    *,
    device_name: str = "auto",
    batch_size: int = 10,
    iterations: int = 20,
    backward: bool = True,
    hidden_size: int = 384,
    model_width: int = 384,
    entity_layers: int = 4,
    num_heads: int = 8,
) -> PreflightReport:
    if batch_size < 1 or iterations < 1:
        raise ValueError("batch_size and iterations must be positive")
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
    finite_outputs = all(
        bool(torch.isfinite(value).all().item())
        for key, value in warm.items()
        if key not in {"direction", "distance"}
    )
    expected = {
        "kind": (batch_size, ACTION_KINDS),
        "target": (batch_size, ACTION_KINDS, MAX_ENTITIES),
        "h": (batch_size, hidden_size),
        "c": (batch_size, hidden_size),
    }
    for key, shape in expected.items():
        if tuple(warm[key].shape) != shape:
            raise RuntimeError(f"actor output {key} has shape {tuple(warm[key].shape)}, want {shape}")

    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = actor(**inputs)
    _synchronize(device)
    elapsed = time.perf_counter() - started

    finite_gradients = True
    if backward:
        actor.train()
        actor.zero_grad(set_to_none=True)
        output = actor(**inputs)
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
    return PreflightReport(
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
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--entity-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(
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
