"""Benchmark the AI-42 actor boundary with PyTorch and ONNX Runtime.

The benchmark measures the complete synchronous actor call seen by a rollout
loop: observations enter from CPU memory and all logits plus recurrent state
are materialized before the next tick.  It does not benchmark the simulator or
the Python action-selection code; those phases are measured by evaluate_ai42.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    MAX_ENTITIES,
)
from .evaluate_ai42 import load_actor
from .export_ai42 import (
    ACTOR_INPUT_NAMES,
    AI42ActorONNXWrapper,
    assert_onnx_parity,
    export_ai42_actor,
)


def _inputs(actor: Any, batch: int, seed: int = 42) -> tuple[torch.Tensor, ...]:
    if batch < 1:
        raise ValueError("batch must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hero = torch.randn((batch, HERO_FEATURES), generator=generator)
    abilities = torch.randn(
        (batch, ABILITY_COUNT, ABILITY_FEATURES), generator=generator,
    )
    entities = torch.randn(
        (batch, MAX_ENTITIES, ENTITY_FEATURES), generator=generator,
    )
    global_state = torch.randn((batch, GLOBAL_FEATURES), generator=generator)
    entity_mask = torch.rand((batch, MAX_ENTITIES), generator=generator) >= 0.25
    entity_mask[:, 0] = True
    h, c = actor.initial_state(batch, "cpu")
    return hero, abilities, entities, global_state, entity_mask, h, c


def _timing(total_seconds: float, iterations: int, batch: int) -> dict[str, float | int]:
    if total_seconds <= 0 or iterations < 1 or batch < 1:
        raise ValueError("timing inputs must be positive")
    return {
        "iterations": iterations,
        "batch": batch,
        "seconds": total_seconds,
        "milliseconds_per_batch": total_seconds * 1000.0 / iterations,
        "rows_per_second": iterations * batch / total_seconds,
    }


def benchmark_torch(
    actor: Any,
    inputs: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    wrapper = AI42ActorONNXWrapper(actor.to(device).eval()).eval()
    values = tuple(value.to(device) for value in inputs)

    def run_once(current: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = wrapper(*current)
        return (*current[:-2], output[-2].detach(), output[-1].detach())

    for _ in range(warmup):
        values = run_once(values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iterations):
        values = run_once(values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = dict(_timing(elapsed, iterations, int(inputs[0].shape[0])))
    result["device"] = str(device)
    result["autocast"] = "bfloat16" if device.type == "cuda" else "disabled"
    result["peak_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result["peak_reserved_bytes"] = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    return result


def benchmark_onnx(
    session: Any,
    inputs: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    values = {
        name: np.ascontiguousarray(value.detach().cpu().numpy())
        for name, value in zip(ACTOR_INPUT_NAMES, inputs)
    }

    def run_once() -> None:
        output = session.run(None, values)
        values["h"] = output[-2]
        values["c"] = output[-1]

    for _ in range(warmup):
        run_once()
    started = time.perf_counter()
    for _ in range(iterations):
        run_once()
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = dict(_timing(elapsed, iterations, int(inputs[0].shape[0])))
    result["providers"] = list(session.get_providers())
    return result


def benchmark_generation(
    checkpoint: Path,
    config: Path,
    *,
    batch: int,
    iterations: int,
    warmup: int,
    device: torch.device,
    onnx_path: Path,
) -> dict[str, Any]:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    actor, lineage = load_actor(checkpoint, config, torch.device("cpu"))
    inputs = _inputs(actor, batch)
    export_started = time.perf_counter()
    export_ai42_actor(actor, onnx_path, example_inputs=inputs)
    export_seconds = time.perf_counter() - export_started

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        # Reuse the CUDA/cuDNN runtime bundled with the installed CUDA PyTorch
        # wheel instead of requiring a second system-wide toolkit install.
        ort.preload_dlls()
    cpu_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity = assert_onnx_parity(actor, cpu_session, inputs, rtol=2e-4, atol=2e-5)
    available = ort.get_available_providers()
    if device.type == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"ONNX Runtime CUDA provider is unavailable: {available}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_providers = list(session.get_providers())
    if device.type == "cuda" and (
        not active_providers or active_providers[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "ONNX Runtime silently fell back from CUDA; "
            f"active providers are {active_providers}"
        )
    return {
        "format": "AI42-inference-benchmark-v1",
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config.resolve()),
        "onnx": str(onnx_path.resolve()),
        "lineage": lineage,
        "export_seconds": export_seconds,
        "parity": parity.to_dict(),
        "torch": benchmark_torch(
            actor, inputs, iterations=iterations, warmup=warmup, device=device,
        ),
        "onnxruntime": benchmark_onnx(
            session, inputs, iterations=iterations, warmup=warmup,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tanat-ai42-onnx-") as directory:
        onnx_path = args.onnx or Path(directory) / "actor.onnx"
        report = benchmark_generation(
            args.checkpoint,
            args.config,
            batch=args.batch,
            iterations=args.iterations,
            warmup=args.warmup,
            device=torch.device(args.device),
            onnx_path=onnx_path,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()


__all__ = [
    "benchmark_generation", "benchmark_onnx", "benchmark_torch", "main",
]
