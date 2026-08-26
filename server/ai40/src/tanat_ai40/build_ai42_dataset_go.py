"""Production Go-backed AI42 dataset control plane.

Python owns the deterministic schedule, bounded subprocess orchestration,
resumable staging, frozen split assignment, and immutable publication.  Go
owns simulation and one-match shard production.  The merge path rewrites only
canonical JSON headers and shard names; compressed payload bytes are copied
without decompression.  The existing Python collector is available only via
the explicit ``--python-fallback`` CLI switch.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any

from .build_ai42_dataset import (
    COMMAND,
    DEFAULT_MAX_STEPS,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WORKERS,
    MatchSpec,
    STEP_SECONDS,
    build_match_specs,
    validate_timeout,
    _runtime_manifest,
)
from .dataset_ai42 import (
    AI42DatasetError,
    DATASET_SCHEMA_VERSION,
    _MATCH_FIELDS,
    _SHARD_FIELDS,
    _deterministic_split,
    _deterministic_stratified_split,
)
from .env import AI42_PROTOCOL_VERSION, AI42_REWARD_HASH, AI42_SCHEMA_HASH
from .go_shard_ai42 import (
    GO_MANIFEST_FILENAME,
    GO_SHARD_MAGIC_V2,
    GO_SHARD_SCHEMA_VERSION_V2,
    compact_match_entry,
    _parse_header,
    _sha256_bytes,
    load_go_dataset,
)
from .trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, canonical_json_bytes, hash_payload


GO_STAGING_SCHEMA_VERSION = "AI42-go-staging-v1"
GO_SCHEDULE_SCHEMA_VERSION = "AI42-go-schedule-v1"
COMPLETE_MARKER = b"AI42-go-staging-complete-v1\n"
DEFAULT_IN_FLIGHT = DEFAULT_WORKERS
DEFAULT_GO_WORKDIR = Path(__file__).resolve().parents[3]
DEFAULT_GO_COMMAND = ("go", "run", "./cmd/assaultdataset")
_SCENARIO_NAMES = frozenset({"ai30_mirror", "ai30_vs_ai20", "ai20_vs_ai30"})

Runner = Callable[[MatchSpec, Mapping[str, Any], Path], None]


def _canonical_write(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AI42DatasetError(f"cannot read {label}: {exc}") from exc
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AI42DatasetError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise AI42DatasetError(f"{label} is not canonical JSON")
    return value, payload


def _scenario_values(scenario: str | Sequence[str], match_count: int) -> tuple[str, ...]:
    if isinstance(scenario, str):
        values = tuple(item.strip() for item in scenario.split(",") if item.strip())
    else:
        values = tuple(scenario)
    if not values:
        values = ("ai30_mirror",)
    if len(values) == 1:
        return values * match_count
    if len(values) != match_count:
        raise ValueError("scenario schedule must contain one value per match")
    return values


def _normalize_scenario_mix(
    value: Mapping[str, Any],
    *,
    match_count: int,
    validation_fraction: float,
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("scenario_mix must be a non-empty mapping")
    normalized: dict[str, dict[str, int]] = {}
    for scenario, quota in sorted(value.items(), key=lambda item: str(item[0])):
        if scenario not in _SCENARIO_NAMES:
            raise ValueError(f"scenario_mix contains unsupported scenario {scenario!r}")
        if not isinstance(quota, Mapping) or set(quota) != {"train", "validation"}:
            raise ValueError(f"scenario_mix[{scenario!r}] must contain exactly train and validation")
        values: dict[str, int] = {}
        for split in ("train", "validation"):
            count = quota[split]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"scenario_mix[{scenario!r}].{split} must be a non-negative integer")
            values[split] = count
        normalized[scenario] = values
    total = sum(quota["train"] + quota["validation"] for quota in normalized.values())
    validation = sum(quota["validation"] for quota in normalized.values())
    if total != match_count:
        raise ValueError(f"scenario_mix quotas total {total}, expected match_count {match_count}")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if validation != match_count * fraction:
        raise ValueError("scenario_mix validation quotas do not match validation_fraction")
    return normalized


def _scenario_values_from_mix(seed: int, scenario_mix: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    ranked: list[tuple[bytes, str, int]] = []
    for scenario in sorted(scenario_mix):
        total = scenario_mix[scenario]["train"] + scenario_mix[scenario]["validation"]
        for ordinal in range(total):
            digest = hashlib.sha256(
                f"AI42-scenario-mix-v1\0{seed}\0{scenario}\0{ordinal}".encode("utf-8")
            ).digest()
            ranked.append((digest, scenario, ordinal))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(scenario for _, scenario, _ in ranked)


def build_go_schedule(
    *,
    seed: int,
    match_count: int,
    max_steps: int,
    timeout_minutes: float,
    scenario: str | Sequence[str],
    validation_fraction: float,
    split_seed: int,
    scenario_mix: Mapping[str, Any] | None = None,
    match_id_prefix: str = "ai42-match",
) -> tuple[dict[str, Any], tuple[MatchSpec, ...]]:
    """Build the worker-independent canonical schedule consumed by Go."""

    validate_timeout(max_steps, timeout_minutes)
    if not 0.0 <= float(validation_fraction) <= 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 1:
        raise ValueError("match_count must be a positive integer")
    normalized_mix = None
    if scenario_mix is not None:
        normalized_mix = _normalize_scenario_mix(
            scenario_mix, match_count=match_count, validation_fraction=float(validation_fraction)
        )
        scenario_values = _scenario_values_from_mix(seed, normalized_mix)
    else:
        scenario_values = _scenario_values(scenario, match_count)
    specs = build_match_specs(
        seed, match_count, scenario_values, match_id_prefix=match_id_prefix
    )
    # The worker count is orchestration state, not dataset identity.  It is
    # intentionally absent from this manifest so worker-count changes produce
    # byte-identical shards and final manifests.
    runtime = _runtime_manifest(
        specs,
        seed=seed,
        max_steps=max_steps,
        workers=1,
        shard_size=1,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    runtime.pop("workers", None)
    runtime["schedule_schema_version"] = GO_SCHEDULE_SCHEMA_VERSION
    runtime["backend"] = "go"
    if normalized_mix is not None:
        runtime["scenario_mix"] = normalized_mix
    return runtime, specs


def _command_value(value: str | Path | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_GO_COMMAND
    if isinstance(value, (str, Path)):
        parts = tuple(shlex.split(str(value), posix=False))
    else:
        parts = tuple(str(item) for item in value)
    if not parts:
        raise ValueError("go_command must not be empty")
    if any(
        item in {"-max-steps", "--max-steps"}
        or item.startswith("-max-steps=")
        or item.startswith("--max-steps=")
        for item in parts
    ):
        raise ValueError("Go CLI max-steps override is forbidden; max_steps must come from schedule.json")
    return parts


def _contract(
    schedule: Mapping[str, Any],
    specs: Sequence[MatchSpec],
    *,
    workers: int,
    in_flight: int,
    go_command: Sequence[str],
    max_steps: int,
    timeout_minutes: float,
    staging: Path,
) -> dict[str, Any]:
    return {
        "staging_schema_version": GO_STAGING_SCHEMA_VERSION,
        "schedule_schema_version": GO_SCHEDULE_SCHEMA_VERSION,
        "schedule": dict(schedule),
        "schedule_hash": hash_payload(schedule),
        "runtime_manifest_hash": hash_payload(schedule),
        "match_ids": [spec.match_id for spec in specs],
        "max_steps": max_steps,
        "timeout_minutes": timeout_minutes,
        "workers": workers,
        "in_flight": in_flight,
        "go_command": list(go_command),
        "staging_path": str(staging),
    }


def _ensure_staging(root: Path, contract: Mapping[str, Any], schedule: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    matches = root / "matches"
    matches.mkdir(exist_ok=True)
    contract_path = root / "contract.json"
    schedule_path = root / "schedule.json"
    expected_contract = canonical_json_bytes(contract)
    expected_schedule = canonical_json_bytes(schedule)
    if contract_path.exists():
        if contract_path.read_bytes() != expected_contract:
            raise AI42DatasetError("Go staging contract mismatch")
    else:
        contract_path.write_bytes(expected_contract)
    if schedule_path.exists():
        if schedule_path.read_bytes() != expected_schedule:
            raise AI42DatasetError("Go staging schedule mismatch")
    else:
        schedule_path.write_bytes(expected_schedule)
    expected_dirs = {f"match-{index:06d}" for index in range(len(contract["match_ids"]))}
    for child in matches.iterdir():
        if child.name not in expected_dirs:
            raise AI42DatasetError(f"Go staging contains an unexpected entry: {child.name}")


def _match_root(staging: Path, index: int) -> Path:
    return staging / "matches" / f"match-{index:06d}"


def _validate_staged_match(
    root: Path,
    spec: MatchSpec,
    schedule: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    marker = root / "COMPLETE"
    if require_complete and (not marker.is_file() or marker.read_bytes() != COMPLETE_MARKER):
        raise AI42DatasetError(f"staged match {spec.match_id} is partial or missing COMPLETE")
    dataset = load_go_dataset(
        root,
        expected_runtime_manifest_hash=hash_payload(schedule),
        allow_partial_schedule="scenario_mix" in schedule,
    )
    if dataset.match_ids() != (spec.match_id,) or len(dataset.manifest["shards"]) != 1:
        raise AI42DatasetError(f"staged match {spec.match_id} does not contain exactly one match/shard")
    entry = dataset.manifest["matches"][0]
    if entry["tick_count"] < 1 or entry["tick_count"] > int(schedule["max_steps"]):
        raise AI42DatasetError(f"staged match {spec.match_id} has an invalid tick count")
    if entry["scenario"] != spec.scenario or tuple(entry["controller_by_slot"]) != spec.controller_by_slot:
        raise AI42DatasetError(f"staged match {spec.match_id} provenance mismatch")
    if tuple(entry["roster_ids"]) != spec.roster_ids or tuple(entry["side_by_slot"]) != spec.side_by_slot:
        raise AI42DatasetError(f"staged match {spec.match_id} roster/side mismatch")
    shard_name = dataset.manifest["shards"][0]["name"]
    payload = (root / shard_name).read_bytes()
    header, _compressed = _parse_header(payload, str(root / shard_name))
    if header["matches"] != [entry]:
        raise AI42DatasetError(f"staged match {spec.match_id} header metadata mismatch")


def _run_go_match(
    spec: MatchSpec,
    output: Path,
    *,
    schedule_path: Path,
    go_command: Sequence[str],
    go_workdir: Path,
    timeout_seconds: float | None,
) -> None:
    command = [
        *go_command,
        "-schedule", str(schedule_path),
        "-output", str(output),
        "-match-index", str(spec.index),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=go_workdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AI42DatasetError(f"Go producer failed for {spec.match_id}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise AI42DatasetError(
            f"Go producer failed for {spec.match_id} with exit {completed.returncode}: {detail}"
        )


def _stage_one(
    spec: MatchSpec,
    schedule: Mapping[str, Any],
    staging: Path,
    *,
    go_command: Sequence[str],
    go_workdir: Path,
    runner: Runner | None,
    timeout_seconds: float | None,
) -> None:
    destination = _match_root(staging, spec.index)
    if destination.exists():
        _validate_staged_match(destination, spec, schedule)
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".match-{spec.index:06d}-", dir=staging / "matches"))
    try:
        if runner is None:
            _run_go_match(
                spec, temporary / "generation", schedule_path=staging / "schedule.json",
                go_command=go_command, go_workdir=go_workdir, timeout_seconds=timeout_seconds,
            )
            generated = temporary / "generation"
            if not generated.exists():
                raise AI42DatasetError(f"Go producer did not create {generated}")
            for child in generated.iterdir():
                os.replace(child, temporary / child.name)
            generated.rmdir()
        else:
            runner(spec, schedule, temporary)
        _validate_staged_match(temporary, spec, schedule, require_complete=False)
        (temporary / "COMPLETE").write_bytes(COMPLETE_MARKER)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _recanonical_shard(
    source_root: Path,
    source_entry: Mapping[str, Any],
    final_entry: Mapping[str, Any],
    final_name: str,
) -> tuple[bytes, dict[str, Any]]:
    source_name = source_entry["shards"][0]["name"] if "shards" in source_entry else None
    if source_name is None:
        raise AI42DatasetError("source shard metadata is missing")
    source_path = source_root / source_name
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise AI42DatasetError(f"cannot read staged shard {source_path}: {exc}") from exc
    header, compressed = _parse_header(payload, str(source_path))
    if header["shard_schema_version"] == GO_SHARD_SCHEMA_VERSION_V2:
        match = compact_match_entry(final_entry, schema_version=GO_SHARD_SCHEMA_VERSION_V2)
    else:
        match = compact_match_entry(final_entry, schema_version=header["shard_schema_version"])
    match["shard"] = final_name
    match["row_offset"] = 0
    header["shard_schema_version"] = GO_SHARD_SCHEMA_VERSION_V2
    header["matches"] = [match]
    header_bytes = canonical_json_bytes(header)
    rewritten = (
        GO_SHARD_MAGIC_V2
        + len(header_bytes).to_bytes(4, "little")
        + header_bytes
        + compressed
    )
    verified_header, verified_compressed = _parse_header(rewritten, f"{final_name} (rewritten)")
    if verified_header["matches"] != [match] or verified_compressed != compressed:
        raise AI42DatasetError(f"rewritten shard {final_name} failed header/payload verification")
    shard = {
        "name": final_name,
        "sha256": _sha256_bytes(rewritten),
        "match_ids": [match["match_id"]],
        "row_count": match["tick_count"],
        "raw_bytes": header["raw_bytes"],
        "stored_bytes": len(rewritten),
        "compression": header["codec"],
    }
    return rewritten, shard


def merge_go_staging(
    destination: str | Path,
    staging: str | Path,
    *,
    schedule: Mapping[str, Any],
    specs: Sequence[MatchSpec],
    split_seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    """Merge staged one-match shards without decompressing their payloads."""

    destination_path = Path(destination)
    staging_path = Path(staging)
    if destination_path.exists():
        raise AI42DatasetError("final Go dataset destination already exists; generations are immutable")
    if "scenario_mix" in schedule:
        assignments = _deterministic_stratified_split(
            [(spec.match_id, spec.scenario) for spec in specs],
            schedule["scenario_mix"],
            int(split_seed),
            validation_fraction=float(validation_fraction),
        )
    else:
        assignments = _deterministic_split(
            [spec.match_id for spec in specs], float(validation_fraction), int(split_seed)
        )
    entries: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent))
    try:
        for output_index, spec in enumerate(sorted(specs, key=lambda item: item.match_id)):
            source = _match_root(staging_path, spec.index)
            _validate_staged_match(source, spec, schedule)
            manifest, _ = _read_canonical(source / GO_MANIFEST_FILENAME, f"{source}/manifest.json")
            source_entry = manifest["matches"][0]
            final_name = f"shard-{output_index:06d}.a42"
            final_entry = dict(source_entry)
            final_entry["split"] = assignments[spec.match_id]
            final_entry["row_offset"] = 0
            final_entry["shard"] = final_name
            rewritten, shard = _recanonical_shard(source, manifest, final_entry, final_name)
            final_entry = compact_match_entry(
                final_entry,
                schema_version=manifest["shard_schema_version"],
            )
            final_entry["shard"] = final_name
            (temporary / final_name).write_bytes(rewritten)
            entries.append(final_entry)
            shards.append(shard)
        entries.sort(key=lambda item: item["match_id"])
        shards.sort(key=lambda item: item["name"])
        manifest_unsigned: dict[str, Any] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "shard_schema_version": GO_SHARD_SCHEMA_VERSION_V2,
            "protocol_version": AI42_PROTOCOL_VERSION,
            "schema_hash": AI42_SCHEMA_HASH.hex(),
            "reward_hash": AI42_REWARD_HASH.hex(),
            "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
            "runtime_manifest_hash": hash_payload(schedule),
            "runtime_manifest": dict(schedule),
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "matches": entries,
            "shards": shards,
        }
        manifest = dict(manifest_unsigned)
        manifest["manifest_hash"] = hash_payload(manifest_unsigned)
        (temporary / GO_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    dataset = load_go_dataset(destination_path, expected_runtime_manifest=schedule)
    return {
        "dataset": str(destination_path),
        "manifest_hash": dataset.manifest_hash,
        "runtime_manifest_hash": dataset.runtime_manifest_hash,
        "matches": len(dataset),
        "ticks": sum(int(entry["tick_count"]) for entry in dataset.manifest["matches"]),
        "compression": dataset.compression,
        "decompressed_during_merge": False,
    }


def build_ai42_dataset_go(
    destination: str | Path,
    *,
    match_count: int = 1,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
    in_flight: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES,
    scenario: str | Sequence[str] = "ai30_mirror",
    scenario_mix: Mapping[str, Any] | None = None,
    staging: str | Path | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = 42,
    match_id_prefix: str = "ai42-match",
    go_command: str | Path | Sequence[str] | None = None,
    go_workdir: str | Path | None = None,
    runner: Runner | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run bounded native Go collection, resume staging, and publish."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    active = workers if in_flight is None else in_flight
    if isinstance(active, bool) or not isinstance(active, int) or active < 1:
        raise ValueError("in_flight must be a positive integer")
    active = min(active, workers)
    command = _command_value(go_command)
    workdir = (Path(go_workdir) if go_workdir is not None else DEFAULT_GO_WORKDIR).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise AI42DatasetError("final Go dataset destination already exists; generations are immutable")
    staging_path = (
        Path(staging).resolve()
        if staging is not None
        else (destination_path.parent / f".{destination_path.name}.go-staging").resolve()
    )
    schedule, specs = build_go_schedule(
        seed=seed, match_count=match_count, max_steps=max_steps,
        timeout_minutes=timeout_minutes, scenario=scenario,
        validation_fraction=validation_fraction, split_seed=split_seed,
        scenario_mix=scenario_mix, match_id_prefix=match_id_prefix,
    )
    contract = _contract(
        schedule, specs, workers=workers, in_flight=active, go_command=command,
        max_steps=max_steps, timeout_minutes=timeout_minutes, staging=staging_path,
    )
    _ensure_staging(staging_path, contract, schedule)
    remaining = [spec for spec in specs if not _match_root(staging_path, spec.index).exists()]
    for spec in specs:
        if _match_root(staging_path, spec.index).exists():
            _validate_staged_match(_match_root(staging_path, spec.index), spec, schedule)
    started = time.perf_counter()
    for start in range(0, len(remaining), active):
        batch = remaining[start : start + active]
        with ThreadPoolExecutor(max_workers=min(active, len(batch))) as executor:
            futures = [executor.submit(
                _stage_one, spec, schedule, staging_path,
                go_command=command, go_workdir=workdir, runner=runner,
                timeout_seconds=timeout_seconds,
            ) for spec in batch]
            for future in as_completed(futures):
                future.result()
    collection_seconds = time.perf_counter() - started
    summary = merge_go_staging(
        destination_path, staging_path, schedule=schedule, specs=specs,
        split_seed=split_seed, validation_fraction=validation_fraction,
    )
    summary.update({
        "command": COMMAND,
        "backend": "go",
        "staging": str(staging_path),
        "schedule_hash": hash_payload(schedule),
        "workers": workers,
        "in_flight": active,
        "max_steps": max_steps,
        "timeout_minutes": timeout_minutes,
        "seed": seed,
        "resumed_matches": match_count - len(remaining),
        "timings_seconds": {"collection": collection_seconds},
    })
    return summary


def _config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="destination, or legacy executable followed by destination")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python-fallback", action="store_true")
    parser.add_argument("--go-command")
    parser.add_argument("--go-workdir", type=Path)
    parser.add_argument("--matches", dest="match_count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--in-flight", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--timeout-minutes", type=float)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--scenario-mix", type=Path)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--match-id-prefix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else list(__import__("sys").argv[1:])
    if "--python-fallback" in raw:
        raw.remove("--python-fallback")
        from .build_ai42_dataset import main as python_main

        return python_main(raw)
    args = _parser().parse_args(raw)
    if len(args.paths) == 1:
        executable = "go"
        destination = Path(args.paths[0])
    elif len(args.paths) == 2:
        executable = args.paths[0]
        destination = Path(args.paths[1])
    else:
        raise SystemExit("expected DESTINATION or GO_EXECUTABLE DESTINATION")
    values: dict[str, Any] = {}
    if args.config is not None:
        values.update(_config(args.config))
    command = args.go_command
    if command is None and executable != "go":
        command = (executable,)
    overrides = {
        "match_count": args.match_count, "seed": args.seed, "workers": args.workers,
        "in_flight": args.in_flight, "max_steps": args.max_steps,
        "timeout_minutes": args.timeout_minutes, "staging": args.staging,
        "validation_fraction": args.validation_fraction, "split_seed": args.split_seed,
        "match_id_prefix": args.match_id_prefix,
        "go_workdir": args.go_workdir, "go_command": command,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if args.scenario is not None:
        values["scenario"] = args.scenario if len(args.scenario) > 1 else args.scenario[0]
    if args.scenario_mix is not None:
        values["scenario_mix"] = _config(args.scenario_mix)
    print(canonical_json_bytes(build_ai42_dataset_go(destination, **values)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GO_SCHEDULE_SCHEMA_VERSION", "GO_STAGING_SCHEMA_VERSION",
    "build_ai42_dataset_go", "build_go_schedule", "main", "merge_go_staging",
]
