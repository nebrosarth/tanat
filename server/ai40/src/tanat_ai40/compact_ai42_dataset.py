"""Atomically compact an AI42 Go v1 generation to metadata-only v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .dataset_ai42 import AI42DatasetError
from .go_shard_ai42 import (
    GO_MANIFEST_FILENAME,
    GO_SHARD_MAGIC_V2,
    GO_SHARD_SCHEMA_VERSION_V1,
    GO_SHARD_SCHEMA_VERSION_V2,
    _parse_header,
    _sha256_bytes,
    compact_match_entry,
    load_go_dataset,
)
from .trajectory_ai42 import canonical_json_bytes, hash_payload


def _rewrite_shard(
    source_root: Path,
    destination_root: Path,
    source_manifest: Mapping[str, Any],
    shard_entry: Mapping[str, Any],
) -> dict[str, Any]:
    name = shard_entry["name"]
    source_path = source_root / name
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise AI42DatasetError(f"cannot read v1 shard {source_path}: {exc}") from exc
    header, compressed = _parse_header(payload, str(source_path))
    if header["shard_schema_version"] != GO_SHARD_SCHEMA_VERSION_V1:
        raise AI42DatasetError(f"{source_path} is not an AI42 v1 shard")
    expected_matches = [
        entry for entry in source_manifest["matches"] if entry["shard"] == name
    ]
    if header["matches"] != expected_matches:
        raise AI42DatasetError(f"{source_path} header metadata does not match manifest")
    matches = [
        compact_match_entry(entry, schema_version=GO_SHARD_SCHEMA_VERSION_V1)
        for entry in expected_matches
    ]
    header = dict(header)
    header["shard_schema_version"] = GO_SHARD_SCHEMA_VERSION_V2
    header["matches"] = matches
    header_bytes = canonical_json_bytes(header)
    rewritten = GO_SHARD_MAGIC_V2 + len(header_bytes).to_bytes(4, "little") + header_bytes + compressed
    verified_header, verified_compressed = _parse_header(rewritten, f"{name} (v2)")
    if verified_compressed != compressed or verified_header["matches"] != matches:
        raise AI42DatasetError(f"{name} v1-to-v2 rewrite verification failed")
    destination_path = destination_root / name
    destination_path.write_bytes(rewritten)
    return {
        "name": name,
        "sha256": _sha256_bytes(rewritten),
        "match_ids": list(shard_entry["match_ids"]),
        "row_count": int(shard_entry["row_count"]),
        "raw_bytes": int(shard_entry["raw_bytes"]),
        "stored_bytes": len(rewritten),
        "compression": header["codec"],
    }


def compact_ai42_dataset(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Publish an immutable v2 generation by rewriting headers only."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise AI42DatasetError("final AI42 v2 destination already exists; generations are immutable")
    source_dataset = load_go_dataset(source_path)
    source_manifest = source_dataset.manifest
    if source_manifest["shard_schema_version"] != GO_SHARD_SCHEMA_VERSION_V1:
        raise AI42DatasetError("source generation must use AI42-go-shard-v1")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.compact-", dir=destination_path.parent))
    try:
        compact_matches = [
            compact_match_entry(entry, schema_version=GO_SHARD_SCHEMA_VERSION_V1)
            for entry in source_manifest["matches"]
        ]
        compact_shards = [
            _rewrite_shard(source_path, temporary, source_manifest, shard)
            for shard in source_manifest["shards"]
        ]
        unsigned = dict(source_manifest)
        unsigned["shard_schema_version"] = GO_SHARD_SCHEMA_VERSION_V2
        unsigned["matches"] = compact_matches
        unsigned["shards"] = compact_shards
        unsigned.pop("manifest_hash", None)
        manifest = dict(unsigned)
        manifest["manifest_hash"] = hash_payload(unsigned)
        (temporary / GO_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    compacted = load_go_dataset(destination_path)
    return {
        "dataset": str(destination_path),
        "source": str(source_path),
        "source_schema": GO_SHARD_SCHEMA_VERSION_V1,
        "schema": GO_SHARD_SCHEMA_VERSION_V2,
        "manifest_hash": compacted.manifest_hash,
        "runtime_manifest_hash": compacted.runtime_manifest_hash,
        "matches": len(compacted),
        "raw_bytes": sum(int(item["raw_bytes"]) for item in compacted.manifest["shards"]),
        "stored_bytes": sum(int(item["stored_bytes"]) for item in compacted.manifest["shards"]),
        "decompressed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(canonical_json_bytes(compact_ai42_dataset(args.source, args.destination)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compact_ai42_dataset", "main"]
