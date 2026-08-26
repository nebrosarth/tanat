"""Control-plane and lazy reader for the AI42 Go shard contracts.

The Go collector owns shard production.  This module deliberately does not
launch Go or write production shards; it validates and reads the immutable
``AI42-go-shard-v1`` generation it publishes.  The format is intentionally
small and language-neutral:

* ``manifest.json`` is canonical UTF-8 JSON with the same match/provenance
  contract as the Python dataset, but a distinct shard schema version.
* v1 shards use ``AI42GS1\\0`` and retain explicit tick/lineage matrices.
* v2 shards use ``AI42GS2\\0`` and compact match metadata; ticks and lineage
  are derived only by an explicit metadata request.
* The payload is a contiguous C-order concatenation of the fixed AI42 arrays.
  Header offsets and dtype keys are authoritative and are checked exactly.

The reader verifies the manifest and complete shard-file hashes at load, then
decompresses at most one shard and copies at most one match at a time.  It
never imports learner, optimizer, torch, or training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Iterator, Mapping
import hashlib
import zlib

import numpy as np

from .dataset_ai42 import (
    AI42DatasetError,
    _ARRAY_DTYPES,
    _ARRAY_NAMES,
    _hash_hex,
    _runtime_manifest_hash,
    _deterministic_split,
    _deterministic_stratified_split,
    _strict_json,
    _TOP_LEVEL_FIELDS,
    _MATCH_FIELDS,
    _SHARD_FIELDS,
)
from .env import AI42_PROTOCOL_VERSION, AI42_REWARD_HASH, AI42_SCHEMA_HASH
from .trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, canonical_json_bytes, hash_payload


GO_SHARD_SCHEMA_VERSION_V1 = "AI42-go-shard-v1"
GO_SHARD_MAGIC_V1 = b"AI42GS1\0"
GO_SHARD_SCHEMA_VERSION_V2 = "AI42-go-shard-v2"
GO_SHARD_MAGIC_V2 = b"AI42GS2\0"
# Legacy aliases remain v1 for callers that construct or inspect the native
# Go staging format.  Python publication uses the explicit v2 constants.
GO_SHARD_SCHEMA_VERSION = GO_SHARD_SCHEMA_VERSION_V1
GO_SHARD_MAGIC = GO_SHARD_MAGIC_V1
GO_SHARD_CODEC = "deflate-raw-3"
GO_SHARD_LEGACY_CODEC = "deflate-raw-6"
_GO_SHARD_CODECS = frozenset((GO_SHARD_CODEC, GO_SHARD_LEGACY_CODEC))
GO_MANIFEST_FILENAME = "manifest.json"
GO_V2_LINEAGE_SCHEMA = "implicit-match-boundary-v1"
_SCHEMA_BY_MAGIC = {
    GO_SHARD_MAGIC_V1: GO_SHARD_SCHEMA_VERSION_V1,
    GO_SHARD_MAGIC_V2: GO_SHARD_SCHEMA_VERSION_V2,
}
_MAGIC_BY_SCHEMA = {value: key for key, value in _SCHEMA_BY_MAGIC.items()}
_SUPPORTED_SHARD_SCHEMAS = frozenset(_MAGIC_BY_SCHEMA)
_HEADER_PREFIX = struct.Struct("<8sI")
_MAX_HEADER_BYTES = 16 * 1024 * 1024

_GO_HEADER_FIELDS = frozenset({
    "shard_schema_version", "protocol_version", "schema_hash", "reward_hash",
    "trajectory_schema_hash", "runtime_manifest_hash", "codec", "raw_bytes",
    "stored_bytes", "raw_sha256", "payload_sha256", "arrays", "matches",
})
_GO_ARRAY_FIELDS = frozenset({"name", "dtype", "shape", "offset", "nbytes"})
_V2_MATCH_FIELDS = frozenset(
    (_MATCH_FIELDS - {"ticks", "recurrent_parent_ids", "recurrent_boundary_ids"})
    | {"first_step", "recurrent_lineage_schema"}
)

# These names are part of the cross-language contract.  ``action`` is the
# fixed four-field v13 wire dtype: u1, little-endian u2, u1, u1 (five bytes).
_GO_DTYPE_KEYS: dict[str, np.dtype] = {
    "u1": np.dtype("u1"),
    "<f4": np.dtype("<f4"),
    "<i4": np.dtype("<i4"),
    "<u4": np.dtype("<u4"),
    "action": _ARRAY_DTYPES["teacher_action"],
}


class AI42GoShardError(AI42DatasetError):
    """Raised when a Go generation violates the versioned shard contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AI42GoShardError(f"{name} must be an integer >= {minimum}")
    return value


def _dtype_key(name: str) -> str:
    try:
        dtype = _ARRAY_DTYPES[name]
    except KeyError as exc:
        raise AI42GoShardError(f"unknown AI42 array {name!r}") from exc
    if dtype == _ARRAY_DTYPES["teacher_action"]:
        return "action"
    if dtype == np.dtype("u1"):
        return "u1"
    if dtype == np.dtype("<f4"):
        return "<f4"
    if dtype == np.dtype("<i4"):
        return "<i4"
    if dtype == np.dtype("<u4"):
        return "<u4"
    raise AI42GoShardError(f"AI42 array {name!r} has no Go dtype key")


def _expected_shape(name: str, rows: int) -> tuple[int, ...]:
    if name == "hero":
        return rows, 10, 32
    if name == "abilities":
        return rows, 10, 4, 40
    if name == "entities":
        return rows, 10, 96, 16
    if name == "global":
        return rows, 10, 32
    if name == "entity_mask":
        return rows, 10, 96
    if name == "kind_mask":
        return rows, 10, 8
    if name == "target_mask":
        return rows, 10, 96
    if name == "skill_target_mask":
        return rows, 10, 4, 96
    if name in {"teacher_status", "executed_valid", "rejection_reason", "invalid"}:
        return rows, 10
    if name in {"teacher_action", "projected_action", "executed_action"}:
        return rows, 10
    if name == "rewards":
        return rows, 10
    if name == "elapsed":
        return rows,
    if name in {"done", "winner", "step"}:
        return rows,
    raise AI42GoShardError(f"unknown AI42 array {name!r}")


def _validate_match_entry(entry: Any, path: str, schema_version: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise AI42GoShardError(f"{path} must be an object")
    expected_fields = _MATCH_FIELDS if schema_version == GO_SHARD_SCHEMA_VERSION_V1 else _V2_MATCH_FIELDS
    if frozenset(entry) != expected_fields:
        raise AI42GoShardError(f"{path} field set mismatch")
    result = dict(entry)
    match_id = result["match_id"]
    if not isinstance(match_id, str) or not match_id:
        raise AI42GoShardError(f"{path}.match_id must be a non-empty string")
    ticks = _strict_int(result["tick_count"], f"{path}.tick_count", minimum=1)
    _strict_int(result["row_offset"], f"{path}.row_offset")
    if schema_version == GO_SHARD_SCHEMA_VERSION_V1:
        tick_values = result["ticks"]
        if not isinstance(tick_values, list) or len(tick_values) != ticks or any(
            isinstance(value, bool) or not isinstance(value, int) for value in tick_values
        ):
            raise AI42GoShardError(f"{path}.ticks must be contiguous")
        if tick_values != list(range(tick_values[0], tick_values[0] + ticks)):
            raise AI42GoShardError(f"{path}.ticks length mismatch")
    else:
        first_step = _strict_int(result["first_step"], f"{path}.first_step")
        if result["recurrent_lineage_schema"] != GO_V2_LINEAGE_SCHEMA:
            raise AI42GoShardError(f"{path}.recurrent_lineage_schema mismatch")
    for field in ("hero_ids", "trajectory_ids", "trajectory_hashes"):
        values = result[field]
        if not isinstance(values, list) or len(values) != 10 or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise AI42GoShardError(f"{path}.{field} must contain ten values")
        if len(set(values)) != 10:
            raise AI42GoShardError(f"{path}.{field} must contain unique values")
    for index, value in enumerate(result["trajectory_hashes"]):
        _hash_hex(value, f"{path}.trajectory_hashes[{index}]")
    for field in ("controller_by_slot", "roster_ids", "side_by_slot"):
        if not isinstance(result[field], list) or len(result[field]) != 10:
            raise AI42GoShardError(f"{path}.{field} must contain ten values")
    controllers = result["controller_by_slot"]
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3 for value in controllers):
        raise AI42GoShardError(f"{path}.controller_by_slot contains an invalid controller")
    sides = result["side_by_slot"]
    if any(isinstance(value, bool) or value not in (0, 1) for value in sides) or sides.count(0) != 5 or sides.count(1) != 5:
        raise AI42GoShardError(f"{path}.side_by_slot must contain five slots per side")
    roster = result["roster_ids"]
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value) for value in roster):
        raise AI42GoShardError(f"{path}.roster_ids contains an invalid roster ID")
    if len(set(roster)) != 10:
        raise AI42GoShardError(f"{path}.roster_ids must be unique")
    if schema_version == GO_SHARD_SCHEMA_VERSION_V1:
        for field in ("recurrent_parent_ids", "recurrent_boundary_ids"):
            rows = result[field]
            if not isinstance(rows, list) or len(rows) != ticks or any(
                not isinstance(row, list) or len(row) != 10 or any(not isinstance(value, str) or not value for value in row)
                for row in rows
            ):
                raise AI42GoShardError(f"{path}.{field} has incomplete lineage")
        for tick in range(ticks):
            if any(result["recurrent_parent_ids"][tick][slot] == result["recurrent_boundary_ids"][tick][slot] for slot in range(10)):
                raise AI42GoShardError(f"{path} has identical recurrent parent/boundary IDs")
            if tick and any(
                result["recurrent_parent_ids"][tick][slot] != result["recurrent_boundary_ids"][tick - 1][slot]
                for slot in range(10)
            ):
                raise AI42GoShardError(f"{path} has non-contiguous recurrent IDs")
    if result["split"] not in {"train", "validation"}:
        raise AI42GoShardError(f"{path}.split is invalid")
    return result


def _validate_go_manifest(
    manifest: Mapping[str, Any],
    expected_runtime_manifest_hash: str | bytes | None,
    *,
    allow_partial_schedule: bool = False,
) -> None:
    if frozenset(manifest) != _TOP_LEVEL_FIELDS:
        raise AI42GoShardError("Go manifest field set mismatch")
    if manifest["dataset_schema_version"] != "AI42-dataset-v1":
        raise AI42GoShardError("Go manifest dataset schema mismatch")
    schema_version = manifest["shard_schema_version"]
    if not isinstance(schema_version, str) or schema_version not in _SUPPORTED_SHARD_SCHEMAS:
        raise AI42GoShardError("Go manifest shard schema mismatch")
    if manifest["protocol_version"] != AI42_PROTOCOL_VERSION:
        raise AI42GoShardError("Go manifest protocol mismatch")
    if manifest["schema_hash"] != AI42_SCHEMA_HASH.hex() or manifest["reward_hash"] != AI42_REWARD_HASH.hex():
        raise AI42GoShardError("Go manifest protocol hash mismatch")
    if manifest["trajectory_schema_hash"] != AI42_TRAJECTORY_SCHEMA_HASH:
        raise AI42GoShardError("Go manifest trajectory schema mismatch")
    runtime = manifest["runtime_manifest"]
    runtime_hash = _hash_hex(manifest["runtime_manifest_hash"], "manifest.runtime_manifest_hash")
    if _runtime_manifest_hash(runtime) != runtime_hash:
        raise AI42GoShardError("Go manifest runtime manifest hash mismatch")
    if expected_runtime_manifest_hash is not None and runtime_hash != _hash_hex(
        expected_runtime_manifest_hash, "expected_runtime_manifest_hash"
    ):
        raise AI42GoShardError("Go runtime manifest hash mismatch")
    matches = manifest["matches"]
    shards = manifest["shards"]
    if not isinstance(matches, list) or not matches or not isinstance(shards, list) or not shards:
        raise AI42GoShardError("Go manifest must contain matches and shards")
    checked_matches = [
        _validate_match_entry(entry, "manifest.match", schema_version)
        for entry in matches
    ]
    match_ids = [entry["match_id"] for entry in checked_matches]
    if match_ids != sorted(match_ids) or len(set(match_ids)) != len(match_ids):
        raise AI42GoShardError("Go manifest matches must be unique and ordered")
    seen_shards: set[str] = set()
    for index, shard in enumerate(shards):
        path = f"manifest.shards[{index}]"
        if not isinstance(shard, Mapping) or frozenset(shard) != _SHARD_FIELDS:
            raise AI42GoShardError(f"{path} field set mismatch")
        name = shard["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or Path(name).is_absolute() or name in seen_shards:
            raise AI42GoShardError(f"{path}.name is invalid or duplicated")
        seen_shards.add(name)
        _hash_hex(shard["sha256"], f"{path}.sha256")
        if shard["compression"] not in _GO_SHARD_CODECS:
            raise AI42GoShardError(f"{path}.compression mismatch")
        _strict_int(shard["raw_bytes"], f"{path}.raw_bytes", minimum=1)
        _strict_int(shard["stored_bytes"], f"{path}.stored_bytes", minimum=1)
        _strict_int(shard["row_count"], f"{path}.row_count", minimum=1)
        if not isinstance(shard["match_ids"], list) or not shard["match_ids"] or shard["match_ids"] != sorted(shard["match_ids"]):
            raise AI42GoShardError(f"{path}.match_ids must be ordered")
        if any(not isinstance(match_id, str) or not match_id for match_id in shard["match_ids"]):
            raise AI42GoShardError(f"{path}.match_ids contains an invalid ID")
    if [shard["name"] for shard in shards] != sorted(seen_shards):
        raise AI42GoShardError("Go manifest shards must be canonically ordered")
    by_id = {entry["match_id"]: entry for entry in checked_matches}
    for shard in shards:
        ids = shard["match_ids"]
        if len(ids) != len(set(ids)):
            raise AI42GoShardError(f"Go manifest shard {shard['name']} contains duplicate match IDs")
        if any(match_id not in by_id for match_id in ids):
            raise AI42GoShardError(f"Go manifest shard {shard['name']} references an unknown match")
        expected = [entry["match_id"] for entry in checked_matches if entry["shard"] == shard["name"]]
        if ids != expected:
            raise AI42GoShardError(f"Go manifest shard match ordering mismatch for {shard['name']}")
        expected_offset = 0
        for match_id in ids:
            entry = by_id[match_id]
            if entry["row_offset"] != expected_offset:
                raise AI42GoShardError(f"Go manifest row offsets are not contiguous for {shard['name']}")
            expected_offset += entry["tick_count"]
        if shard["row_count"] != expected_offset:
            raise AI42GoShardError(f"Go manifest row_count mismatch for {shard['name']}")
    if {match_id for shard in shards for match_id in shard["match_ids"]} != set(match_ids):
        raise AI42GoShardError("Go manifest does not assign every match to exactly one shard")
    if {entry["match_id"] for entry in checked_matches if entry["split"] == "train"} & {
        entry["match_id"] for entry in checked_matches if entry["split"] == "validation"
    }:
        raise AI42GoShardError("Go manifest train/validation match leakage")
    scenario_mix = runtime.get("scenario_mix") if isinstance(runtime, Mapping) else None
    enforce_split = True
    if scenario_mix is not None:
        schedule = runtime.get("match_schedule") if isinstance(runtime, Mapping) else None
        if isinstance(schedule, list) and schedule:
            split_pairs = [
                (item["match_id"], item["scenario"])
                for item in schedule
                if isinstance(item, Mapping) and "match_id" in item and "scenario" in item
            ]
        else:
            split_pairs = [(entry["match_id"], entry["scenario"]) for entry in checked_matches]
        schedule_ids = {match_id for match_id, _ in split_pairs}
        if not allow_partial_schedule and set(match_ids) != schedule_ids:
            raise AI42GoShardError("Go manifest match set does not match the frozen schedule")
        enforce_split = not (allow_partial_schedule and len(checked_matches) < len(split_pairs))
        expected_splits = _deterministic_stratified_split(
            split_pairs,
            scenario_mix,
            manifest["split_seed"],
            validation_fraction=float(manifest["validation_fraction"]),
        )
    else:
        expected_splits = _deterministic_split(
            match_ids, float(manifest["validation_fraction"]), manifest["split_seed"]
        )
    # A staged Go generation contains one match while its runtime schedule
    # describes the complete dataset.  The merger assigns the frozen split;
    # enforce it here only once the manifest contains that complete schedule.
    if any(entry["match_id"] not in expected_splits for entry in checked_matches):
        raise AI42GoShardError("Go manifest contains a match outside the frozen schedule")
    if enforce_split and any(entry["split"] != expected_splits[entry["match_id"]] for entry in checked_matches):
        raise AI42GoShardError("Go manifest contains a non-deterministic split")


def _parse_header(payload: bytes, path: str) -> tuple[dict[str, Any], bytes]:
    if len(payload) < _HEADER_PREFIX.size:
        raise AI42GoShardError(f"{path} is truncated")
    magic, header_bytes = _HEADER_PREFIX.unpack_from(payload)
    schema_version = _SCHEMA_BY_MAGIC.get(magic)
    if schema_version is None:
        raise AI42GoShardError(f"{path} magic mismatch")
    if header_bytes < 2 or header_bytes > _MAX_HEADER_BYTES:
        raise AI42GoShardError(f"{path} header length is invalid")
    start = _HEADER_PREFIX.size
    end = start + header_bytes
    if end > len(payload):
        raise AI42GoShardError(f"{path} header is truncated")
    header = _strict_json(payload[start:end], f"{path}.header")
    if not isinstance(header, Mapping) or frozenset(header) != _GO_HEADER_FIELDS:
        raise AI42GoShardError(f"{path} header field set mismatch")
    compressed = payload[end:]
    if header["shard_schema_version"] != schema_version:
        raise AI42GoShardError(f"{path} shard schema mismatch")
    if header["codec"] not in _GO_SHARD_CODECS:
        raise AI42GoShardError(f"{path} codec mismatch")
    if _strict_int(header["stored_bytes"], f"{path}.header.stored_bytes") != len(compressed):
        raise AI42GoShardError(f"{path} stored byte count mismatch")
    if _sha256_bytes(compressed) != _hash_hex(header["payload_sha256"], f"{path}.payload_sha256"):
        raise AI42GoShardError(f"{path} payload hash mismatch")
    return dict(header), compressed


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise AI42GoShardError(f"cannot read {path}: {exc}") from exc
    return size, digest.hexdigest()


def _load_arrays(header: Mapping[str, Any], raw: bytes, path: str) -> dict[str, np.ndarray]:
    descriptors = header["arrays"]
    if not isinstance(descriptors, list) or [item.get("name") for item in descriptors if isinstance(item, Mapping)] != list(_ARRAY_NAMES):
        raise AI42GoShardError(f"{path}.arrays must contain the ordered AI42 array set")
    rows = sum(int(item["tick_count"]) for item in header["matches"])
    expected_offset = 0
    arrays: dict[str, np.ndarray] = {}
    for index, descriptor in enumerate(descriptors):
        name = f"{path}.arrays[{index}]"
        if not isinstance(descriptor, Mapping) or frozenset(descriptor) != _GO_ARRAY_FIELDS:
            raise AI42GoShardError(f"{name} field set mismatch")
        array_name = descriptor["name"]
        dtype_key = descriptor["dtype"]
        if dtype_key != _dtype_key(array_name) or dtype_key not in _GO_DTYPE_KEYS:
            raise AI42GoShardError(f"{name}.dtype mismatch")
        shape = descriptor["shape"]
        if not isinstance(shape, list) or tuple(shape) != _expected_shape(array_name, rows):
            raise AI42GoShardError(f"{name}.shape mismatch")
        offset = _strict_int(descriptor["offset"], f"{name}.offset")
        nbytes = _strict_int(descriptor["nbytes"], f"{name}.nbytes", minimum=1)
        if offset != expected_offset or offset + nbytes > len(raw):
            raise AI42GoShardError(f"{name} has a non-contiguous payload range")
        dtype = _GO_DTYPE_KEYS[dtype_key]
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected_nbytes:
            raise AI42GoShardError(f"{name}.nbytes mismatch")
        array = np.frombuffer(raw, dtype=dtype, count=expected_nbytes // dtype.itemsize, offset=offset)
        arrays[array_name] = array.reshape(tuple(shape))
        arrays[array_name].flags.writeable = False
        expected_offset += nbytes
    if expected_offset != len(raw) or _strict_int(header["raw_bytes"], f"{path}.raw_bytes") != len(raw):
        raise AI42GoShardError(f"{path} raw byte count mismatch")
    if _sha256_bytes(raw) != _hash_hex(header["raw_sha256"], f"{path}.raw_sha256"):
        raise AI42GoShardError(f"{path} raw payload hash mismatch")
    return arrays


@dataclass(slots=True)
class AI42GoDataset:
    """Lazy, bounded reader implementing the common dataset access surface."""

    root: Path
    manifest: Mapping[str, Any]
    _shards: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] | None = None

    def __post_init__(self) -> None:
        self._shards = {}

    def __len__(self) -> int:
        return len(self.manifest["matches"])

    @property
    def manifest_hash(self) -> str:
        return self.manifest["manifest_hash"]

    @property
    def runtime_manifest_hash(self) -> str:
        return self.manifest["runtime_manifest_hash"]

    @property
    def compression(self) -> str:
        codecs = {str(shard["compression"]) for shard in self.manifest["shards"]}
        return next(iter(codecs)) if len(codecs) == 1 else "mixed-deflate-raw"

    def match_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split not in (None, "train", "validation"):
            raise ValueError("split must be 'train', 'validation', or omitted")
        return tuple(entry["match_id"] for entry in self.manifest["matches"] if split is None or entry["split"] == split)

    def split_match_ids(self) -> dict[str, tuple[str, ...]]:
        """Return the frozen match IDs using the common dataset API."""

        return {split: self.match_ids(split) for split in ("train", "validation")}

    def iter_matches(self, split: str | None = None) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        """Yield one copied match at a time while retaining at most one shard.

        ``arrays_for_match`` copies only the requested row range.  The shard
        cache itself is replaced by ``_load_shard`` whenever iteration moves
        to another shard, so recurrent consumers cannot accumulate a
        generation's arrays in memory.
        """

        for match_id in self.match_ids(split):
            yield match_id, self.arrays_for_match(match_id)

    def _load_shard(self, name: str) -> Mapping[str, np.ndarray]:
        if self._shards and name in self._shards:
            return self._shards[name][1]
        shard = next((item for item in self.manifest["shards"] if item["name"] == name), None)
        if shard is None:
            raise AI42GoShardError(f"unknown Go shard {name!r}")
        path = self.root / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AI42GoShardError(f"cannot read {path}: {exc}") from exc
        if len(payload) != shard["stored_bytes"] or _sha256_bytes(payload) != shard["sha256"]:
            raise AI42GoShardError(f"Go shard file integrity mismatch for {name}")
        header, compressed = _parse_header(payload, str(path))
        if header["codec"] != shard["compression"]:
            raise AI42GoShardError(f"{path} codec does not match manifest compression")
        if header["shard_schema_version"] != self.manifest["shard_schema_version"]:
            raise AI42GoShardError(f"{path} shard schema mismatch")
        if header["protocol_version"] != self.manifest["protocol_version"]:
            raise AI42GoShardError(f"{path} protocol mismatch")
        for field in ("schema_hash", "reward_hash", "trajectory_schema_hash", "runtime_manifest_hash"):
            if header[field] != self.manifest[field]:
                raise AI42GoShardError(f"{path} {field} mismatch")
        if header["raw_bytes"] != shard["raw_bytes"]:
            raise AI42GoShardError(f"{path} raw byte metadata mismatch")
        header_matches = header["matches"]
        expected_matches = [entry for entry in self.manifest["matches"] if entry["shard"] == name]
        if header_matches != expected_matches:
            raise AI42GoShardError(f"{path} match metadata mismatch")
        try:
            decompressor = zlib.decompressobj(wbits=-15)
            raw = decompressor.decompress(compressed) + decompressor.flush()
        except zlib.error as exc:
            raise AI42GoShardError(f"{path} compressed payload is corrupt: {exc}") from exc
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            raise AI42GoShardError(f"{path} compressed payload has trailing or incomplete data")
        arrays = _load_arrays(header, raw, str(path))
        if self._shards:
            self._shards.clear()
        self._shards = {name: (header, arrays)}
        return arrays

    def arrays_for_match(self, match_id: str) -> dict[str, np.ndarray]:
        entry = next((item for item in self.manifest["matches"] if item["match_id"] == match_id), None)
        if entry is None:
            raise KeyError(match_id)
        arrays = self._load_shard(entry["shard"])
        start = int(entry["row_offset"])
        end = start + int(entry["tick_count"])
        return {name: np.array(value[start:end], copy=True) for name, value in arrays.items()}

    def match_metadata(self, match_id: str, *, derive: bool = False) -> dict[str, Any]:
        """Return compact match metadata, deriving v2 matrices only on request."""

        entry = next((item for item in self.manifest["matches"] if item["match_id"] == match_id), None)
        if entry is None:
            raise KeyError(match_id)
        result = dict(entry)
        if not derive or self.manifest["shard_schema_version"] == GO_SHARD_SCHEMA_VERSION_V1:
            return result
        first_step = int(result["first_step"])
        ticks = int(result["tick_count"])
        result["ticks"] = list(range(first_step, first_step + ticks))
        parents: list[list[str]] = []
        boundaries: list[list[str]] = []
        for ordinal in range(ticks):
            parent = [
                f"{match_id}:root:{slot:02d}" if ordinal == 0
                else f"{match_id}:boundary:{ordinal - 1}:{slot:02d}"
                for slot in range(len(result["hero_ids"]))
            ]
            boundary = [
                f"{match_id}:boundary:{ordinal}:{slot:02d}"
                for slot in range(len(result["hero_ids"]))
            ]
            parents.append(parent)
            boundaries.append(boundary)
        result["recurrent_parent_ids"] = parents
        result["recurrent_boundary_ids"] = boundaries
        return result


def compact_match_entry(entry: Mapping[str, Any], *, schema_version: str) -> dict[str, Any]:
    """Convert one validated v1 match entry to the compact v2 metadata form."""

    checked = _validate_match_entry(entry, "match", schema_version)
    if schema_version == GO_SHARD_SCHEMA_VERSION_V2:
        return checked
    compact = dict(checked)
    ticks = compact.pop("ticks")
    compact.pop("recurrent_parent_ids")
    compact.pop("recurrent_boundary_ids")
    compact["first_step"] = ticks[0]
    compact["recurrent_lineage_schema"] = GO_V2_LINEAGE_SCHEMA
    return compact


def load_go_dataset(
    root: str | Path,
    *,
    expected_runtime_manifest: Mapping[str, Any] | Any | None = None,
    expected_runtime_manifest_hash: str | bytes | None = None,
    allow_partial_schedule: bool = False,
) -> AI42GoDataset:
    """Load and verify one immutable Go shard generation without decompressing it."""

    path = Path(root)
    manifest_path = path / GO_MANIFEST_FILENAME
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        raise AI42GoShardError(f"cannot read {manifest_path}: {exc}") from exc
    manifest = _strict_json(payload, str(manifest_path))
    if not isinstance(manifest, Mapping):
        raise AI42GoShardError("Go manifest root must be an object")
    supplied = _hash_hex(manifest.get("manifest_hash", ""), "manifest.manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if hash_payload(unsigned) != supplied:
        raise AI42GoShardError("Go manifest hash mismatch")
    expected_hash = expected_runtime_manifest_hash
    if expected_runtime_manifest is not None:
        if hasattr(expected_runtime_manifest, "to_dict") and callable(expected_runtime_manifest.to_dict):
            expected_runtime_manifest = expected_runtime_manifest.to_dict()
        if not isinstance(expected_runtime_manifest, Mapping):
            raise AI42GoShardError("expected_runtime_manifest must be a mapping")
        expected_mapping = dict(expected_runtime_manifest)
        actual = _runtime_manifest_hash(expected_mapping)
        if expected_hash is not None and _hash_hex(expected_hash, "expected_runtime_manifest_hash") != actual:
            raise AI42GoShardError("expected runtime manifest/hash disagree")
        expected_hash = actual
    _validate_go_manifest(manifest, expected_hash, allow_partial_schedule=allow_partial_schedule)
    for shard in manifest["shards"]:
        shard_path = path / shard["name"]
        if shard_path.name != shard["name"] or Path(shard["name"]).is_absolute():
            raise AI42GoShardError(f"invalid Go shard path {shard['name']!r}")
        size, digest = _file_digest(shard_path)
        if size != shard["stored_bytes"] or digest != shard["sha256"]:
            raise AI42GoShardError(f"Go shard file integrity mismatch for {shard['name']}")
    return AI42GoDataset(path, dict(manifest))


__all__ = [
    "AI42GoDataset", "AI42GoShardError", "GO_MANIFEST_FILENAME",
    "GO_SHARD_CODEC", "GO_SHARD_MAGIC", "GO_SHARD_MAGIC_V1", "GO_SHARD_MAGIC_V2",
    "GO_SHARD_SCHEMA_VERSION", "GO_SHARD_SCHEMA_VERSION_V1", "GO_SHARD_SCHEMA_VERSION_V2",
    "GO_V2_LINEAGE_SCHEMA", "compact_match_entry",
    "load_go_dataset",
]
