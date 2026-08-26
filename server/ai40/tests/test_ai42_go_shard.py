from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import zlib

import numpy as np

from tanat_ai40.audit_ai42_dataset import audit_ai42_dataset
from tanat_ai40.dataset_ai42 import _ARRAY_DTYPES, _ARRAY_NAMES, load_dataset
from tanat_ai40.env import AI42_PROTOCOL_VERSION, AI42_REWARD_HASH, AI42_SCHEMA_HASH
from tanat_ai40.go_shard_ai42 import (
    GO_SHARD_CODEC,
    GO_SHARD_MAGIC_V1,
    GO_SHARD_SCHEMA_VERSION_V1,
    load_go_dataset,
)
from tanat_ai40.trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, canonical_json_bytes, hash_payload


def _shape(name: str, rows: int = 1) -> tuple[int, ...]:
    return {
        "hero": (rows, 10, 32),
        "abilities": (rows, 10, 4, 40),
        "entities": (rows, 10, 96, 16),
        "global": (rows, 10, 32),
        "entity_mask": (rows, 10, 96),
        "kind_mask": (rows, 10, 8),
        "target_mask": (rows, 10, 96),
        "skill_target_mask": (rows, 10, 4, 96),
        "teacher_status": (rows, 10),
        "teacher_action": (rows, 10),
        "projected_action": (rows, 10),
        "executed_action": (rows, 10),
        "executed_valid": (rows, 10),
        "rejection_reason": (rows, 10),
        "rewards": (rows, 10),
        "done": (rows,),
        "winner": (rows,),
        "step": (rows,),
        "elapsed": (rows,),
        "invalid": (rows, 10),
    }[name]


def _match(shard_name: str) -> dict[str, object]:
    heroes = [f"match-000000:hero:{index:02d}" for index in range(10)]
    return {
        "match_id": "ai42-match-000000",
        "split": "train",
        "shard": shard_name,
        "row_offset": 0,
        "tick_count": 1,
        "ticks": [0],
        "hero_ids": heroes,
        "trajectory_ids": [f"{hero}:trajectory" for hero in heroes],
        "trajectory_hashes": [f"{index + 1:064x}" for index in range(10)],
        "recurrent_parent_ids": [[f"root:{index}" for index in range(10)]],
        "recurrent_boundary_ids": [[f"boundary:0:{index}" for index in range(10)]],
        "seed": 1,
        "scenario": "ai30_mirror",
        "controller_by_slot": [2] * 10,
        "roster_ids": list(range(10)),
        "side_by_slot": [0] * 5 + [1] * 5,
    }


def _generation(
    *, codec: str = GO_SHARD_CODEC, compression_level: int = 3,
) -> tuple[bytes, dict[str, object]]:
    shard_name = "shard-000000.a42"
    arrays = {
        name: np.zeros(_shape(name), dtype=_ARRAY_DTYPES[name])
        for name in _ARRAY_NAMES
    }
    arrays["done"][0] = 1
    raw = b"".join(np.ascontiguousarray(arrays[name]).tobytes() for name in _ARRAY_NAMES)
    descriptors = []
    offset = 0
    for name in _ARRAY_NAMES:
        value = arrays[name]
        nbytes = value.nbytes
        dtype = "action" if _ARRAY_DTYPES[name] == _ARRAY_DTYPES["teacher_action"] else (
            "u1" if value.dtype == np.dtype("u1") else np.dtype(value.dtype).str
        )
        descriptors.append({
            "name": name,
            "dtype": dtype,
            "shape": list(value.shape),
            "offset": offset,
            "nbytes": nbytes,
        })
        offset += nbytes
    compressor = zlib.compressobj(level=compression_level, method=zlib.DEFLATED, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    match = _match(shard_name)
    header = {
        "shard_schema_version": GO_SHARD_SCHEMA_VERSION_V1,
        "protocol_version": AI42_PROTOCOL_VERSION,
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
        "runtime_manifest_hash": hash_payload({"fixture": "go"}),
        "codec": codec,
        "raw_bytes": len(raw),
        "stored_bytes": len(compressed),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": hashlib.sha256(compressed).hexdigest(),
        "arrays": descriptors,
        "matches": [match],
    }
    header_bytes = canonical_json_bytes(header)
    shard = GO_SHARD_MAGIC_V1 + len(header_bytes).to_bytes(4, "little") + header_bytes + compressed
    manifest = {
        "dataset_schema_version": "AI42-dataset-v1",
        "shard_schema_version": GO_SHARD_SCHEMA_VERSION_V1,
        "protocol_version": AI42_PROTOCOL_VERSION,
        "schema_hash": AI42_SCHEMA_HASH.hex(),
        "reward_hash": AI42_REWARD_HASH.hex(),
        "trajectory_schema_hash": AI42_TRAJECTORY_SCHEMA_HASH,
        "runtime_manifest_hash": hash_payload({"fixture": "go"}),
        "runtime_manifest": {"fixture": "go"},
        "split_seed": 42,
        "validation_fraction": 0.2,
        "matches": [match],
        "shards": [{
            "name": shard_name,
            "sha256": hashlib.sha256(shard).hexdigest(),
            "match_ids": [match["match_id"]],
            "row_count": 1,
            "raw_bytes": len(raw),
            "stored_bytes": len(shard),
            "compression": codec,
        }],
    }
    manifest["manifest_hash"] = hash_payload(manifest)
    return shard, manifest


class AI42GoShardTests(unittest.TestCase):
    def test_legacy_level_6_generation_remains_readable(self) -> None:
        shard, manifest = _generation(codec="deflate-raw-6", compression_level=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            (root / "shard-000000.a42").write_bytes(shard)
            dataset = load_go_dataset(root)
            self.assertEqual(dataset.compression, "deflate-raw-6")
            self.assertEqual(int(dataset.arrays_for_match("ai42-match-000000")["done"][0]), 1)

    def test_deterministic_lazy_decode_and_audit(self) -> None:
        first_shard, first_manifest = _generation()
        second_shard, second_manifest = _generation()
        self.assertEqual(first_shard, second_shard)
        self.assertEqual(first_manifest, second_manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(canonical_json_bytes(first_manifest))
            (root / "shard-000000.a42").write_bytes(first_shard)
            dataset = load_go_dataset(root)
            self.assertEqual(dataset._shards, {})
            generic = load_dataset(root)
            self.assertEqual(generic.manifest_hash, dataset.manifest_hash)
            self.assertEqual(generic.match_ids(), dataset.match_ids())
            arrays = dataset.arrays_for_match("ai42-match-000000")
            self.assertEqual(len(dataset._shards), 1)
            self.assertEqual(int(arrays["done"][0]), 1)
            self.assertFalse(dataset._shards["shard-000000.a42"][1]["hero"].flags.writeable)
            report = audit_ai42_dataset(root, format="go")
            self.assertEqual(report["format"], "go")
            self.assertEqual(report["matches"], 1)
            self.assertEqual(report["ticks"], 1)
            self.assertEqual(report["bytes"]["compression"], GO_SHARD_CODEC)

    def test_corruption_rejected_before_or_during_lazy_decode(self) -> None:
        shard, manifest = _generation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            corrupted = bytearray(shard)
            corrupted[-1] ^= 0x01
            (root / "shard-000000.a42").write_bytes(corrupted)
            with self.assertRaises(ValueError):
                load_go_dataset(root)


if __name__ == "__main__":
    unittest.main()
