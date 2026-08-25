from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib

from .test_ai42_go_shard import _generation
from tanat_ai40.compact_ai42_dataset import compact_ai42_dataset
from tanat_ai40.dataset_ai42 import AI42DatasetError
from tanat_ai40.go_shard_ai42 import (
    GO_SHARD_MAGIC_V2,
    GO_SHARD_SCHEMA_VERSION_V1,
    GO_SHARD_SCHEMA_VERSION_V2,
    _parse_header,
    load_go_dataset,
)
from tanat_ai40.trajectory_ai42 import canonical_json_bytes, hash_payload


def _write_v1_generation(root: Path, *, match_count: int = 1) -> tuple[bytes, dict[str, object]]:
    shard, manifest = _generation()
    if match_count > 1:
        header, compressed = _parse_header(shard, "fixture")
        raw = zlib.decompress(compressed, wbits=-15)
        raw_parts = []
        descriptors = []
        for descriptor in header["arrays"]:
            start = int(descriptor["offset"])
            end = start + int(descriptor["nbytes"])
            descriptor = dict(descriptor)
            descriptor["shape"] = [match_count, *descriptor["shape"][1:]]
            descriptor["offset"] = sum(int(item["nbytes"]) for item in descriptors)
            descriptor["nbytes"] = int(descriptor["nbytes"]) * match_count
            descriptors.append(descriptor)
            raw_parts.append(raw[start:end] * match_count)
        raw = b"".join(raw_parts)
        compressor = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15)
        compressed = compressor.compress(raw) + compressor.flush()
        base_match = manifest["matches"][0]
        matches = []
        for index in range(match_count):
            match = dict(base_match)
            match_id = f"ai42-match-{index:06d}"
            match.update({
                "match_id": match_id,
                "row_offset": index,
                "ticks": [index],
                "hero_ids": [f"{match_id}:hero:{slot:02d}" for slot in range(10)],
                "trajectory_ids": [f"{match_id}:trajectory:{slot:02d}" for slot in range(10)],
                "trajectory_hashes": [f"{index * 10 + slot + 1:064x}" for slot in range(10)],
                "recurrent_parent_ids": [[f"{match_id}:root:{slot:02d}" for slot in range(10)]],
                "recurrent_boundary_ids": [[f"{match_id}:boundary:0:{slot:02d}" for slot in range(10)]],
                "seed": index + 1,
            })
            matches.append(match)
        header = dict(header)
        header["raw_bytes"] = len(raw)
        header["stored_bytes"] = len(compressed)
        header["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        header["payload_sha256"] = hashlib.sha256(compressed).hexdigest()
        header["arrays"] = descriptors
        header["matches"] = matches
        header_bytes = canonical_json_bytes(header)
        shard = b"AI42GS1\0" + len(header_bytes).to_bytes(4, "little") + header_bytes + compressed
        manifest = dict(manifest)
        manifest["matches"] = matches
        manifest["validation_fraction"] = 0.0
        manifest["shards"] = [{
            "name": "shard-000000.a42",
            "sha256": hashlib.sha256(shard).hexdigest(),
            "match_ids": [match["match_id"] for match in matches],
            "row_count": match_count,
            "raw_bytes": len(raw),
            "stored_bytes": len(shard),
            "compression": "deflate-raw-6",
        }]
        manifest["manifest_hash"] = hash_payload({
            key: value for key, value in manifest.items() if key != "manifest_hash"
        })
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    (root / "shard-000000.a42").write_bytes(shard)
    return shard, manifest


class AI42MetadataV2Tests(unittest.TestCase):
    def test_v1_to_v2_is_deterministic_header_only_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-v1"
            source_shard, source_manifest = _write_v1_generation(source, match_count=16)
            first = root / "first-v2"
            second = root / "second-v2"
            with mock.patch(
                "tanat_ai40.go_shard_ai42.zlib.decompressobj",
                side_effect=AssertionError("metadata compaction decompressed payload"),
            ):
                compact_ai42_dataset(source, first)
                compact_ai42_dataset(source, second)

            self.assertEqual(
                (first / "manifest.json").read_bytes(),
                (second / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                (first / "shard-000000.a42").read_bytes(),
                (second / "shard-000000.a42").read_bytes(),
            )
            source_header, source_compressed = _parse_header(source_shard, "source")
            compacted_shard = (first / "shard-000000.a42").read_bytes()
            compacted_header, compacted_compressed = _parse_header(compacted_shard, "v2")
            self.assertEqual(compacted_shard[:8], GO_SHARD_MAGIC_V2)
            self.assertEqual(compacted_compressed, source_compressed)
            self.assertEqual(compacted_header["raw_sha256"], source_header["raw_sha256"])
            self.assertEqual(
                compacted_header["payload_sha256"], source_header["payload_sha256"]
            )
            self.assertLess(len(canonical_json_bytes(compacted_header)), len(canonical_json_bytes(source_header)) * 0.75)

            source_match = source_manifest["matches"][0]
            dataset = load_go_dataset(first)
            self.assertEqual(dataset.manifest["shard_schema_version"], GO_SHARD_SCHEMA_VERSION_V2)
            match = dataset.manifest["matches"][0]
            self.assertEqual(match["trajectory_hashes"], source_match["trajectory_hashes"])
            self.assertNotIn("ticks", match)
            self.assertNotIn("recurrent_parent_ids", match)
            self.assertNotIn("recurrent_boundary_ids", match)
            self.assertEqual(match["first_step"], 0)
            self.assertEqual(match["recurrent_lineage_schema"], "implicit-match-boundary-v1")
            self.assertNotIn("ticks", dataset.match_metadata("ai42-match-000000"))
            derived = dataset.match_metadata("ai42-match-000000", derive=True)
            self.assertEqual(derived["ticks"], [0])
            self.assertEqual(derived["recurrent_parent_ids"][0][0], "ai42-match-000000:root:00")
            self.assertEqual(derived["recurrent_boundary_ids"][0][0], "ai42-match-000000:boundary:0:00")

    def test_v1_reader_compatibility_and_v2_split_or_corruption_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_shard, _ = _write_v1_generation(root / "source-v1")
            source = load_go_dataset(root / "source-v1")
            self.assertEqual(source.manifest["shard_schema_version"], GO_SHARD_SCHEMA_VERSION_V1)
            destination = root / "v2"
            compact_ai42_dataset(root / "source-v1", destination)

            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["matches"][0]["split"] = "validation"
            manifest["manifest_hash"] = hash_payload({
                key: value for key, value in manifest.items() if key != "manifest_hash"
            })
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(AI42DatasetError):
                load_go_dataset(destination)

            compact_ai42_dataset(root / "source-v1", root / "v2-fresh")
            shard_path = root / "v2-fresh" / "shard-000000.a42"
            corrupted = bytearray(shard_path.read_bytes())
            corrupted[0] ^= 1
            shard_path.write_bytes(corrupted)
            with self.assertRaises(AI42DatasetError):
                load_go_dataset(root / "v2-fresh")

    def test_compactor_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_v1_generation(root / "source-v1")
            destination = root / "v2"
            compact_ai42_dataset(root / "source-v1", destination)
            with self.assertRaisesRegex(AI42DatasetError, "immutable"):
                compact_ai42_dataset(root / "source-v1", destination)


if __name__ == "__main__":
    unittest.main()
