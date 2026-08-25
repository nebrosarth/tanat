from __future__ import annotations

import io
from pathlib import Path
from unittest import mock
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import json
import tempfile
import unittest

from .test_ai42_go_shard import _generation

from tanat_ai40.build_ai42_dataset import build_match_specs
from tanat_ai40.build_ai42_dataset_go import (
    COMPLETE_MARKER,
    build_ai42_dataset_go,
    build_go_schedule,
)
from tanat_ai40.dataset_ai42 import (
    AI42DatasetError,
    _deterministic_split,
    _deterministic_stratified_split,
    load_dataset,
)
from tanat_ai40.go_shard_ai42 import AI42GoDataset, _parse_header, load_go_dataset
from tanat_ai40.trajectory_ai42 import canonical_json_bytes, hash_payload


def _fake_generation(spec, schedule, root: Path) -> None:
    source_shard, source_manifest = _generation()
    source_header, compressed = _parse_header(source_shard, "fixture")
    heroes = [f"{spec.match_id}:hero:{index:02d}" for index in range(10)]
    match = dict(source_manifest["matches"][0])
    match.update({
        "match_id": spec.match_id,
        "shard": "shard-000000.a42",
        "ticks": [0],
        "hero_ids": heroes,
        "trajectory_ids": [f"{spec.match_id}:trajectory:{index:02d}" for index in range(10)],
        "trajectory_hashes": [f"{index + 1:064x}" for index in range(10)],
        "recurrent_parent_ids": [[f"{spec.match_id}:root:{index:02d}" for index in range(10)]],
        "recurrent_boundary_ids": [[f"{spec.match_id}:boundary:0:{index:02d}" for index in range(10)]],
        "seed": spec.seed,
        "scenario": spec.scenario,
        "controller_by_slot": list(spec.controller_by_slot),
        "roster_ids": list(spec.roster_ids),
        "side_by_slot": list(spec.side_by_slot),
        "split": "train",
    })
    source_header.update({
        "runtime_manifest_hash": hash_payload(schedule),
        "matches": [match],
    })
    header_bytes = canonical_json_bytes(source_header)
    shard = b"AI42GS1\0" + len(header_bytes).to_bytes(4, "little") + header_bytes + compressed
    shard_name = "shard-000000.a42"
    raw_bytes = source_header["raw_bytes"]
    manifest = dict(source_manifest)
    manifest.update({
        "runtime_manifest_hash": hash_payload(schedule),
        "runtime_manifest": dict(schedule),
        "validation_fraction": schedule["validation_fraction"],
        "split_seed": schedule["split_seed"],
        "matches": [match],
        "shards": [{
            "name": shard_name,
            "sha256": hashlib.sha256(shard).hexdigest(),
            "match_ids": [spec.match_id],
            "row_count": 1,
            "raw_bytes": raw_bytes,
            "stored_bytes": len(shard),
            "compression": "deflate-raw-6",
        }],
    })
    manifest["manifest_hash"] = hash_payload({key: value for key, value in manifest.items() if key != "manifest_hash"})
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    (root / shard_name).write_bytes(shard)


class AI42GoBuilderTests(unittest.TestCase):
    def _build_two_match_v2(self, root: Path) -> Path:
        destination = root / "dataset"
        build_ai42_dataset_go(
            destination,
            match_count=2,
            seed=420000,
            workers=1,
            in_flight=1,
            max_steps=1,
            timeout_minutes=1 / 300,
            validation_fraction=0.5,
            staging=root / "stage",
            runner=_fake_generation,
        )
        return destination

    def test_generic_v2_loader_streams_splits_and_one_shard(self) -> None:
        """The generic API must preserve v2's lazy Go reader contract."""

        with tempfile.TemporaryDirectory() as directory:
            destination = self._build_two_match_v2(Path(directory))
            dataset = load_dataset(destination)
            self.assertIsInstance(dataset, AI42GoDataset)
            self.assertEqual(
                {split: len(match_ids) for split, match_ids in dataset.split_match_ids().items()},
                {"train": 1, "validation": 1},
            )
            self.assertEqual(dataset._shards, {})
            for split in ("train", "validation"):
                match_id, arrays = next(dataset.iter_matches(split))
                self.assertIn(match_id, dataset.split_match_ids()[split])
                self.assertEqual(int(arrays["done"][-1]), 1)
                self.assertLessEqual(len(dataset._shards), 1)

    def test_v2_recurrent_batches_and_validation_only_bc_preserve_parameters(self) -> None:
        try:
            import torch
            from tanat_ai40.learner_ai42 import iter_ai42_dataset_batches
            from tanat_ai40.train_ai42_bc import main as train_ai42_bc
        except ImportError as exc:  # pragma: no cover - dependency-specific CI gate
            self.skipTest(f"AI-42 learner tests require torch: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            destination = self._build_two_match_v2(Path(directory))
            dataset = load_dataset(destination)
            for split in ("train", "validation"):
                batch = next(iter(iter_ai42_dataset_batches(
                    dataset, split=split, sequence_length=1, batch_size=1,
                )))
                self.assertEqual((batch.batch_size, batch.sequence_length), (1, 1))
                self.assertLessEqual(len(dataset._shards), 1)

            # This is the existing validation-only CLI path.  It may build a
            # graph and inspect gradients, but an optimizer update is forbidden.
            arguments = [
                "--dataset", str(destination), "--device", "cpu",
                "--hidden-size", "8", "--model-width", "8",
                "--entity-layers", "1", "--num-heads", "2",
                "--ff-multiplier", "1", "--timing-bins", "2",
                "--sequence-length", "1", "--batch-size", "1",
            ]
            output = io.StringIO()
            with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")):
                with redirect_stdout(output):
                    self.assertEqual(train_ai42_bc(arguments), 0)
            report = json.loads(output.getvalue())
            self.assertTrue(report["dataset"]["provided"])
            self.assertEqual(report["dataset"]["train_matches"], 1)
            self.assertEqual(report["dataset"]["validation_matches"], 1)
            self.assertTrue(report["parameters_unchanged"])

    def test_scenario_mix_exact_counts_determinism_and_worker_independence(self) -> None:
        mix = {
            "ai30_mirror": {"train": 3, "validation": 1},
            "ai30_vs_ai20": {"train": 1, "validation": 1},
            "ai20_vs_ai30": {"train": 2, "validation": 0},
        }
        schedule_a, specs_a = build_go_schedule(
            seed=420000, match_count=8, max_steps=4500, timeout_minutes=15.0,
            scenario="ai30_mirror", scenario_mix=mix,
            validation_fraction=0.25, split_seed=42,
        )
        schedule_b, specs_b = build_go_schedule(
            seed=420000, match_count=8, max_steps=4500, timeout_minutes=15.0,
            scenario="ai30_mirror", scenario_mix=mix,
            validation_fraction=0.25, split_seed=42,
        )
        self.assertEqual(canonical_json_bytes(schedule_a), canonical_json_bytes(schedule_b))
        self.assertEqual(tuple(spec.scenario for spec in specs_a), tuple(spec.scenario for spec in specs_b))
        assignments = _deterministic_stratified_split(
            [(spec.match_id, spec.scenario) for spec in specs_a], mix, 42,
            validation_fraction=0.25,
        )
        counts = Counter((spec.scenario, assignments[spec.match_id]) for spec in specs_a)
        self.assertEqual(
            counts,
            Counter({
                ("ai30_mirror", "train"): 3,
                ("ai30_mirror", "validation"): 1,
                ("ai30_vs_ai20", "train"): 1,
                ("ai30_vs_ai20", "validation"): 1,
                ("ai20_vs_ai30", "train"): 2,
            }),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = []
            for workers in (1, 2):
                destination = root / f"dataset-{workers}"
                build_ai42_dataset_go(
                    destination, match_count=8, seed=420000, workers=workers,
                    in_flight=workers, max_steps=4500, timeout_minutes=15.0,
                    scenario_mix=mix, validation_fraction=0.25,
                    staging=root / f"stage-{workers}", runner=_fake_generation,
                )
                destinations.append(destination)
            self.assertEqual(
                (destinations[0] / "manifest.json").read_bytes(),
                (destinations[1] / "manifest.json").read_bytes(),
            )
            manifest = json.loads((destinations[0] / "manifest.json").read_text(encoding="utf-8"))
            published_counts = Counter((entry["scenario"], entry["split"]) for entry in manifest["matches"])
            self.assertEqual(published_counts, counts)

    def test_production_scenario_mix_config_is_exact(self) -> None:
        config_path = Path(__file__).parents[1] / "config" / "ai42_dataset01.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["match_count"], 320)
        self.assertEqual(config["seed"], 420000)
        self.assertEqual(config["split_seed"], 42)
        self.assertEqual(config["workers"], 8)
        self.assertEqual(config["in_flight"], 8)
        self.assertEqual(config["max_steps"], 4500)
        self.assertEqual(config["timeout_minutes"], 15.0)
        self.assertEqual(config["validation_fraction"], 0.125)
        self.assertEqual(config["scenario_mix"], {
            "ai30_mirror": {"train": 210, "validation": 30},
            "ai30_vs_ai20": {"train": 35, "validation": 5},
            "ai20_vs_ai30": {"train": 35, "validation": 5},
        })

    def test_worker_count_determinism_split_and_both_side_directions(self) -> None:
        scenarios = ("ai30_vs_ai20", "ai20_vs_ai30", "ai30_vs_ai20", "ai20_vs_ai30")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for workers in (1, 2):
                destination = root / f"dataset-{workers}"
                with mock.patch("tanat_ai40.go_shard_ai42.zlib.decompressobj", side_effect=AssertionError("merge decompressed")):
                    summary = build_ai42_dataset_go(
                        destination, match_count=4, seed=420001, workers=workers,
                        in_flight=workers, max_steps=1, timeout_minutes=1 / 300,
                        scenario=scenarios, validation_fraction=0.5,
                        staging=root / f"stage-{workers}",
                        runner=_fake_generation,
                    )
                self.assertFalse(summary["decompressed_during_merge"])
                outputs.append(destination)
            self.assertEqual(
                (outputs[0] / "manifest.json").read_bytes(),
                (outputs[1] / "manifest.json").read_bytes(),
            )
            for left, right in zip(sorted(outputs[0].glob("shard-*.a42")), sorted(outputs[1].glob("shard-*.a42"))):
                self.assertEqual(left.read_bytes(), right.read_bytes())
            dataset = load_go_dataset(outputs[0])
            self.assertEqual(dataset.manifest["shard_schema_version"], "AI42-go-shard-v2")
            assignments = _deterministic_split(dataset.match_ids(), 0.5, 42)
            self.assertEqual(
                {entry["match_id"]: entry["split"] for entry in dataset.manifest["matches"]},
                assignments,
            )
            for index, entry in enumerate(dataset.manifest["matches"]):
                self.assertEqual(entry["shard"], f"shard-{index:06d}.a42")
                self.assertNotIn("ticks", entry)
                self.assertNotIn("recurrent_parent_ids", entry)
                self.assertNotIn("recurrent_boundary_ids", entry)
                self.assertEqual(entry["recurrent_lineage_schema"], "implicit-match-boundary-v1")
                arrays = dataset.arrays_for_match(entry["match_id"])
                self.assertEqual(int(arrays["done"][-1]), 1)
                self.assertLessEqual(len(dataset._shards), 1)
            self.assertEqual(
                [entry["controller_by_slot"][:5] for entry in dataset.manifest["matches"]],
                [[2] * 5, [1] * 5, [2] * 5, [1] * 5],
            )

    def test_resume_and_corrupt_staging_rejection(self) -> None:
        calls: list[str] = []
        fail_once = {"ai42-match-000001"}

        def flaky(spec, schedule, root):
            calls.append(spec.match_id)
            if spec.match_id in fail_once:
                fail_once.remove(spec.match_id)
                raise RuntimeError("injected Go failure")
            _fake_generation(spec, schedule, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            with self.assertRaises(RuntimeError):
                build_ai42_dataset_go(
                    root / "dataset", match_count=2, workers=1,
                    max_steps=1, timeout_minutes=1 / 300, staging=staging,
                    runner=flaky,
                )
            self.assertTrue((staging / "matches" / "match-000000" / "COMPLETE").exists())
            shard = next((staging / "matches" / "match-000000").glob("shard-*.a42"))
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 1
            shard.write_bytes(payload)
            with self.assertRaises(AI42DatasetError):
                build_ai42_dataset_go(
                    root / "dataset", match_count=2, workers=1,
                    max_steps=1, timeout_minutes=1 / 300, staging=staging,
                    runner=_fake_generation,
                )

    def test_split_tampering_and_max_steps_override_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule, specs = build_go_schedule(
                seed=7, match_count=2, max_steps=1, timeout_minutes=1 / 300,
                scenario="ai30_mirror", validation_fraction=0.5, split_seed=42,
            )
            destination = root / "dataset"
            build_ai42_dataset_go(
                destination, match_count=2, seed=7, workers=1, max_steps=1,
                timeout_minutes=1 / 300, validation_fraction=0.5,
                staging=root / "stage", runner=_fake_generation,
            )
            manifest_path = destination / "manifest.json"
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            manifest["matches"][0]["split"] = "validation" if manifest["matches"][0]["split"] == "train" else "train"
            manifest["manifest_hash"] = hash_payload({key: value for key, value in manifest.items() if key != "manifest_hash"})
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(AI42DatasetError):
                load_go_dataset(destination)
            with self.assertRaisesRegex(ValueError, "max-steps override"):
                build_ai42_dataset_go(
                    root / "other", go_command=("go", "run", "-max-steps=1"),
                    max_steps=1, timeout_minutes=1 / 300, runner=_fake_generation,
                )

    def test_production_module_has_no_training_imports(self) -> None:
        source = Path(__file__).parents[1] / "src" / "tanat_ai40" / "build_ai42_dataset_go.py"
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("import torch", text)
        self.assertNotIn("learner", text)
        self.assertNotIn("optimizer", text)


if __name__ == "__main__":
    unittest.main()
