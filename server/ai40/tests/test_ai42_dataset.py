from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tanat_ai40.bridge_ai42 import REJECTION_REASON_MASKED, TEACHER_STATUS_ACTION, TEACHER_STATUS_UNAVAILABLE
from tanat_ai40.collect_ai42 import AI42Collector, collect_ai42_match
from tanat_ai40.dataset_ai42 import AI42DatasetError, load_dataset, write_dataset
from tanat_ai40.env import ACTION_DTYPE, AI42_REWARD_HASH, AI42_SCHEMA_HASH, HERO_COUNT, StepResult, HeroAction
from tanat_ai40.trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, Outcome, canonical_json_bytes, hash_payload


class AI42DatasetTest(unittest.TestCase):
    @staticmethod
    def _result(step: int, *, done: bool = False) -> StepResult:
        teacher = np.zeros(HERO_COUNT, dtype=ACTION_DTYPE)
        teacher[0] = (3, 4, 40, 0)
        executed = np.zeros(HERO_COUNT, dtype=ACTION_DTYPE)
        executed[0] = (3, 4, 40, 0)
        statuses = np.full(HERO_COUNT, TEACHER_STATUS_UNAVAILABLE, dtype=np.uint8)
        statuses[0] = TEACHER_STATUS_ACTION
        valid = np.ones(HERO_COUNT, dtype=np.uint8)
        valid[1] = 0
        reasons = np.zeros(HERO_COUNT, dtype=np.uint8)
        reasons[1] = REJECTION_REASON_MASKED
        result = StepResult(
            step, 0.2 * step, done, 1 if done else -1,
            np.zeros(HERO_COUNT, dtype=np.uint8), np.full(HERO_COUNT, step + 0.5, dtype="<f4"),
            np.full((HERO_COUNT, 32), step, dtype="<f4"),
            np.full((HERO_COUNT, 96, 16), step, dtype="<f4"),
            np.full((HERO_COUNT, 32), step, dtype="<f4"),
            np.ones((HERO_COUNT, 96), dtype=np.uint8),
            np.ones((HERO_COUNT, 8), dtype=np.uint8),
            np.ones((HERO_COUNT, 96), dtype=np.uint8),
            np.ones((HERO_COUNT, 4, 96), dtype=np.uint8),
            np.full((HERO_COUNT, 4, 40), step, dtype="<f4"),
            None, None, teacher, statuses, executed, valid, reasons,
            AI42_SCHEMA_HASH, AI42_REWARD_HASH,
        )
        return result

    @classmethod
    def _capture(cls, match_id: str, *, ticks: int = 2):
        results = [cls._result(step, done=step == ticks - 1) for step in range(ticks)]
        submitted = [[HeroAction(kind=1, direction=40)] + [HeroAction() for _ in range(9)] for _ in results]
        boundaries = [[f"boundary:{step}:{hero}" for hero in range(10)] for step in range(ticks)]
        parents = [[f"root:{hero}" for hero in range(10)]] + [list(boundaries[step - 1]) for step in range(1, ticks)]
        outcomes = [
            [Outcome(reward=float(result.rewards[hero]), terminal=result.done, winner=result.winner) for hero in range(10)]
            for result in results
        ]
        runtime_manifest = {"runtime": "test", "version": 13}
        return collect_ai42_match(
            results, submitted, parents, outcomes,
            recurrent_boundaries=boundaries,
            hero_ids=[f"hero-{hero}" for hero in range(10)], match_id=match_id,
            runtime_manifest=runtime_manifest, expected_ticks=ticks,
        )

    def test_round_trip_and_deterministic_shards(self):
        first = self._capture("match-b")
        second = self._capture("match-a")
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            dataset_left = write_dataset(left, [first, second], shard_size=1, validation_fraction=0.5, split_seed=77)
            dataset_right = write_dataset(right, [second, first], shard_size=1, validation_fraction=0.5, split_seed=77)
            self.assertEqual(dataset_left.manifest_hash, dataset_right.manifest_hash)
            self.assertEqual(dataset_left.match_ids(), ("match-a", "match-b"))
            self.assertEqual((Path(left) / "manifest.json").read_bytes(), (Path(right) / "manifest.json").read_bytes())
            self.assertEqual((Path(left) / "shard-000000.npz").read_bytes(), (Path(right) / "shard-000000.npz").read_bytes())
            loaded = load_dataset(left, expected_runtime_manifest={"runtime": "test", "version": 13})
            arrays = loaded.arrays_for_match("match-a")
            self.assertEqual(arrays["hero"].dtype, np.dtype("<f4"))
            self.assertEqual(arrays["hero"].shape, (2, 10, 32))
            self.assertEqual(arrays["teacher_action"].dtype, ACTION_DTYPE)
            self.assertEqual(arrays["projected_action"][0, 0]["direction"], 40)
            self.assertEqual(arrays["rejection_reason"][0, 1], REJECTION_REASON_MASKED)

    def test_corruption_hash_schema_and_trailing_rejection(self):
        capture = self._capture("match")
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(directory, [capture])
            shard = Path(directory) / "shard-000000.npz"
            original = shard.read_bytes()
            shard.write_bytes(original + b"trailing")
            with self.assertRaisesRegex(AI42DatasetError, "shard hash mismatch"):
                load_dataset(directory)
            manifest = Path(directory) / "manifest.json"
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_value["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
            manifest_value["manifest_hash"] = hash_payload(
                {key: value for key, value in manifest_value.items() if key != "manifest_hash"}
            )
            manifest.write_bytes(canonical_json_bytes(manifest_value))
            with self.assertRaisesRegex(AI42DatasetError, "trailing bytes"):
                load_dataset(directory)
            shard.write_bytes(original)
            manifest_value["shards"][0]["sha256"] = hashlib.sha256(original).hexdigest()
            manifest_value["schema_hash"] = "0" * 64
            manifest_value["manifest_hash"] = hash_payload(
                {key: value for key, value in manifest_value.items() if key != "manifest_hash"}
            )
            manifest.write_bytes(canonical_json_bytes(manifest_value))
            with self.assertRaisesRegex(AI42DatasetError, "schema_hash mismatch"):
                load_dataset(directory)

    def test_incomplete_tick_and_nonfinite_rejected(self):
        capture = self._capture("match")
        with self.assertRaisesRegex(Exception, "contiguous"):
            collect_ai42_match(
                [self._result(0), self._result(2, done=True)],
                [[HeroAction() for _ in range(10)] for _ in range(2)],
                [[f"root:{hero}" for hero in range(10)], [f"boundary:0:{hero}" for hero in range(10)]],
                [[Outcome() for _ in range(10)] for _ in range(2)],
                recurrent_boundaries=[
                    [f"boundary:0:{hero}" for hero in range(10)],
                    [f"boundary:2:{hero}" for hero in range(10)],
                ],
                hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="bad-ticks",
                runtime_manifest={"runtime": "test"},
            )
        bad_result = self._result(0)
        bad_result.hero[0, 0] = np.nan
        with self.assertRaisesRegex(AI42DatasetError, "non-finite"):
            collect_ai42_match(
                [bad_result], [[HeroAction() for _ in range(10)]],
                [[f"root:{hero}" for hero in range(10)]], [[Outcome() for _ in range(10)]],
                recurrent_boundaries=[[f"boundary:{hero}" for hero in range(10)]],
                hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="bad",
                runtime_manifest={"runtime": "test"},
            )

    def test_split_is_match_disjoint_and_hashes_are_recorded(self):
        captures = [self._capture(f"match-{index}") for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            loaded = write_dataset(directory, captures, shard_size=2, validation_fraction=0.34, split_seed=3)
            train = set(loaded.match_ids("train"))
            validation = set(loaded.match_ids("validation"))
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(loaded.match_ids()))
            self.assertEqual(len(loaded.manifest["shards"]), 3)
            for shard in loaded.manifest["shards"]:
                self.assertEqual(len(shard["sha256"]), 64)
            self.assertEqual(loaded.runtime_manifest_hash, hash_payload({"runtime": "test", "version": 13}))

    def test_incremental_collector_snapshots_inputs_and_requires_terminal_tick(self):
        collector = AI42Collector(
            match_id="snapshot",
            hero_ids=tuple(f"hero-{hero}" for hero in range(10)),
            runtime_manifest={"runtime": "test", "version": 13},
            expected_ticks=1,
        )
        result = self._result(0, done=True)
        submitted = [HeroAction(kind=1, direction=40)] + [HeroAction() for _ in range(9)]
        parents = [f"root:{hero}" for hero in range(10)]
        boundaries = [f"boundary:{hero}" for hero in range(10)]
        outcomes = [Outcome(reward=float(result.rewards[hero]), terminal=True, winner=1) for hero in range(10)]
        collector.record_tick(result, submitted, parents, boundaries, outcomes)
        result.hero.fill(99)
        submitted[0].kind = 0
        capture = collector.finish()
        with tempfile.TemporaryDirectory() as directory:
            arrays = write_dataset(directory, [capture]).arrays_for_match("snapshot")
            self.assertFalse(np.all(arrays["hero"] == 99))
            self.assertEqual(int(arrays["projected_action"][0, 0]["kind"]), 1)

        incomplete = AI42Collector(
            match_id="incomplete",
            hero_ids=tuple(f"hero-{hero}" for hero in range(10)),
            runtime_manifest={"runtime": "test", "version": 13},
        )
        result = self._result(0, done=False)
        incomplete.record_tick(result, submitted, parents, boundaries, [Outcome() for _ in range(10)])
        with self.assertRaisesRegex(AI42DatasetError, "terminal"):
            incomplete.finish()

    def test_raw_authoritative_fields_must_match_trajectory(self):
        capture = self._capture("mismatch")
        capture.done[0] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AI42DatasetError, "disagrees with the authoritative trajectory"):
                write_dataset(directory, [capture])
        direct = self._capture("direct").output
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AI42DatasetError, "lacks authoritative raw"):
                write_dataset(directory, [direct])


if __name__ == "__main__":
    unittest.main()
