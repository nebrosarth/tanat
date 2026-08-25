from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tanat_ai40.audit_ai42_dataset import audit_ai42_dataset
from tanat_ai40.build_ai42_dataset import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TIMEOUT_MINUTES,
    build_ai42_dataset,
    build_match_specs,
    deterministic_seed_schedule,
    validate_timeout,
)
from tanat_ai40.dataset_ai42 import AI42DatasetError, AI42DatasetStaging, load_dataset
from tanat_ai40.env import ACTION_DTYPE, AI42_REWARD_HASH, AI42_SCHEMA_HASH, HERO_COUNT, StepResult
from tanat_ai40.bridge_ai42 import (
    REJECTION_REASON_INVALID,
    REJECTION_REASON_MASKED,
    REJECTION_REASON_NONE,
    REJECTION_REASON_POLICY_ERROR,
    REJECTION_REASON_SAFETY,
    REJECTION_REASON_SERVER_REJECTED,
    REJECTION_REASON_TIMEOUT,
    REJECTION_REASON_UNKNOWN,
    TEACHER_STATUS_ACTION,
    TEACHER_STATUS_CANCEL,
    TEACHER_STATUS_HOLD,
    TEACHER_STATUS_NONE,
    TEACHER_STATUS_UNAVAILABLE,
    TEACHER_STATUS_WAIT,
)
from tanat_ai40.trajectory_ai42 import canonical_json_bytes, hash_payload


def _result(step: int, *, done: bool) -> StepResult:
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
    return StepResult(
        step, step * 0.2, done, 1 if done else -1,
        np.zeros(HERO_COUNT, dtype=np.uint8), np.full(HERO_COUNT, step, dtype="<f4"),
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


class _FakeVectorEnv:
    def __init__(self, workers: int, *, fail: bool = False) -> None:
        self.workers = workers
        self.fail = fail
        self.cursors = [0] * workers
        self.closed = False

    def __enter__(self) -> "_FakeVectorEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def reset(self, seeds, **kwargs):
        if self.fail:
            raise RuntimeError("injected reset failure")
        self.cursors = [0] * self.workers
        return [_result(0, done=False) for _ in range(self.workers)]

    def step(self, actions):
        values = []
        for worker in range(self.workers):
            tick = self.cursors[worker]
            values.append(_result(tick, done=tick == 1))
            self.cursors[worker] += 1
        return values


class AI42DatasetBuilderTest(unittest.TestCase):
    def test_seed_schedule_and_timeout_contract(self):
        self.assertEqual(deterministic_seed_schedule(7, 4), deterministic_seed_schedule(7, 4))
        self.assertEqual(len(set(deterministic_seed_schedule(7, 100))), 100)
        self.assertEqual(validate_timeout(DEFAULT_MAX_STEPS), DEFAULT_MAX_STEPS)
        self.assertEqual(DEFAULT_TIMEOUT_MINUTES, 15.0)
        self.assertEqual(validate_timeout(600, 2.0), 600)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_timeout(599, 2.0)

    def test_specs_provide_controller_roster_side_provenance(self):
        specs = build_match_specs(42, 2, ("ai30_mirror", "ai30_vs_ai20"))
        self.assertEqual(specs[0].controller_by_slot, (2,) * 10)
        self.assertEqual(specs[1].controller_by_slot, (2,) * 5 + (1,) * 5)
        self.assertEqual(len(specs[0].roster_ids), 10)
        self.assertEqual(specs[0].side_by_slot, (0,) * 5 + (1,) * 5)

    def test_build_is_deterministic_and_audit_is_lazy_contract_checked(self):
        def factory(executable, workers, protocol):
            return _FakeVectorEnv(workers)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left"
            right = root / "right"
            first = build_ai42_dataset(
                "unused", left, match_count=2, seed=101, workers=2,
                max_steps=2, timeout_minutes=2 / 300, validation_fraction=0.5,
                env_factory=factory,
            )
            second = build_ai42_dataset(
                "unused", right, match_count=2, seed=101, workers=2,
                max_steps=2, timeout_minutes=2 / 300, validation_fraction=0.5,
                env_factory=factory,
            )
            self.assertEqual(first["manifest_hash"], second["manifest_hash"])
            self.assertEqual((left / "manifest.json").read_bytes(), (right / "manifest.json").read_bytes())
            dataset = load_dataset(left)
            self.assertEqual(dataset._shards, {})
            summary = audit_ai42_dataset(left)
            self.assertEqual(summary["matches"], 2)
            self.assertEqual(summary["ticks"], 4)
            self.assertTrue(summary["terminal"]["exactly_one_terminal_tick_per_match"])
            self.assertEqual(summary["scenario"], {"ai30_mirror": 2})
            self.assertIn("action", summary["teacher_statuses"])
            self.assertIn("skill_1", summary["skills"])
            self.assertEqual(summary["bytes"]["compression"], "deflate-6")
            self.assertGreater(summary["bytes"]["raw"], summary["bytes"]["stored"])
            manifest_path = left / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shards"][0]["stored_bytes"] += 1
            manifest["manifest_hash"] = hash_payload(
                {key: value for key, value in manifest.items() if key != "manifest_hash"}
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(AI42DatasetError, "stored shard size mismatch"):
                load_dataset(left)

    def test_audit_counts_actions_only_for_action_status_and_maps_wire_reasons(self):
        statuses = np.array([[
            TEACHER_STATUS_ACTION, TEACHER_STATUS_ACTION,
            TEACHER_STATUS_WAIT, TEACHER_STATUS_HOLD,
            TEACHER_STATUS_CANCEL, TEACHER_STATUS_UNAVAILABLE,
            TEACHER_STATUS_NONE, TEACHER_STATUS_WAIT,
            TEACHER_STATUS_HOLD, TEACHER_STATUS_CANCEL,
        ]], dtype=np.uint8)
        actions = np.zeros((1, HERO_COUNT), dtype=ACTION_DTYPE)
        actions[0]["kind"] = np.array([1, 3, 2, 4, 5, 6, 7, 0, 6, 2], dtype=np.uint8)
        reasons = np.array([[
            REJECTION_REASON_NONE, REJECTION_REASON_MASKED,
            REJECTION_REASON_INVALID, REJECTION_REASON_SERVER_REJECTED,
            REJECTION_REASON_SAFETY, REJECTION_REASON_TIMEOUT,
            REJECTION_REASON_POLICY_ERROR, REJECTION_REASON_UNKNOWN,
            REJECTION_REASON_NONE, REJECTION_REASON_MASKED,
        ]], dtype=np.uint8)
        arrays = {
            "teacher_status": statuses,
            "teacher_action": actions,
            "rejection_reason": reasons,
            "elapsed": np.array([0.2], dtype="<f4"),
            "done": np.array([1], dtype=np.uint8),
            "winner": np.array([1], dtype="<i4"),
        }

        class _AuditDataset:
            manifest = {
                "matches": [{
                    "match_id": "audit-match",
                    "split": "train",
                    "tick_count": 1,
                    "scenario": "ai30_mirror",
                    "roster_ids": list(range(HERO_COUNT)),
                    "side_by_slot": [0] * 5 + [1] * 5,
                    "controller_by_slot": [2] * HERO_COUNT,
                }],
                "shards": [{"raw_bytes": 100, "stored_bytes": 50}],
            }
            manifest_hash = "manifest"
            runtime_manifest_hash = "runtime"
            compression = "fixture"

            def arrays_for_match(self, match_id):
                self.assert_match_id = match_id
                return arrays

        with mock.patch("tanat_ai40.audit_ai42_dataset.load_dataset", return_value=_AuditDataset()):
            summary = audit_ai42_dataset("fixture")

        self.assertEqual(summary["teacher_statuses"], {
            "action": 2, "cancel": 2, "hold": 2,
            "none": 1, "unavailable": 1, "wait": 2,
        })
        self.assertEqual(summary["action_kinds"], {"move": 1, "skill_1": 1})
        self.assertEqual(summary["skills"], {"skill_1": 1})
        self.assertEqual(summary["rejections"], {
            "invalid": 1,
            "masked": 2,
            "none": 2,
            "policy_error": 1,
            "safety": 1,
            "server_rejected": 1,
            "timeout": 1,
            "unknown": 1,
        })

    def test_resume_rejects_contract_change_and_reuses_completed_matches(self):
        calls = 0

        def failing_factory(executable, workers, protocol):
            nonlocal calls
            calls += 1
            return _FakeVectorEnv(workers, fail=calls == 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "dataset"
            staging = root / "staging"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                build_ai42_dataset(
                    "unused", destination, match_count=2, workers=1,
                    max_steps=2, timeout_minutes=2 / 300, staging=staging,
                    env_factory=failing_factory,
                )
            self.assertFalse(destination.exists())
            resumed = build_ai42_dataset(
                "unused", destination, match_count=2, workers=1,
                max_steps=2, timeout_minutes=2 / 300, staging=staging,
                env_factory=lambda executable, workers, protocol: _FakeVectorEnv(workers),
            )
            self.assertEqual(resumed["resumed_matches"], 1)
            with self.assertRaisesRegex(AI42DatasetError, "contract mismatch"):
                build_ai42_dataset(
                    "unused", root / "other", match_count=2, workers=2,
                    max_steps=2, timeout_minutes=2 / 300, staging=staging,
                    env_factory=lambda executable, workers, protocol: _FakeVectorEnv(workers),
                )

    def test_partial_staging_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "staging"
            contract = {"match_ids": [], "runtime_manifest": {"test": True}}
            AI42DatasetStaging(root, contract)
            partial = root / "matches" / "match-partial"
            partial.mkdir()
            (partial / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(AI42DatasetError, "partial"):
                AI42DatasetStaging(root, contract)


if __name__ == "__main__":
    unittest.main()
