from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tanat_ai40.bridge_ai42 import REJECTION_REASON_NONE, TEACHER_STATUS_UNAVAILABLE
from tanat_ai40.env import (
    ACTION_DTYPE,
    AI40_ROSTER,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    HERO_COUNT,
    StepResult,
)
from tanat_ai40.smoke_ai42_dataset import DEFAULT_MAX_STEPS, run_dataset_smoke


class _FakeEnv:
    def __init__(self, results: list[StepResult], *, fail_on_step: bool = False):
        self._results = results
        self._fail_on_step = fail_on_step
        self.reset_calls: list[tuple[object, ...]] = []
        self.step_calls: list[list[object]] = []
        self.closed = False

    def __enter__(self) -> "_FakeEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def reset(self, *args: object, **kwargs: object) -> object:
        self.reset_calls.append((args, kwargs))
        # This is intentionally not a StepResult: reset must never be collected.
        return object()

    def step(self, actions: list[object]) -> StepResult:
        self.step_calls.append(list(actions))
        if self._fail_on_step:
            raise RuntimeError("injected step failure")
        return self._results[len(self.step_calls) - 1]


def _result(step: int, *, done: bool, winner: int = 1) -> StepResult:
    teacher = np.zeros(HERO_COUNT, dtype=ACTION_DTYPE)
    executed = np.zeros(HERO_COUNT, dtype=ACTION_DTYPE)
    return StepResult(
        step=step,
        elapsed=float(step),
        done=done,
        winner=winner if done else -1,
        invalid=np.zeros(HERO_COUNT, dtype=np.uint8),
        rewards=np.arange(HERO_COUNT, dtype="<f4") + step,
        hero=np.full((HERO_COUNT, 32), step, dtype="<f4"),
        entities=np.full((HERO_COUNT, 96, 16), step, dtype="<f4"),
        global_state=np.full((HERO_COUNT, 32), step, dtype="<f4"),
        entity_mask=np.ones((HERO_COUNT, 96), dtype=np.uint8),
        kind_mask=np.ones((HERO_COUNT, 8), dtype=np.uint8),
        target_mask=np.ones((HERO_COUNT, 96), dtype=np.uint8),
        skill_target_mask=np.ones((HERO_COUNT, 4, 96), dtype=np.uint8),
        abilities=np.full((HERO_COUNT, 4, 40), step, dtype="<f4"),
        teacher_actions=None,
        teacher_valid=None,
        teacher_intent=teacher,
        teacher_status=np.full(HERO_COUNT, TEACHER_STATUS_UNAVAILABLE, dtype=np.uint8),
        executed_actions=executed,
        executed_valid=np.ones(HERO_COUNT, dtype=np.uint8),
        rejection_reason=np.full(HERO_COUNT, REJECTION_REASON_NONE, dtype=np.uint8),
        schema_hash=AI42_SCHEMA_HASH,
        reward_hash=AI42_REWARD_HASH,
    )


class AI42DatasetSmokeTest(unittest.TestCase):
    def test_fake_capture_excludes_reset_reloads_ten_slots_and_closes(self) -> None:
        fake = _FakeEnv([_result(0, done=False), _result(1, done=True)])
        with tempfile.TemporaryDirectory() as directory:
            summary = run_dataset_smoke(
                "unused-assaultenv",
                directory,
                seed=17,
                max_steps=2,
                env_factory=lambda executable: fake,
            )
            self.assertEqual(summary["ticks"], 2)
            self.assertEqual(len(fake.reset_calls), 1)
            self.assertEqual(len(fake.step_calls), 2)
            args, kwargs = fake.reset_calls[0]
            self.assertEqual(args, (17, 2))
            self.assertEqual(kwargs["roster"], AI40_ROSTER.tolist())
            self.assertEqual(kwargs["controllers"], [2] * HERO_COUNT)
            self.assertTrue(fake.closed)

            manifest = __import__("json").loads((Path(directory) / "manifest.json").read_text())
            entry = manifest["matches"][0]
            self.assertEqual(entry["tick_count"], 2)
            self.assertEqual(len(entry["hero_ids"]), HERO_COUNT)
            self.assertEqual(len(entry["recurrent_parent_ids"][0]), HERO_COUNT)
            self.assertEqual(len(entry["recurrent_boundary_ids"][1]), HERO_COUNT)
            self.assertEqual(entry["recurrent_parent_ids"][0][0].split(":")[-2:], ["root", "00"])
            self.assertEqual(
                entry["recurrent_parent_ids"][1],
                entry["recurrent_boundary_ids"][0],
            )

            from tanat_ai40.dataset_ai42 import load_dataset

            loaded = load_dataset(directory, expected_runtime_manifest=manifest["runtime_manifest"])
            arrays = loaded.arrays_for_match(summary["match_id"])
            self.assertEqual(arrays["rewards"].shape, (2, HERO_COUNT))
            self.assertEqual(arrays["projected_action"].shape, (2, HERO_COUNT))
            self.assertEqual(arrays["done"].tolist(), [0, 1])
            self.assertEqual(arrays["winner"].tolist(), [-1, 1])
            self.assertGreater(DEFAULT_MAX_STEPS, 0)

        failing = _FakeEnv([], fail_on_step=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected step failure"):
                run_dataset_smoke("unused", directory, env_factory=lambda executable: failing)
        self.assertTrue(failing.closed)


if __name__ == "__main__":
    unittest.main()
