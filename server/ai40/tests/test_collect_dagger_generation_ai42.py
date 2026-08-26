from __future__ import annotations

import unittest

from tanat_ai40.collect_dagger_generation_ai42 import build_dagger_schedule
from tanat_ai40.env import CONTROLLER_AI30, CONTROLLER_AI40


class AI42DAggerGenerationTests(unittest.TestCase):
    def test_schedule_is_deterministic_side_balanced_and_single_lineage(self) -> None:
        kwargs = dict(
            seed=91, matches=4, max_steps=600,
            lineage={"checkpoint_sha256": "b" * 64}, threshold=0.08,
            min_gap_ticks=5, split_seed=7, validation_fraction=0.25,
        )
        first, first_specs = build_dagger_schedule(**kwargs)
        second, second_specs = build_dagger_schedule(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_specs, second_specs)
        self.assertEqual(len({spec.seed for spec in first_specs}), 4)
        self.assertEqual(len({spec.roster_ids for spec in first_specs}), 4)
        for index, spec in enumerate(first_specs):
            candidate = CONTROLLER_AI40 if index % 2 == 0 else CONTROLLER_AI30
            opponent = CONTROLLER_AI30 if index % 2 == 0 else CONTROLLER_AI40
            self.assertEqual(spec.controller_by_slot[:5], (candidate,) * 5)
            self.assertEqual(spec.controller_by_slot[5:], (opponent,) * 5)
        self.assertEqual(first["policy_lineage"], kwargs["lineage"])

    def test_schedule_validation_fails_before_launching_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "matches"):
            build_dagger_schedule(
                seed=1, matches=0, max_steps=1, lineage={}, threshold=0.1,
                min_gap_ticks=1, split_seed=1, validation_fraction=0,
            )
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            build_dagger_schedule(
                seed=1, matches=1, max_steps=1, lineage={}, threshold=0.1,
                min_gap_ticks=1, split_seed=1, validation_fraction=2,
            )
        with self.assertRaisesRegex(ValueError, "matches"):
            build_dagger_schedule(
                seed=1, matches=1.5, max_steps=1, lineage={}, threshold=0.1,
                min_gap_ticks=1, split_seed=1, validation_fraction=0,
            )
        with self.assertRaisesRegex(ValueError, "max_steps"):
            build_dagger_schedule(
                seed=1, matches=1, max_steps=1.5, lineage={}, threshold=0.1,
                min_gap_ticks=1, split_seed=1, validation_fraction=0,
            )


if __name__ == "__main__":
    unittest.main()
