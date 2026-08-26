from __future__ import annotations

import unittest

import numpy as np

from tanat_ai40.collect_dagger_ai42 import MarginInterventionGate, _build_schedule
from tanat_ai40.env import CONTROLLER_AI30, CONTROLLER_AI40


class AI42DAggerCollectorTests(unittest.TestCase):
    def test_margin_gate_is_per_actor_deterministic_and_death_aware(self) -> None:
        gate = MarginInterventionGate(3, threshold=0.1, min_gap_ticks=3)
        selected = gate.select(
            np.asarray([0.05, 0.2, 0.01]), np.asarray([True, True, False]), 0,
        )
        np.testing.assert_array_equal(selected, [True, False, False])
        np.testing.assert_array_equal(
            gate.select(np.asarray([0.01, 0.01, 0.01]), np.ones(3, bool), 1),
            [False, True, True],
        )
        np.testing.assert_array_equal(
            gate.select(np.zeros(3), np.ones(3, bool), 3), [True, False, False],
        )

    def test_schedule_freezes_policy_and_intervention_provenance(self) -> None:
        schedule = _build_schedule(
            seed=9, max_steps=100, candidate_side=2, roster=tuple(range(10)),
            lineage={"checkpoint_sha256": "a" * 64}, threshold=0.08,
            min_gap_ticks=5,
        )
        controllers = schedule["match_schedule"][0]["controller_by_slot"]
        self.assertEqual(controllers[:5], [CONTROLLER_AI30] * 5)
        self.assertEqual(controllers[5:], [CONTROLLER_AI40] * 5)
        self.assertEqual(schedule["intervention_policy"]["threshold"], 0.08)
        self.assertEqual(schedule["policy_lineage"]["checkpoint_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
