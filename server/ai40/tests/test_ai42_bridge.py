from __future__ import annotations

import hashlib
import unittest

import numpy as np

from tanat_ai40.bridge_ai42 import (
    AI42BridgeError,
    build_ai42_trajectory,
)
from tanat_ai40.env import (
    ACTION_DTYPE,
    AI42_REWARD_HASH,
    AI42_SCHEMA_HASH,
    HeroAction,
    StepResult,
)
from tanat_ai40.trajectory_ai42 import AI42_TRAJECTORY_SCHEMA_HASH, Outcome, hash_payload


class AI42BridgeTest(unittest.TestCase):
    def test_schema_hash_matches_cross_language_golden(self) -> None:
        self.assertEqual(
            AI42_SCHEMA_HASH.hex(),
            "915e2e4547ccf727567304839f4780c60d48521f3dd1f0dbef7c4a5cc9131274",
        )

    @staticmethod
    def _result(step: int) -> StepResult:
        teacher = np.zeros(10, dtype=ACTION_DTYPE)
        executed = np.zeros(10, dtype=ACTION_DTYPE)
        executed[0] = (1, 0, 40, 0)
        valid = np.ones(10, dtype=np.uint8)
        valid[1] = 0
        reason = np.zeros(10, dtype=np.uint8)
        reason[1] = 1
        return StepResult(
            step, 0.2 * step, False, 0,
            np.zeros(10, dtype=np.uint8), np.zeros(10, dtype="<f4"),
            np.zeros((10, 32), dtype="<f4"),
            np.zeros((10, 96, 16), dtype="<f4"),
            np.zeros((10, 32), dtype="<f4"),
            np.ones((10, 96), dtype=np.uint8),
            np.ones((10, 8), dtype=np.uint8),
            np.ones((10, 96), dtype=np.uint8),
            np.ones((10, 4, 96), dtype=np.uint8),
            np.zeros((10, 4, 40), dtype="<f4"),
            None, None, teacher, np.full(10, 5, dtype=np.uint8),
            executed, valid, reason, AI42_SCHEMA_HASH, AI42_REWARD_HASH,
        )

    def _build(self):
        results = [self._result(1), self._result(2)]
        submitted = [[HeroAction(kind=1, direction=40)] + [HeroAction() for _ in range(9)] for _ in results]
        boundaries = [
            [f"boundary:1:{hero}" for hero in range(10)],
            [f"boundary:2:{hero}" for hero in range(10)],
        ]
        parents = [
            [f"root:{hero}" for hero in range(10)],
            boundaries[0],
        ]
        outcomes = [[Outcome() for _ in range(10)] for _ in results]
        manifest = {"schema_hash": AI42_SCHEMA_HASH.hex(), "reward_hash": AI42_REWARD_HASH.hex()}
        output = build_ai42_trajectory(
            results, submitted, parents, outcomes,
            recurrent_boundaries=boundaries,
            hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="match-1",
            trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
            schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
            manifest=manifest, manifest_hash=hash_payload(manifest),
        )
        return output

    def test_builds_all_slots_and_strict_replay(self):
        output = self._build()
        self.assertEqual(len(output.trajectories), 10)
        self.assertTrue(all(len(item.records) == 2 for item in output.trajectories))
        replay = output.validate_replay(
            observation_payloads=output.observation_payloads,
            mask_payloads=output.mask_payloads,
            action_payloads=output.action_payloads,
            executed_actions=output.executed_actions,
            outcomes=output.outcomes,
            recurrent_parent_ids=output.recurrent_parent_ids,
            recurrent_boundary_ids=output.recurrent_boundary_ids,
        )
        self.assertEqual(len(replay), 10)

    def test_missing_authoritative_reason_fails_closed(self):
        results = [self._result(1)]
        results[0].executed_valid[2] = 0
        results[0].rejection_reason[2] = 0
        with self.assertRaisesRegex(AI42BridgeError, "rejected action has reason none"):
            build_ai42_trajectory(
                results,
                [[HeroAction() for _ in range(10)]],
                [[f"root:{hero}" for hero in range(10)]],
                [[Outcome() for _ in range(10)]],
                recurrent_boundaries=[[f"boundary:1:{hero}" for hero in range(10)]],
                hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="match-1",
                trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
                schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
                manifest={"ok": True}, manifest_hash=hashlib.sha256(b'{"ok":true}').hexdigest(),
            )

    def test_result_hashes_are_mandatory(self):
        for field in ("schema_hash", "reward_hash"):
            with self.subTest(field=field):
                result = self._result(1)
                setattr(result, field, None)
                with self.assertRaisesRegex(AI42BridgeError, rf"{field} is required"):
                    build_ai42_trajectory(
                        [result], [[HeroAction() for _ in range(10)]],
                        [[f"root:{hero}" for hero in range(10)]],
                        [[Outcome() for _ in range(10)]],
                        recurrent_boundaries=[[f"boundary:1:{hero}" for hero in range(10)]],
                        hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="match-1",
                        trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
                        schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
                        manifest={"ok": True}, manifest_hash=hash_payload({"ok": True}),
                    )

    def test_post_boundaries_are_mandatory_and_lineage_is_strict(self):
        args = (
            [self._result(1)], [[HeroAction() for _ in range(10)]],
            [[f"root:{hero}" for hero in range(10)]],
            [[Outcome() for _ in range(10)]],
        )
        kwargs = dict(
            hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="match-1",
            trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
            schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
            manifest={"ok": True}, manifest_hash=hash_payload({"ok": True}),
        )
        with self.assertRaisesRegex(AI42BridgeError, "recurrent_boundaries is required"):
            build_ai42_trajectory(*args, **kwargs)

        results = [self._result(1), self._result(2)]
        boundaries = [[f"boundary:{tick}:{hero}" for hero in range(10)] for tick in (1, 2)]
        parents = [[f"root:{hero}" for hero in range(10)], list(boundaries[0])]
        parents[1][3] = "wrong-lineage"
        with self.assertRaisesRegex(AI42BridgeError, "recurrent parent does not match prior boundary"):
            build_ai42_trajectory(
                results, [[HeroAction() for _ in range(10)] for _ in results], parents,
                [[Outcome() for _ in range(10)] for _ in results],
                recurrent_boundaries=boundaries, **kwargs,
            )

    def test_skill_navigation_offset_round_trips_and_anchor_is_rejected(self):
        result = self._result(1)
        result.teacher_status[0] = 1
        result.teacher_intent[0] = (3, 7, 80, 0)
        result.executed_actions[0] = (3, 7, 80, 0)
        submitted = [HeroAction() for _ in range(10)]
        submitted[0] = HeroAction(kind=3, target=7, direction=80, distance=0)
        output = build_ai42_trajectory(
            [result], [submitted], [[f"root:{hero}" for hero in range(10)]],
            [[Outcome() for _ in range(10)]],
            recurrent_boundaries=[[f"boundary:1:{hero}" for hero in range(10)]],
            hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="match-1",
            trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
            schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
            manifest={"ok": True}, manifest_hash=hash_payload({"ok": True}),
        )
        point = output.trajectories[0].records[0]
        for action in (point.original_ai30_intent, point.projected_neural_action, point.executed_action):
            self.assertEqual((action.kind, action.target, action.skill, action.point, action.anchor),
                             ("skill", 7, 1, (4.0, 4.0), None))
        bad = self._result(2)
        bad.teacher_status[2] = 1
        bad.teacher_intent[2] = (4, 8, 12, 14)
        with self.assertRaisesRegex(AI42BridgeError, "cannot use navigation anchors"):
            build_ai42_trajectory(
                [bad], [[HeroAction() for _ in range(10)]],
                [[f"root:{hero}" for hero in range(10)]],
                [[Outcome() for _ in range(10)]],
                recurrent_boundaries=[[f"boundary:2:{hero}" for hero in range(10)]],
                hero_ids=[f"hero-{hero}" for hero in range(10)], match_id="bad-anchor",
                trajectory_schema_hash=AI42_TRAJECTORY_SCHEMA_HASH,
                schema_hash=AI42_SCHEMA_HASH, reward_hash=AI42_REWARD_HASH,
                manifest={"ok": True}, manifest_hash=hash_payload({"ok": True}),
            )


if __name__ == "__main__":
    unittest.main()
