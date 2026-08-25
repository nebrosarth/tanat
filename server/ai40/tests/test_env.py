from __future__ import annotations

import os
import argparse
import hashlib
import io
import struct
from pathlib import Path
import unittest

import numpy as np

from tanat_ai40.env import (
    ACTION_DTYPE, AI40_ROSTER, AI40_SELF_PLAY_CONTROLLERS, AI41_EVALUATION_PROTOCOL_VERSION,
    AI41_NAVIGATION_PROTOCOL_VERSION, AI41_PROTOCOL_VERSION,
    AI41_STRATEGIC_PROTOCOL_VERSION, AI41_TEACHER_PROTOCOL_VERSION,
    AI41_STRATEGIC_REWARD_HASH, AI42_EVALUATION_PROTOCOL_VERSION,
    AI42_EVALUATION_SCHEMA, AI42_EVALUATION_SCHEMA_HASH,
    AI42_PROTOCOL_VERSION, AI42_REWARD_HASH, AI42_SCHEMA, AI42_SCHEMA_HASH,
    CONTROLLED_ACTION_DTYPE, COMMAND_VECTOR_STEP,
    AssaultEnvProcess, AssaultProtocolError, AssaultVectorEnv, AssaultVectorProcess,
    HeroAction, HERO_COUNT,
    MAGIC, RESPONSE_RESULT, SCHEMA_HASH, STDERR_TAIL_BYTES, _result_layout,
    CONTROLLER_AI30, self_play_rosters,
)
from tanat_ai40.train_matches import (
    build_schedule, historical_actor_indices, policy_actor_mask,
)
from tanat_ai40.train_async import partition_workers
from tanat_ai40.train import horizon_discounts, stack_observations
from tanat_ai40.evaluate_ai30 import controllers_for_side, wilson_interval
from tanat_ai40.train_campaign import evaluation_plan


class AssaultEnvProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("TANAT_ASSAULTENV")
        if not configured:
            raise unittest.SkipTest("set TANAT_ASSAULTENV to the built Go runner")
        cls.executable = Path(configured)

    def run_trajectory(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=99, max_steps=50)
            snapshots = []
            for _ in range(45):
                result = env.step([HeroAction() for _ in range(HERO_COUNT)])
                snapshots.append((result.hero.copy(), result.entities.copy(), result.entity_mask.copy()))
            return result, snapshots

    def test_schema_shapes_and_determinism(self):
        first, first_trajectory = self.run_trajectory()
        second, second_trajectory = self.run_trajectory()
        self.assertEqual(first.step, 45)
        self.assertEqual(first.hero.shape, (10, 32))
        self.assertEqual(first.entities.shape, (10, 96, 16))
        self.assertEqual(first.skill_target_mask.shape, (10, 4, 96))
        self.assertGreater(int(first.entity_mask.sum()), 0)
        for a, b in zip(first_trajectory, second_trajectory):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_ai30_teacher_controller(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=101, max_steps=10,
                               controllers=[CONTROLLER_AI30] * HERO_COUNT)
            self.assertEqual(result.step, 0)
            result = env.step([HeroAction() for _ in range(HERO_COUNT)])
            self.assertEqual(result.step, 1)

    def test_ai41_protocol_exposes_ability_records(self):
        with AssaultEnvProcess(self.executable, AI41_PROTOCOL_VERSION) as env:
            result = env.reset(seed=109, max_steps=10)
            self.assertEqual(result.abilities.shape, (10, 4, 40))
            np.testing.assert_allclose(
                result.abilities[:, :, 1], np.repeat(result.hero[:, 0, None], 4, axis=1),
            )
            self.assertTrue(np.all(result.abilities[:, :, 3] > 0))
            self.assertGreater(int(np.count_nonzero(result.abilities[:, :, 10:39])), 0)
            self.assertTrue(np.all(result.global_state[:, 8] == 1))
            np.testing.assert_array_equal(result.global_state[:, 9:12].sum(axis=1), 1)
            self.assertTrue(np.all(result.global_state[:, 12] == 0))
            self.assertTrue(np.all(result.global_state[:, 13] == 0))
            for side in (0, 1):
                np.testing.assert_array_equal(
                    result.global_state[side * 5:(side + 1) * 5, 9:12].sum(axis=0),
                    [2, 1, 2],
                )

    def test_ai41_evaluation_disables_lane_randomization(self):
        with AssaultEnvProcess(self.executable, AI41_EVALUATION_PROTOCOL_VERSION) as env:
            result = env.reset(seed=109, max_steps=10)
            self.assertEqual(result.abilities.shape, (10, 4, 40))
            self.assertTrue(np.all(result.global_state[:, 8:14] == 0))

    def test_ai41_navigation_protocol_accepts_grid_and_anchor_actions(self):
        actions = np.zeros((1, HERO_COUNT, 4), dtype=np.int64)
        actions[0, :, 0] = 1  # Move
        actions[0, :, 2] = 80  # local grid: +12,+12
        actions[0, :, 3] = 0  # local mode
        with AssaultVectorEnv(
            self.executable, 1, AI41_NAVIGATION_PROTOCOL_VERSION,
        ) as env:
            first = env.reset([311], max_steps=20)[0]
            first_positions = first.hero[:, 2:4].copy()
            result = env.step(actions)[0]
            self.assertEqual(int(result.invalid.sum()), 0)
            self.assertGreater(
                float(np.abs(result.hero[:, 2:4] - first_positions).sum()), 0,
            )
            actions[0, :, 3] = 6  # north lane, own-relative 80% anchor
            result = env.step(actions)[0]
            self.assertEqual(int(result.invalid.sum()), 0)

    def test_ai41_strategic_protocol_uses_calibrated_reward_contract(self):
        with AssaultEnvProcess(self.executable, AI41_STRATEGIC_PROTOCOL_VERSION) as env:
            result = env.reset(seed=313, max_steps=10)
            self.assertEqual(result.step, 0)

    def test_ai41_teacher_protocol_exports_ai30_actions(self):
        with AssaultEnvProcess(self.executable, AI41_TEACHER_PROTOCOL_VERSION) as env:
            env.reset(seed=317, max_steps=10, controllers=[CONTROLLER_AI30] * HERO_COUNT)
            result = env.step([HeroAction() for _ in range(HERO_COUNT)])
            self.assertIsNotNone(result.teacher_actions)
            self.assertIsNotNone(result.teacher_valid)
            self.assertEqual(result.teacher_actions.shape, (HERO_COUNT,))
            self.assertGreater(int(result.teacher_valid.sum()), 0)

    def test_ai40_mirror_self_play_controller(self):
        with AssaultEnvProcess(self.executable) as env:
            result = env.reset(seed=103, max_steps=10,
                               controllers=AI40_SELF_PLAY_CONTROLLERS)
            self.assertEqual(result.step, 0)
            result = env.step([HeroAction() for _ in range(HERO_COUNT)])
            self.assertEqual(result.step, 1)
            self.assertEqual(int(result.invalid.sum()), 0)

    def test_vector_process_batches_steps_and_index_resets(self):
        actions = np.zeros((3, HERO_COUNT, 4), dtype=np.int64)
        with AssaultVectorEnv(self.executable, 3) as env:
            results = env.reset(
                [201, 202, 203], max_steps=20,
                controller_sets=[AI40_SELF_PLAY_CONTROLLERS] * 3,
                rosters=self_play_rosters(np.random.default_rng(9), 3),
            )
            self.assertEqual([result.step for result in results], [0, 0, 0])
            results = env.step(actions)
            self.assertEqual([result.step for result in results], [1, 1, 1])
            untouched = results[0].hero.copy()
            replacements = env.reset_indices(
                [1], [301], max_steps=20,
                controller_sets=[AI40_SELF_PLAY_CONTROLLERS],
                rosters=self_play_rosters(np.random.default_rng(10), 1),
            )
            self.assertEqual(replacements[1].step, 0)
            np.testing.assert_array_equal(results[0].hero, untouched)
            results = env.step(actions)
            self.assertEqual([result.step for result in results], [2, 1, 2])

    def test_teacher_labels_survive_partial_vector_reset_fallback(self):
        """The trainer stacks this path after one match finishes before peers."""
        actions = np.zeros((2, HERO_COUNT, 4), dtype=np.int64)
        with AssaultVectorEnv(self.executable, 2, AI41_TEACHER_PROTOCOL_VERSION) as env:
            results = env.reset(
                [411, 412], max_steps=20,
                controller_sets=[[CONTROLLER_AI30] * HERO_COUNT] * 2,
                rosters=self_play_rosters(np.random.default_rng(11), 2),
            )
            results = env.step(actions)
            replacements = env.reset_indices(
                [1], [413], max_steps=20,
                controller_sets=[[CONTROLLER_AI30] * HERO_COUNT],
                rosters=self_play_rosters(np.random.default_rng(12), 1),
            )
            results[1] = replacements[1]
            batch = stack_observations(results)
            self.assertEqual(batch.teacher_actions.shape, (2 * HERO_COUNT,))
            self.assertEqual(batch.teacher_valid.shape, (2 * HERO_COUNT,))
            self.assertGreater(int(batch.teacher_valid.sum()), 0)


class ProtocolContractTest(unittest.TestCase):
    @staticmethod
    def _header(body: bytearray, version: int, response: int, count: int | None = None) -> None:
        body[:4] = MAGIC
        struct.pack_into("<HH", body, 4, version, response)
        body[8:40] = b"S" * 32
        body[40:72] = b"R" * 32
        if count is not None:
            struct.pack_into("<I", body, 72, count)

    @staticmethod
    def _scalar_env(frame: bytes, version: int = AI41_TEACHER_PROTOCOL_VERSION) -> AssaultEnvProcess:
        env = object.__new__(AssaultEnvProcess)
        env.protocol_version = version
        env.schema_hash = frame[12:44]
        env.reward_hash = frame[44:76]
        env.observation_dtype = np.dtype([
            ("hero", "<f4", (32,)),
            ("abilities", "<f4", (4, 40)),
            ("entities", "<f4", (96, 16)),
            ("global_state", "<f4", (32,)),
            ("entity_mask", "u1", (96,)),
            ("kind_mask", "u1", (8,)),
            ("target_mask", "u1", (96,)),
            ("skill_target_mask", "u1", (4, 96)),
        ], align=False)
        env._out = io.BytesIO(frame)
        env._stderr_text = lambda: ""
        return env

    @staticmethod
    def _vector_env(frame: bytes, version: int = AI41_TEACHER_PROTOCOL_VERSION) -> AssaultVectorProcess:
        env = object.__new__(AssaultVectorProcess)
        env.protocol_version = version
        env.schema_hash = frame[12:44]
        env.reward_hash = frame[44:76]
        env._out = io.BytesIO(frame)
        env._result_buffer = bytearray()
        env._partial_result_buffer = bytearray()
        return env

    def test_strategic_teacher_parse_is_symmetric_and_sizes_are_exact(self):
        layout = _result_layout(AI41_TEACHER_PROTOCOL_VERSION)
        body = bytearray(layout.size)
        self._header(body, AI41_TEACHER_PROTOCOL_VERSION, RESPONSE_RESULT)
        action_offset = next(offset for name, offset, _ in layout.fields if name == "result.teacher_actions")
        valid_offset = next(offset for name, offset, _ in layout.fields if name == "result.teacher_valid")
        body[action_offset:action_offset + ACTION_DTYPE.itemsize] = struct.pack("<B H B B", 2, 0x1234, 9, 8)
        body[valid_offset] = 1
        frame = struct.pack("<I", len(body)) + body

        result = self._scalar_env(frame)._read_result()
        self.assertEqual(int(result.teacher_actions[0]["kind"]), 2)
        self.assertEqual(int(result.teacher_actions[0]["target"]), 0x1234)
        self.assertEqual(int(result.teacher_actions[0]["direction"]), 9)
        self.assertEqual(int(result.teacher_actions[0]["distance"]), 8)
        self.assertEqual(int(result.teacher_valid[0]), 1)
        self.assertEqual(layout.size, 76438)
        self.assertEqual(_result_layout(AI42_PROTOCOL_VERSION).size, 76508)
        fields = {name: (offset, size) for name, offset, size in layout.fields}
        self.assertEqual(fields["result.observation[0].hero"], (138, 128))
        self.assertEqual(fields["result.observation[0].abilities"], (266, 640))
        self.assertEqual(fields["result.teacher_actions"], (76378, 50))
        self.assertEqual(fields["result.teacher_valid"], (76428, 10))
        self.assertEqual(_result_layout(AI41_TEACHER_PROTOCOL_VERSION, 2, vector=True).size, 152802)
        self.assertEqual(_result_layout(AI42_PROTOCOL_VERSION, 2, vector=True).size, 152942)

    def test_ai42_hashes_are_derived_and_runtime_opt_in_is_allowed(self):
        self.assertEqual(AI42_SCHEMA_HASH, hashlib.sha256(AI42_SCHEMA.encode()).digest())
        self.assertEqual(AI42_REWARD_HASH, AI41_STRATEGIC_REWARD_HASH)
        self.assertEqual(_result_layout(AI42_PROTOCOL_VERSION).size, 76508)

    def test_ai42_evaluation_hash_and_controlled_action_wire_are_exact(self):
        self.assertEqual(
            AI42_EVALUATION_SCHEMA_HASH,
            hashlib.sha256(AI42_EVALUATION_SCHEMA.encode()).digest(),
        )
        self.assertEqual(
            AI42_EVALUATION_SCHEMA_HASH.hex(),
            "f55ea7220d095f52fbd218e4e8665977b30824de13b59da139252eabc5bc212c",
        )
        self.assertEqual(
            _result_layout(AI42_EVALUATION_PROTOCOL_VERSION).size, 76378,
        )
        process = object.__new__(AssaultVectorProcess)
        process.protocol_version = AI42_EVALUATION_PROTOCOL_VERSION
        process.workers = 1
        captured = {}
        process._write = lambda command, payload: captured.update(
            command=command, payload=bytes(payload),
        )
        process._read_vector_results = lambda count: []
        values = np.zeros((1, HERO_COUNT, 5), dtype=np.int16)
        values[0, 0] = [2, 1, 0x1234, 80, 14]
        process.step(values)
        self.assertEqual(captured["command"], COMMAND_VECTOR_STEP)
        self.assertEqual(struct.unpack_from("<I", captured["payload"])[0], 1)
        packed = np.frombuffer(
            captured["payload"], dtype=CONTROLLED_ACTION_DTYPE,
            count=HERO_COUNT, offset=4,
        )
        self.assertEqual(tuple(packed[0]), (2, 1, 0x1234, 80, 14))

    def test_ai42_scalar_parser_reads_append_fields_exactly(self):
        layout = _result_layout(AI42_PROTOCOL_VERSION)
        body = bytearray(layout.size)
        self._header(body, AI42_PROTOCOL_VERSION, RESPONSE_RESULT)
        offsets = {name: offset for name, offset, _ in layout.fields}
        body[offsets["result.teacher_intent"]:offsets["result.teacher_intent"] + ACTION_DTYPE.itemsize] = struct.pack(
            "<B H B B", 1, 0, 80, 14,
        )
        body[offsets["result.teacher_status"]] = 1
        body[offsets["result.executed_actions"]:offsets["result.executed_actions"] + ACTION_DTYPE.itemsize] = struct.pack(
            "<B H B B", 2, 0x1234, 0, 0,
        )
        body[offsets["result.executed_valid"]] = 1
        body[offsets["result.rejection_reason"] + 1] = 1
        frame = struct.pack("<I", len(body)) + body
        result = self._scalar_env(frame, AI42_PROTOCOL_VERSION)._read_result()
        self.assertEqual(tuple(result.teacher_intent[0]), (1, 0, 80, 14))
        self.assertEqual(int(result.teacher_status[0]), 1)
        self.assertEqual(tuple(result.executed_actions[0]), (2, 0x1234, 0, 0))
        self.assertEqual(int(result.executed_valid[0]), 1)
        self.assertEqual(int(result.rejection_reason[1]), 1)

    def test_scalar_and_vector_size_errors_identify_field_and_offset(self):
        scalar_layout = _result_layout(AI41_TEACHER_PROTOCOL_VERSION)
        for size, expected in (
            (scalar_layout.size - 1, "field=result.teacher_valid"),
            (scalar_layout.size + 1, "field=result.trailing"),
        ):
            body = bytearray(size)
            self._header(body, AI41_TEACHER_PROTOCOL_VERSION, RESPONSE_RESULT)
            frame = struct.pack("<I", len(body)) + body
            with self.assertRaisesRegex(AssaultProtocolError, expected):
                self._scalar_env(frame)._read_result()

        vector_layout = _result_layout(AI41_TEACHER_PROTOCOL_VERSION, 2, vector=True)
        for size, expected in (
            (vector_layout.size - 1, "field=result.teacher_valid"),
            (vector_layout.size + 1, "field=result.trailing"),
        ):
            body = bytearray(size)
            self._header(body, AI41_TEACHER_PROTOCOL_VERSION, 101, count=2)
            frame = struct.pack("<I", len(body)) + body
            with self.assertRaisesRegex(AssaultProtocolError, expected):
                self._vector_env(frame)._read_vector_results(2)

    def test_truncated_stream_reports_frame_body_offset(self):
        layout = _result_layout(AI41_TEACHER_PROTOCOL_VERSION)
        body = bytearray(16)
        body[:4] = MAGIC
        struct.pack_into("<HH", body, 4, AI41_TEACHER_PROTOCOL_VERSION, RESPONSE_RESULT)
        frame = struct.pack("<I", layout.size) + body
        with self.assertRaisesRegex(AssaultProtocolError, r"field=frame\.body offset=16"):
            self._scalar_env(frame)._read_result()

    def test_vector_process_has_bounded_stderr_drain(self):
        self.assertTrue(hasattr(AssaultVectorProcess, "_drain_stderr"))
        self.assertTrue(hasattr(AssaultVectorProcess, "_stderr_text"))
        self.assertEqual(STDERR_TAIL_BYTES, 65_536)


class SelfPlayRosterTest(unittest.TestCase):
    def test_rosters_are_reproducible_full_pool_permutations(self):
        first = self_play_rosters(np.random.default_rng(77), 4)
        second = self_play_rosters(np.random.default_rng(77), 4)
        self.assertEqual(first, second)
        expected = sorted(AI40_ROSTER.tolist())
        for roster in first:
            self.assertEqual(sorted(roster), expected)
        self.assertGreater(len({tuple(roster[:5]) for roster in first}), 1)

    def test_negative_roster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self_play_rosters(np.random.default_rng(1), -1)


class MixedTrainingScheduleTest(unittest.TestCase):
    def test_balanced_schedule_and_teacher_side_swap(self):
        schedule = build_schedule(3, 4)
        self.assertEqual(sum(match.opponent == "ai40" for match in schedule), 3)
        teachers = [match for match in schedule if match.opponent == "ai30"]
        self.assertEqual([match.ai40_side for match in teachers], [1, 2, 1, 2])

    def test_resumed_teacher_schedule_preserves_side_parity(self):
        teachers = build_schedule(0, 3, teacher_start_index=5)
        self.assertEqual([match.ai40_side for match in teachers], [2, 1, 2])

    def test_scripted_teacher_slots_are_excluded_from_policy_mask(self):
        teacher = next(match for match in build_schedule(0, 1))
        mask = policy_actor_mask([teacher])
        self.assertEqual(int(mask.sum()), HERO_COUNT // 2)
        expected = np.asarray(
            [controller != CONTROLLER_AI30 for controller in teacher.controllers],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_empty_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule(0, 0)

    def test_historical_side_is_frozen_and_excluded_from_ppo(self):
        histories = [
            match for match in build_schedule(0, 0, historical_matches=2,
                                              historical_ids=("stage-005",))
        ]
        self.assertEqual([match.ai40_side for match in histories], [1, 2])
        mask = policy_actor_mask(histories)
        self.assertEqual(int(mask.sum()), HERO_COUNT)
        grouped = historical_actor_indices(histories)
        np.testing.assert_array_equal(
            grouped["stage-005"], np.asarray(list(range(5, 10)) + list(range(10, 15))),
        )

    def test_historical_schedule_requires_a_pool(self):
        with self.assertRaises(ValueError):
            build_schedule(0, 0, historical_matches=1)

    def test_long_horizon_discounts_have_requested_trace_decay(self):
        gamma, lam = horizon_discounts(0.2, 1_200, 180)
        self.assertAlmostEqual(gamma ** (1_200 / 0.2), np.exp(-1), places=6)
        self.assertAlmostEqual((gamma * lam) ** (180 / 0.2), np.exp(-1), places=6)


class AsyncRolloutGroupTest(unittest.TestCase):
    def test_partition_workers_keeps_remainder_group(self):
        self.assertEqual(partition_workers(50, 8), [8, 8, 8, 8, 8, 8, 2])

    def test_partition_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            partition_workers(0, 4)
        with self.assertRaises(ValueError):
            partition_workers(4, 0)


class AI30EvaluationTest(unittest.TestCase):
    def test_controller_sides_are_balanced(self):
        first = controllers_for_side(1)
        second = controllers_for_side(2)
        self.assertEqual(first[:5], (3,) * 5)
        self.assertEqual(first[5:], (2,) * 5)
        self.assertEqual(second[:5], (2,) * 5)
        self.assertEqual(second[5:], (3,) * 5)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(120, 200)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)
        self.assertGreater(low, 0.5)

    def test_empty_wilson_interval_is_uninformative(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_adaptive_evaluation_plan(self):
        args = argparse.Namespace(
            eval_matches=50, eval_medium_matches=200, eval_final_matches=500,
            eval_medium_win_rate=0.4, eval_final_win_rate=0.55,
        )
        self.assertEqual(evaluation_plan(args), (
            ("quick", 50, 0.4),
            ("medium", 200, 0.55),
            ("confirmation", 500, None),
        ))


if __name__ == "__main__":
    unittest.main()
