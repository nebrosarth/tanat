from __future__ import annotations

import json
import unittest

from tanat_ai40.trajectory_ai42 import (
    Action,
    ActionHashSource,
    ActionKind,
    MatchValidationError,
    Outcome,
    ReplayStep,
    ReplayValidationError,
    TeacherMatch,
    TeacherTrajectory,
    TeacherTrajectoryRecord,
    TrajectoryDecodeError,
    TrajectoryError,
    audit_dataset,
    canonical_json_bytes,
    generate_group_stratified_sample_indices,
    generate_stratified_sample_indices,
    hash_payload,
    validate_integrity,
    validate_match_trajectories,
    validate_replay,
)


def observation(hero_id: str, tick: int):
    return {"hero": hero_id, "tick": tick, "visible": [tick, tick + 1]}


def mask(tick: int):
    return {"kind": [1, 1, tick % 2], "tick": tick}


def action_for(kind: str, tick: int, lineage_id: str | None = None) -> Action:
    if kind == ActionKind.WAIT.value:
        return Action(kind)
    if kind == ActionKind.HOLD.value:
        return Action(kind, timing="next", lineage_id=lineage_id)
    if kind == ActionKind.CANCEL.value:
        return Action(kind, lineage_id=lineage_id)
    if kind == "attack":
        return Action(kind, target=f"enemy-{tick}", timing="now")
    if kind == "skill":
        return Action(kind, target=f"enemy-{tick}", timing="now", skill=1)
    return Action("move", point=(float(tick), 1.0), anchor=2, timing="now")


def make_record(
    tick: int,
    *,
    hero_id: str = "hero-7",
    sequence: int | None = None,
    kind: str = "move",
    lineage_id: str | None = None,
    valid: bool = True,
    original: Action | None = None,
    projected: Action | None = None,
    executed: Action | None = None,
    outcome_value: Outcome | None = None,
    external_action=None,
) -> TeacherTrajectoryRecord:
    selected = executed or action_for(kind, tick, lineage_id)
    kwargs = {}
    if external_action is not None:
        kwargs["action"] = external_action
    return TeacherTrajectoryRecord.from_payload(
        tick=tick,
        sequence=tick if sequence is None else sequence,
        hero_id=hero_id,
        recurrent_parent_id=(
            f"{hero_id}-initial" if tick == 0 else f"{hero_id}-boundary-{tick - 1}"
        ),
        recurrent_boundary_id=f"{hero_id}-boundary-{tick}",
        observation=observation(hero_id, tick),
        mask=mask(tick),
        original_ai30_intent=original or selected,
        projected_neural_action=projected or selected,
        executed_action=selected,
        valid=valid,
        rejection_reason="masked" if not valid else "none",
        outcome=outcome_value or Outcome(
            reward=float(tick + 1),
            team_reward=0.5,
            damage=float(tick * 2),
            kills=int(tick == 1),
            deaths=int(tick == 2),
            terminal=tick == 2,
            winner="team-1" if tick == 2 else None,
        ),
        **kwargs,
    )


def make_trajectory(
    *,
    hero_id: str = "hero-7",
    match_id: str = "match-1",
    ticks=(0, 1, 2),
    records=None,
    trajectory_id: str | None = None,
) -> TeacherTrajectory:
    rows = tuple(records if records is not None else (make_record(tick, hero_id=hero_id) for tick in ticks))
    return TeacherTrajectory(
        trajectory_id=trajectory_id or f"{match_id}-{hero_id}",
        match_id=match_id,
        hero_id=hero_id,
        records=rows,
        expected_ticks=tuple(ticks),
    )


def strict_replay_inputs(trajectory_or_records):
    records = (
        trajectory_or_records.records
        if isinstance(trajectory_or_records, TeacherTrajectory)
        else tuple(trajectory_or_records)
    )
    return {
        "observation_payloads": [
            observation(record.hero_id, record.tick) for record in records
        ],
        "mask_payloads": [mask(record.tick) for record in records],
        "executed_actions": [record.executed_action for record in records],
        "outcomes": [record.outcome for record in records],
        "recurrent_parent_ids": [record.recurrent_parent_id for record in records],
        "recurrent_boundary_ids": [record.recurrent_boundary_id for record in records],
    }


class AI42TrajectoryTest(unittest.TestCase):
    def test_canonical_round_trip_strict_schema_corruption_and_surrogate(self):
        trajectory = make_trajectory()
        encoded = trajectory.to_bytes()
        self.assertEqual(encoded, trajectory.to_bytes())
        self.assertEqual(TeacherTrajectory.from_bytes(encoded), trajectory)

        parsed = json.loads(encoded)
        del parsed["records"][0]["mask_hash"]
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(canonical_json_bytes(parsed))

        parsed = json.loads(encoded)
        parsed["unexpected"] = True
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(canonical_json_bytes(parsed))

        parsed = json.loads(encoded)
        parsed["schema_version"] = 999
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(canonical_json_bytes(parsed))

        parsed = json.loads(encoded)
        parsed["records"][1]["outcome"]["reward"] = 999.0
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(canonical_json_bytes(parsed))

        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(encoded + b"trailing")
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(b" " + encoded)

        escaped_surrogate = encoded.decode("utf-8").replace("match-1-hero-7", "\\ud800", 1)
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(escaped_surrogate)

        nested_surrogate = encoded.decode("utf-8").replace(
            '"event":null', '"event":"\\ud800"', 1,
        )
        with self.assertRaises(TrajectoryDecodeError):
            TeacherTrajectory.from_bytes(nested_surrogate)

    def test_construction_and_replay_reject_gaps_duplicates_and_bad_sequences(self):
        with self.assertRaises(TrajectoryError):
            make_trajectory(ticks=(0, 1, 2), records=(make_record(0), make_record(2)))

        duplicate = make_record(1)
        object.__setattr__(duplicate, "tick", 0)
        with self.assertRaises(TrajectoryError):
            make_trajectory(ticks=(0, 1), records=(make_record(0), duplicate))

        with self.assertRaises(TrajectoryError):
            TeacherTrajectory(
                trajectory_id="gap", match_id="match-1", hero_id="hero-7",
                records=(make_record(0), make_record(1)), expected_ticks=(0, 2),
            )

        bad_sequence = make_record(1)
        object.__setattr__(bad_sequence, "sequence", 3)
        bad_rows = (make_record(0), bad_sequence)
        with self.assertRaises(ReplayValidationError):
            validate_replay(
                bad_rows, expected_ticks=(0, 1), **strict_replay_inputs(bad_rows),
            )

        gap_rows = (make_record(0), make_record(1))
        with self.assertRaises(ReplayValidationError):
            validate_replay(
                gap_rows, expected_ticks=(0, 2), **strict_replay_inputs(gap_rows),
            )

    def test_replay_requires_payloads_and_detects_action_outcome_and_lineage_tampering(self):
        trajectory = make_trajectory()
        expected = strict_replay_inputs(trajectory)
        self.assertTrue(validate_integrity(trajectory))
        self.assertTrue(validate_replay(trajectory, **expected))
        with self.assertRaises(ReplayValidationError):
            validate_replay(trajectory)
        for required in tuple(expected):
            incomplete = dict(expected)
            del incomplete[required]
            with self.subTest(missing=required), self.assertRaises(ReplayValidationError):
                validate_replay(trajectory, **incomplete)

        wrong_observation = dict(expected)
        wrong_observation["observation_payloads"] = [
            {"wrong": True}, *expected["observation_payloads"][1:],
        ]
        with self.assertRaises(ReplayValidationError):
            validate_replay(trajectory, **wrong_observation)

        action_tampered = make_trajectory()
        authoritative_action = strict_replay_inputs(action_tampered)
        changed_action = Action("attack", target="tampered", timing="now")
        object.__setattr__(
            action_tampered.records[1], "executed_action", changed_action,
        )
        object.__setattr__(
            action_tampered.records[1], "action_hash",
            hash_payload(changed_action.to_dict()),
        )
        object.__setattr__(
            action_tampered.records[1], "integrity_hash",
            action_tampered.records[1].compute_integrity_hash(),
        )
        self.assertTrue(validate_integrity(action_tampered))
        with self.assertRaises(ReplayValidationError):
            validate_replay(action_tampered, **authoritative_action)

        outcome_tampered = make_trajectory()
        authoritative_outcome = strict_replay_inputs(outcome_tampered)
        object.__setattr__(outcome_tampered.records[1], "outcome", Outcome(reward=999.0))
        object.__setattr__(
            outcome_tampered.records[1], "integrity_hash",
            outcome_tampered.records[1].compute_integrity_hash(),
        )
        self.assertTrue(validate_integrity(outcome_tampered))
        with self.assertRaises(ReplayValidationError):
            validate_replay(outcome_tampered, **authoritative_outcome)

        lineage_tampered = make_trajectory()
        authoritative_lineage = strict_replay_inputs(lineage_tampered)
        new_boundary = "coherently-rehashed-boundary"
        object.__setattr__(lineage_tampered.records[0], "recurrent_boundary_id", new_boundary)
        object.__setattr__(lineage_tampered.records[1], "recurrent_parent_id", new_boundary)
        for record in lineage_tampered.records[:2]:
            object.__setattr__(record, "integrity_hash", record.compute_integrity_hash())
        self.assertTrue(validate_integrity(lineage_tampered))
        with self.assertRaises(ReplayValidationError):
            validate_replay(lineage_tampered, **authoritative_lineage)

        callback = lambda record: ReplayStep(
            observation("hero-7", record.tick),
            mask(record.tick),
            executed_action=record.executed_action,
            outcome=record.outcome,
            recurrent_parent_id=record.recurrent_parent_id,
            recurrent_boundary_id=record.recurrent_boundary_id,
        )
        self.assertTrue(validate_replay(trajectory, replay_step=callback))
        with self.assertRaises(ReplayValidationError):
            validate_replay(
                trajectory,
                replay_step=lambda record: ReplayStep(
                    observation("hero-7", record.tick), mask(record.tick),
                ),
            )

    def test_external_action_hash_requires_external_payload(self):
        wire_action = b"authoritative-action-v2"
        record = make_record(0, external_action=wire_action)
        self.assertEqual(record.action_hash_source, ActionHashSource.EXTERNAL.value)
        trajectory = make_trajectory(ticks=(0,), records=(record,))
        expected = strict_replay_inputs(trajectory)
        with self.assertRaises(ReplayValidationError):
            validate_replay(trajectory, **expected)
        self.assertTrue(validate_replay(
            trajectory, action_payloads=[wire_action], **expected,
        ))
        with self.assertRaises(ReplayValidationError):
            validate_replay(
                trajectory, action_payloads=[b"tampered"], **expected,
            )

    def test_wait_hold_cancel_semantics_and_lineage(self):
        with self.assertRaises(TrajectoryError):
            Action("wait", target="enemy")
        with self.assertRaises(TrajectoryError):
            Action("wait", timing="later")
        with self.assertRaises(TrajectoryError):
            Action("hold")
        with self.assertRaises(TrajectoryError):
            Action("cancel", skill=1, lineage_id="boundary")

        move = make_record(0)
        hold_action = action_for("hold", 1, move.recurrent_boundary_id)
        hold = make_record(1, executed=hold_action, original=hold_action, projected=hold_action)
        cancel_action = action_for("cancel", 2, move.recurrent_boundary_id)
        cancel = make_record(2, executed=cancel_action, original=cancel_action, projected=cancel_action)
        wait = make_record(3, kind="wait")
        trajectory = make_trajectory(
            ticks=(0, 1, 2, 3), records=(move, hold, cancel, wait),
        )
        self.assertTrue(validate_replay(trajectory, **strict_replay_inputs(trajectory)))

        bad_hold_action = action_for("hold", 1, "not-the-prior-boundary")
        bad_hold = make_record(
            1, executed=bad_hold_action, original=bad_hold_action,
            projected=bad_hold_action,
        )
        with self.assertRaises(TrajectoryError):
            make_trajectory(ticks=(0, 1), records=(move, bad_hold))

        first_cancel_action = action_for("cancel", 1, move.recurrent_boundary_id)
        first_cancel = make_record(
            1, executed=first_cancel_action, original=first_cancel_action,
            projected=first_cancel_action,
        )
        second_cancel_action = action_for("cancel", 2, move.recurrent_boundary_id)
        second_cancel = make_record(
            2, executed=second_cancel_action, original=second_cancel_action,
            projected=second_cancel_action,
        )
        with self.assertRaises(TrajectoryError):
            make_trajectory(
                ticks=(0, 1, 2), records=(move, first_cancel, second_cancel),
            )

        alias_cancel_action = action_for("cancel", 2, hold.recurrent_boundary_id)
        alias_cancel = make_record(
            2, executed=alias_cancel_action, original=alias_cancel_action,
            projected=alias_cancel_action,
        )
        root_cancel_action = action_for("cancel", 3, move.recurrent_boundary_id)
        root_cancel = make_record(
            3, executed=root_cancel_action, original=root_cancel_action,
            projected=root_cancel_action,
        )
        with self.assertRaises(TrajectoryError):
            make_trajectory(
                ticks=(0, 1, 2, 3),
                records=(move, hold, alias_cancel, root_cancel),
            )

        root_cancel_first_action = action_for("cancel", 2, move.recurrent_boundary_id)
        root_cancel_first = make_record(
            2, executed=root_cancel_first_action, original=root_cancel_first_action,
            projected=root_cancel_first_action,
        )
        alias_cancel_later_action = action_for("cancel", 3, hold.recurrent_boundary_id)
        alias_cancel_later = make_record(
            3, executed=alias_cancel_later_action, original=alias_cancel_later_action,
            projected=alias_cancel_later_action,
        )
        with self.assertRaises(TrajectoryError):
            make_trajectory(
                ticks=(0, 1, 2, 3),
                records=(move, hold, root_cancel_first, alias_cancel_later),
            )

    def test_audit_reports_outcomes_rejections_integrity_lineage_and_head_metrics(self):
        first = make_record(0)
        original = Action("skill", target="enemy-1", timing="now", skill=1)
        projected = Action("skill", target="enemy-1", timing="now", skill=2)
        second = make_record(
            1,
            valid=False,
            original=original,
            projected=projected,
            executed=projected,
            outcome_value=Outcome(
                reward=-2.0, team_reward=-0.5, damage=4.0,
                kills=1, deaths=1, terminal=True, winner="team-2",
            ),
        )
        trajectory = make_trajectory(ticks=(0, 1), records=(first, second))
        audit = audit_dataset(
            trajectory,
            head_losses={"kind": [0.1, 0.3], "skill": 0.4},
            payload_provider=lambda record: ReplayStep(
                observation(record.hero_id, record.tick), mask(record.tick),
            ),
        )
        self.assertEqual(audit.rejection_reason_counts["masked"], 1)
        self.assertEqual(audit.validity_counts["rejected"], 1)
        self.assertEqual(audit.outcome_totals.reward_total, -1.0)
        self.assertEqual(audit.outcome_totals.return_total, -1.0)
        self.assertEqual(audit.outcome_totals.kills_total, 1)
        self.assertEqual(audit.outcome_totals.terminal_count, 1)
        self.assertTrue(audit.recurrent_boundary_unique)
        self.assertEqual(audit.recurrent_lineage_failures, 0)
        self.assertEqual(audit.hash_integrity_failures.get("record", 0), 0)
        self.assertEqual(audit.head_metrics["kind"].accuracy, 1.0)
        self.assertEqual(audit.head_metrics["skill"].count, 1)
        self.assertEqual(audit.head_metrics["skill"].accuracy, 0.0)
        self.assertEqual(audit.per_skill_metrics["1"].count, 1)
        self.assertEqual(audit.per_skill_metrics["1"].accuracy, 0.0)
        self.assertEqual(audit.head_metrics["skill"].loss_total, 0.4)
        self.assertIsNone(audit.per_skill_metrics["1"].loss_total)
        self.assertIsNone(audit.per_skill_metrics["1"].loss_mean)
        self.assertAlmostEqual(audit.head_metrics["kind"].loss_total, 0.4)
        self.assertAlmostEqual(audit.head_metrics["kind"].loss_mean, 0.2)

        tampered = [make_record(0), make_record(1)]
        object.__setattr__(tampered[1], "executed_action", Action("wait"))
        object.__setattr__(tampered[1], "recurrent_parent_id", "bad-parent")
        object.__setattr__(tampered[1], "recurrent_boundary_id", tampered[0].recurrent_boundary_id)
        damaged = audit_dataset(tampered)
        self.assertGreaterEqual(damaged.hash_integrity_failures["record"], 1)
        self.assertGreaterEqual(damaged.hash_integrity_failures["action"], 1)
        self.assertEqual(damaged.duplicate_recurrent_boundaries, 1)
        self.assertEqual(damaged.recurrent_lineage_failures, 1)

    def test_head_metrics_ignore_99_inapplicable_non_skill_actions(self):
        records = [make_record(tick) for tick in range(99)]
        label = Action("skill", target="enemy-99", timing="now", skill=1)
        prediction = Action("skill", target="enemy-99", timing="now", skill=2)
        records.append(make_record(
            99, original=label, projected=prediction, executed=prediction,
        ))
        audit = audit_dataset(make_trajectory(
            ticks=tuple(range(100)), records=records,
        ))
        self.assertEqual(audit.head_metrics["kind"].count, 100)
        self.assertEqual(audit.head_metrics["kind"].accuracy, 1.0)
        self.assertEqual(audit.head_metrics["skill"].count, 1)
        self.assertEqual(audit.head_metrics["skill"].accuracy, 0.0)
        self.assertEqual(audit.per_skill_metrics["1"].count, 1)
        self.assertEqual(audit.per_skill_metrics["1"].accuracy, 0.0)

    def test_scalar_skill_loss_is_aggregate_only(self):
        skill_one = Action("skill", target="enemy-0", timing="now", skill=1)
        skill_two = Action("skill", target="enemy-1", timing="now", skill=2)
        trajectory = make_trajectory(
            ticks=(0, 1),
            records=(
                make_record(0, original=skill_one, projected=skill_one, executed=skill_one),
                make_record(1, original=skill_two, projected=skill_two, executed=skill_two),
            ),
        )
        aggregate = audit_dataset(trajectory, head_losses={"skill": 0.4})
        self.assertEqual(aggregate.head_metrics["skill"].loss_total, 0.4)
        self.assertIsNone(aggregate.per_skill_metrics["1"].loss_total)
        self.assertIsNone(aggregate.per_skill_metrics["2"].loss_total)

        per_record = audit_dataset(
            trajectory, head_losses={"skill": [0.1, 0.3]},
        )
        self.assertAlmostEqual(per_record.head_metrics["skill"].loss_total, 0.4)
        self.assertAlmostEqual(per_record.per_skill_metrics["1"].loss_total, 0.1)
        self.assertAlmostEqual(per_record.per_skill_metrics["2"].loss_total, 0.3)
        self.assertAlmostEqual(
            sum(metric.loss_total for metric in per_record.per_skill_metrics.values()),
            per_record.head_metrics["skill"].loss_total,
        )

        explicit = audit_dataset(
            trajectory, head_losses={"skill": {"1": 0.1, "2": 0.3}},
        )
        self.assertAlmostEqual(explicit.head_metrics["skill"].loss_total, 0.4)
        self.assertAlmostEqual(explicit.per_skill_metrics["1"].loss_total, 0.1)
        self.assertAlmostEqual(explicit.per_skill_metrics["2"].loss_total, 0.3)

    def test_group_aware_sampling_is_deterministic_and_never_crosses_matches(self):
        trajectories = []
        for match_index in range(4):
            for hero_index in range(2):
                trajectories.append(make_trajectory(
                    hero_id=f"hero-{hero_index}",
                    match_id=f"match-{match_index}",
                ))
        first = generate_group_stratified_sample_indices(
            trajectories, group_by="match_id", validation_fraction=0.25, seed=37,
        )
        second = generate_group_stratified_sample_indices(
            trajectories, group_by="match_id", validation_fraction=0.25, seed=37,
        )
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_indices) & set(first.validation_indices))
        self.assertEqual(len(first.validation_indices), len(set(first.validation_indices)))
        self.assertFalse(set(first.train_group_ids) & set(first.validation_group_ids))

        flattened_groups = [
            trajectory.match_id
            for trajectory in trajectories
            for _ in trajectory.records
        ]
        for group in set(flattened_groups):
            indices = {index for index, value in enumerate(flattened_groups) if value == group}
            self.assertTrue(
                indices.issubset(first.validation_indices)
                or indices.issubset(set(first.train_indices))
            )

        with self.assertRaises(ValueError):
            generate_stratified_sample_indices(
                ["move", "move", "wait", "wait"],
                group_ids=["match-a", "match-a", "match-b", "match-b"],
                validation_indices=[0],
            )

    def test_match_collection_requires_all_heroes_same_match_and_tick_coverage(self):
        expected = tuple(f"hero-{index}" for index in range(4))
        trajectories = tuple(
            make_trajectory(hero_id=hero_id, match_id="match-complete")
            for hero_id in expected
        )
        result = validate_match_trajectories(
            trajectories, expected_hero_ids=expected, heroes_per_team=2,
            expected_ticks=(0, 1, 2),
        )
        self.assertTrue(result)
        self.assertEqual(result.tick_count, 3)
        self.assertEqual(
            TeacherMatch(
                "match-complete", trajectories, expected_hero_ids=expected,
                heroes_per_team=2,
            ).match_id,
            "match-complete",
        )

        with self.assertRaises(MatchValidationError):
            validate_match_trajectories(
                trajectories[:-1], expected_hero_ids=expected, heroes_per_team=2,
            )

        with self.assertRaises(MatchValidationError):
            validate_match_trajectories(trajectories, expected_hero_ids=expected)

        wrong_match = list(trajectories)
        wrong_match[-1] = make_trajectory(hero_id="hero-3", match_id="other-match")
        with self.assertRaises(MatchValidationError):
            validate_match_trajectories(
                wrong_match, expected_hero_ids=expected, heroes_per_team=2,
            )

        short = list(trajectories)
        short[-1] = make_trajectory(
            hero_id="hero-3", match_id="match-complete", ticks=(0, 1),
        )
        with self.assertRaises(MatchValidationError):
            validate_match_trajectories(
                short, expected_hero_ids=expected, heroes_per_team=2,
            )

        ten_heroes = tuple(
            make_trajectory(hero_id=f"slot-{index}", match_id="match-10")
            for index in range(10)
        )
        self.assertTrue(validate_match_trajectories(ten_heroes))


if __name__ == "__main__":
    unittest.main()
