from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - dependency-specific CI gate
    raise unittest.SkipTest(f"AI-42 BC-v2 tests require torch and numpy: {exc}")

from tanat_ai40 import train_ai42_bc
from tanat_ai40.bc_metrics_ai42 import AI42MetricAccumulator, compute_ai42_metrics
from tanat_ai40.bc_profile_ai42 import (
    AI42ClassBalanceProfile,
    AI42ProfileError,
    PROFILE_HEADS,
    ordered_train_id_hash,
)
from tanat_ai40.env import ACTION_DTYPE
from tanat_ai40.learner_ai42 import (
    AI42Batch,
    AI42LearnerConfig,
    CheckpointError,
    TeacherStatus,
    compute_behavior_cloning_loss,
    prepare_ai42_supervision,
    save_ai42_checkpoint,
    build_learner_manifest,
    load_ai42_model_warm_start,
)
from tanat_ai40.model_ai42_actor import AI42Actor


def _batch() -> AI42Batch:
    ticks = 8
    hero = torch.zeros(1, ticks, 32)
    abilities = torch.zeros(1, ticks, 4, 40)
    entities = torch.zeros(1, ticks, 96, 16)
    global_state = torch.zeros(1, ticks, 32)
    entity_mask = torch.zeros(1, ticks, 96, dtype=torch.bool)
    entity_mask[..., :4] = True
    actions = torch.zeros(1, ticks, 4, dtype=torch.int64)
    actions[0, 0] = torch.tensor([1, 0, 4, 0])  # MOVE + local offset
    actions[0, 1] = torch.tensor([2, 1, 0, 0])  # ATTACK + target
    actions[0, 2] = torch.tensor([3, 0, 2, 0])  # SKILL + target/offset
    actions[0, 3] = torch.tensor([1, 0, 1, 2])  # MOVE + global anchor
    statuses = torch.tensor([[TeacherStatus.ACTION] * 4 + [TeacherStatus.WAIT, TeacherStatus.HOLD, TeacherStatus.CANCEL, TeacherStatus.NONE]])
    return AI42Batch(
        hero, abilities, entities, global_state, entity_mask,
        teacher_actions=actions,
        teacher_status=statuses,
        kind_mask=torch.ones(1, ticks, 8, dtype=torch.bool),
        target_mask=entity_mask.clone(),
        skill_target_mask=entity_mask.unsqueeze(2).expand(1, ticks, 4, 96).clone(),
    )


def _outputs(batch: AI42Batch) -> dict[str, torch.Tensor]:
    ticks = batch.sequence_length
    result = {
        "control": torch.full((1, ticks, 4), -100.0),
        "kind": torch.full((1, ticks, 8), -100.0),
        "target": torch.full((1, ticks, 8, 96), -100.0),
        "offset": torch.full((1, ticks, 8, 81), -100.0),
        "anchor": torch.full((1, ticks, 8, 15), -100.0),
    }
    actions = batch.teacher_actions
    for time in range(ticks):
        result["control"][0, time, 0] = 10.0  # ISSUE; non-action controls remain intentionally wrong.
        kind = int(actions[0, time, 0])
        result["kind"][0, time, kind] = 10.0
        result["target"][0, time, kind, int(actions[0, time, 1])] = 10.0
        result["offset"][0, time, kind, int(actions[0, time, 2])] = 10.0
        result["anchor"][0, time, kind, int(actions[0, time, 3])] = 10.0
    # Make one valid offset a top-5 hit but not top-1.  Grid cells 4 and 5
    # are Manhattan-adjacent, so the known mean distance is 0.5.
    result["offset"][0, 0, 1, 5] = 11.0
    result["offset"][0, 0, 1, 4] = 9.0
    return result


class AI42BCV2Tests(unittest.TestCase):
    def test_production_config_has_single_data_derived_policy(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "ai42_bc_training.json"
        defaults = train_ai42_bc._training_config_defaults(config_path)
        self.assertEqual(defaults["hidden_size"], 192)
        self.assertEqual(defaults["model_width"], 192)
        self.assertEqual(defaults["entity_layers"], 2)
        self.assertEqual(defaults["num_heads"], 6)
        self.assertEqual(defaults["class_balance_power"], 1.0)
        self.assertEqual(defaults["offset_distance_loss_weight"], 1.0)
        self.assertEqual(defaults["validation_batch_size"], 256)
        self.assertNotIn("class_weight_overrides", defaults)
        self.assertNotIn("combat_focus", defaults)
        self.assertNotIn("trainable_scope", defaults)

    def test_profile_is_train_only_immutable_canonical_and_tamper_evident(self) -> None:
        batch = _batch()
        profile = AI42ClassBalanceProfile.from_batches(
            [batch], dataset_manifest_hash="a" * 64, train_match_ids=("train-a", "train-b"),
        )
        self.assertEqual(profile.train_match_ids_hash, ordered_train_id_hash(("train-a", "train-b")))
        self.assertEqual(profile.counts["control"], (4, 1, 1, 1))
        self.assertEqual(profile.counts["kind"][:4], (0, 2, 1, 1))
        self.assertEqual(profile.counts["target"][0:2], (1, 1))
        self.assertEqual(profile.counts["offset"][2], 1)
        self.assertEqual(profile.counts["anchor"][2], 1)
        self.assertAlmostEqual(sum(value for value in profile.weights["control"] if value) / 4.0, 1.0, places=6)
        self.assertTrue(all(value == 1.0 for value in profile.weights["kind"]))
        self.assertTrue(all(value == 1.0 for value in profile.weights["target"]))
        self.assertTrue(all(value == 1.0 for value in profile.weights["offset"]))
        with self.assertRaises(TypeError):
            profile.counts["control"] = (1,)  # type: ignore[index]
        self.assertEqual(profile.to_json(), profile.to_json())
        self.assertEqual(AI42ClassBalanceProfile.from_json(profile.to_json()).profile_hash, profile.profile_hash)

        tampered = profile.to_dict()
        tampered["weights"]["control"][1] += 0.01
        unsigned = dict(tampered)
        unsigned.pop("profile_hash")
        tampered["profile_hash"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        with self.assertRaises(AI42ProfileError):
            AI42ClassBalanceProfile.from_mapping(tampered)

    def test_profile_dataset_adapter_never_reads_validation(self) -> None:
        calls: list[str] = []

        class Dataset:
            manifest = {"manifest_hash": "b" * 64, "dataset_schema_version": "v2", "shard_schema_version": "s2"}
            manifest_hash = "b" * 64

            def split_match_ids(self):
                return {"train": ("train-a",), "validation": ("validation-a",)}

            def iter_matches(self, split):
                calls.append(split)
                if split != "train":
                    raise AssertionError("profile consumed validation data")
                yield "train-a", {
                    "hero": np.zeros((1, 10, 32), dtype="<f4"),
                    "abilities": np.zeros((1, 10, 4, 40), dtype="<f4"),
                    "entities": np.zeros((1, 10, 96, 16), dtype="<f4"),
                    "global": np.zeros((1, 10, 32), dtype="<f4"),
                    "entity_mask": np.ones((1, 10, 96), dtype="u1"),
                    "kind_mask": np.ones((1, 10, 8), dtype="u1"),
                    "target_mask": np.ones((1, 10, 96), dtype="u1"),
                    "skill_target_mask": np.ones((1, 10, 4, 96), dtype="u1"),
                    "teacher_status": np.full((1, 10), TeacherStatus.WAIT, dtype="u1"),
                    "teacher_action": np.zeros((1, 10), dtype=ACTION_DTYPE),
                    "step": np.asarray([0], dtype="<u4"), "done": np.asarray([1], dtype="u1"),
                }

        profile = AI42ClassBalanceProfile.from_dataset(Dataset(), sequence_length=1, batch_size=2)
        self.assertEqual(calls, ["train"])
        self.assertEqual(profile.train_match_ids, ("train-a",))

    def test_shared_masks_match_loss_counts_and_metrics_known_values(self) -> None:
        batch = _batch()
        prepared = prepare_ai42_supervision(batch)
        actor = AI42Actor(hidden_size=8, model_width=8, entity_layers=1, num_heads=2, ff_multiplier=1)
        result = compute_behavior_cloning_loss(actor, batch, AI42LearnerConfig(model_kwargs={"hidden_size": 8, "model_width": 8, "entity_layers": 1, "num_heads": 2, "ff_multiplier": 1}))
        for head in PROFILE_HEADS:
            expected = tuple(int(item) for item in torch.bincount(prepared.labels[head][prepared.active[head]], minlength=result.class_counts[head].__len__()).tolist()) if bool(prepared.active[head].any()) else tuple(0 for _ in result.class_counts[head])
            self.assertEqual(result.class_counts[head], expected)
            self.assertGreaterEqual(result.metrics["heads"][head]["weighted_denominator"], 0.0)

        metrics = compute_ai42_metrics(batch, _outputs(batch))
        self.assertEqual(metrics["heads"]["control"]["count"], 7)
        self.assertEqual(metrics["heads"]["kind"]["count"], 4)
        self.assertEqual(metrics["heads"]["kind"]["per_class"]["0"]["support"], 0)
        self.assertEqual(metrics["offset"]["count"], 2)
        self.assertEqual(metrics["offset"]["top5_correct"], 2)
        self.assertEqual(metrics["offset"]["top1_correct"], 1)
        self.assertAlmostEqual(metrics["offset"]["mean_manhattan_grid_distance"], 0.5)
        self.assertEqual(metrics["action"]["count"], 4)
        self.assertEqual(metrics["action"]["end_to_end_correct"], 3)
        json.dumps(metrics, sort_keys=True, allow_nan=False)

    def test_offset_loss_directly_penalizes_expected_grid_distance(self) -> None:
        batch = _batch()
        actor = AI42Actor(
            hidden_size=8, model_width=8, entity_layers=1,
            num_heads=2, ff_multiplier=1,
        )
        near = _outputs(batch)
        far = {name: value.clone() for name, value in near.items()}
        far["offset"][0, 0, 1, 5] = -100.0
        far["offset"][0, 0, 1, 80] = 11.0
        config = AI42LearnerConfig(
            model_kwargs={
                "hidden_size": 8, "model_width": 8, "entity_layers": 1,
                "num_heads": 2, "ff_multiplier": 1,
            },
            offset_distance_loss_weight=1.0,
        )
        near_result = compute_behavior_cloning_loss(actor, batch, config, outputs=near)
        far_result = compute_behavior_cloning_loss(actor, batch, config, outputs=far)
        self.assertGreater(
            far_result.metrics["heads"]["offset"]["expected_normalized_manhattan_loss"],
            near_result.metrics["heads"]["offset"]["expected_normalized_manhattan_loss"],
        )
        self.assertGreater(far_result.head_losses["offset"], near_result.head_losses["offset"])

    def test_weighted_probe_aggregation_is_partition_invariant(self) -> None:
        class FakeLearner:
            actor = torch.nn.Linear(1, 1)
            config = SimpleNamespace(head_weights={"control": 1.0})

            def __init__(self, parts):
                self.parts = iter(parts)

            def loss(self, _batch):
                return next(self.parts)

        def item(numerator, denominator, count):
            return SimpleNamespace(
                loss=torch.tensor(numerator / denominator), head_losses={"control": torch.tensor(numerator / denominator)},
                metrics={"supervised_count": count, "action_count": 0, "control_count": count, "heads": {"control": {"count": count, "accuracy": 0.5, "weighted_numerator": numerator, "weighted_denominator": denominator}}},
            )

        one = train_ai42_bc.evaluate_probe(FakeLearner([item(7.0, 3.0, 3)]), [object()])
        split = train_ai42_bc.evaluate_probe(FakeLearner([item(2.0, 1.0, 1), item(5.0, 2.0, 2)]), [object(), object()])
        self.assertAlmostEqual(one.head_losses["control"], split.head_losses["control"])
        self.assertEqual(one.head_weighted_numerators, split.head_weighted_numerators)
        self.assertEqual(one.head_weighted_denominators, split.head_weighted_denominators)

    def _summary(self, *, loss=1.0, mutate=None):
        heads = {
            "control": {"count": 7, "micro_accuracy": 0.5, "supported_macro_f1": 0.5, "balanced_accuracy": 0.5, "per_class": {str(i): {"support": 4 if i == 0 else 1, "recall": 0.5} for i in range(4)}},
            "kind": {
                "count": 4,
                "micro_accuracy": 0.5,
                "per_class": {
                    str(i): {
                        "support": 2 if i in (0, 1) else 0,
                        "recall": 0.5 if i in (0, 1) else 0.0,
                    }
                    for i in range(8)
                },
            },
            "target": {"count": 2, "micro_accuracy": 0.5},
            "anchor": {"count": 1, "micro_accuracy": 0.5},
        }
        metrics = {"heads": heads, "action": {"count": 4, "end_to_end_accuracy": 0.2}, "offset": {"count": 1, "mean_manhattan_grid_distance": 2.0}}
        if mutate:
            mutate(metrics)
        return train_ai42_bc.ProbeSummary(loss, 1, 10, 4, 7, {"control": 0.4, "kind": 0.4, "target": 0.4, "anchor": 0.4}, metrics=metrics, head_denominators={"control": 7, "kind": 4, "target": 2, "anchor": 1})

    def test_gate_uses_stable_aggregate_metrics_and_fails_closed(self) -> None:
        baseline = self._summary()

        def candidate(mutate=None, loss=0.99, control_loss=0.4):
            item = self._summary(loss=loss, mutate=mutate)
            return replace(item, head_losses={"control": control_loss, "kind": 0.4, "target": 0.4, "anchor": 0.4})

        good = candidate(lambda m: (m["heads"]["control"].update(supported_macro_f1=0.6, balanced_accuracy=0.6), m["heads"]["kind"].update(micro_accuracy=0.5), m["heads"]["target"].update(micro_accuracy=0.5), m["heads"]["anchor"].update(micro_accuracy=0.5), m.update(action={"count": 4, "end_to_end_accuracy": 0.3}, offset={"count": 1, "mean_manhattan_grid_distance": 2.0})))
        self.assertTrue(train_ai42_bc.promotion_gate(baseline, good)["accepted"])
        branches = {
            "total_validation_loss_improvement": lambda m: None,
            "control_balanced_accuracy_floor": lambda m: m["heads"]["control"].update(balanced_accuracy=0.48),
            "end_to_end_action_improves": lambda m: m.update(action={"count": 4, "end_to_end_accuracy": 0.2}),
            "offset_distance_no_worse": lambda m: m.update(offset={"count": 1, "mean_manhattan_grid_distance": 2.1}),
        }
        for name, mutate in branches.items():
            candidate_kwargs = {}
            if name == "total_validation_loss_improvement":
                candidate_kwargs["loss"] = 0.996
            failed = train_ai42_bc.promotion_gate(
                baseline, candidate(mutate, **candidate_kwargs),
            )["failed"]
            self.assertIn(name, failed, name)
        for head in ("kind", "target", "anchor"):
            failed = train_ai42_bc.promotion_gate(baseline, candidate(lambda m, h=head: m["heads"][h].update(micro_accuracy=0.48)))["failed"]
            self.assertIn(f"{head}_accuracy_floor", failed)
        rare_metrics = copy.deepcopy(good.metrics)
        rare_metrics["heads"]["kind"]["per_class"]["0"]["recall"] = 0.0
        rare = train_ai42_bc.promotion_gate(baseline, replace(good, metrics=rare_metrics))
        self.assertTrue(rare["accepted"])
        self.assertEqual(rare["diagnostics"]["kind_recall_after"][0], 0.0)
        missing = train_ai42_bc.promotion_gate(train_ai42_bc.ProbeSummary(1, 1, 1, 1, 1, {}), good)
        self.assertFalse(missing["accepted"])

    def test_gate_rejects_every_malformed_composite_metric_schema(self) -> None:
        baseline = self._summary()
        good = self._summary(
            loss=0.99,
            mutate=lambda metrics: (
                metrics["heads"]["control"].update(supported_macro_f1=0.6, balanced_accuracy=0.6),
                metrics.update(
                    action={"count": 4, "end_to_end_accuracy": 0.3},
                    offset={"count": 1, "mean_manhattan_grid_distance": 2.0},
                ),
            ),
        )

        def changed(path, value=Ellipsis):
            metrics = copy.deepcopy(good.metrics)
            parent = metrics
            for item in path[:-1]:
                parent = parent[item]
            if value is Ellipsis:
                parent.pop(path[-1], None)
            else:
                parent[path[-1]] = value
            return replace(good, metrics=metrics)

        changed_target_count = changed(("heads", "target", "count"), 3)
        changed_action_count = changed(("heads", "kind", "count"), 5)
        changed_action_metrics = copy.deepcopy(changed_action_count.metrics)
        changed_action_metrics["action"]["count"] = 5
        malformed = {
            "missing heads mapping": changed(("heads",)),
            "missing action mapping": changed(("action",)),
            "missing offset mapping": changed(("offset",)),
            "missing supported recall": changed(("heads", "control", "per_class", "0", "recall")),
            "missing kind accuracy": changed(("heads", "kind", "micro_accuracy")),
            "missing target accuracy": changed(("heads", "target", "micro_accuracy")),
            "missing anchor accuracy": changed(("heads", "anchor", "micro_accuracy")),
            "missing control count": changed(("heads", "control", "count")),
            "missing kind count": changed(("heads", "kind", "count")),
            "missing target count": changed(("heads", "target", "count")),
            "missing anchor count": changed(("heads", "anchor", "count")),
            "missing action count": changed(("action", "count")),
            "missing offset count": changed(("offset", "count")),
            "missing per-class data": changed(("heads", "control", "per_class")),
            "missing kind per-class data": changed(("heads", "kind", "per_class")),
            "missing kind recall": changed(("heads", "kind", "per_class", "0", "recall")),
            "NaN accuracy": changed(("heads", "kind", "micro_accuracy"), float("nan")),
            "infinite distance": changed(("offset", "mean_manhattan_grid_distance"), float("inf")),
            "string accuracy": changed(("heads", "kind", "micro_accuracy"), "0.5"),
            "boolean accuracy": changed(("heads", "kind", "micro_accuracy"), True),
            "boolean support": changed(("heads", "control", "per_class", "0", "support"), True),
            "support mismatch": changed(("heads", "control", "per_class", "0", "support"), 2),
            "boolean kind support": changed(("heads", "kind", "per_class", "0", "support"), True),
            "kind support mismatch": changed(("heads", "kind", "per_class", "0", "support"), 3),
            "head denominator mismatch": replace(good, head_denominators={**good.head_denominators, "target": 3}),
            "control summary mismatch": replace(good, control_count=8),
            "action summary mismatch": replace(good, action_count=5),
            "baseline/candidate head count mismatch": replace(
                changed_target_count,
                head_denominators={**good.head_denominators, "target": 3},
            ),
            "baseline/candidate action count mismatch": replace(
                changed_action_count,
                metrics=changed_action_metrics,
                action_count=5,
                head_denominators={**good.head_denominators, "kind": 5},
            ),
            "offset count mismatch": changed(("offset", "count"), 2),
        }
        for label, candidate in malformed.items():
            with self.subTest(label=label):
                gate = train_ai42_bc.promotion_gate(baseline, candidate)
                self.assertFalse(gate["accepted"])
                self.assertEqual(gate["checks"], {"metrics_complete": False})
                self.assertEqual(gate["failed"], ["metrics_complete"])

    def test_model_only_warm_start_validates_and_does_not_transfer_optimizer_or_rng(self) -> None:
        kwargs = {"hidden_size": 8, "model_width": 8, "entity_layers": 1, "num_heads": 2, "ff_multiplier": 1}
        source = AI42Actor(**kwargs)
        source_config = AI42LearnerConfig(model_kwargs=kwargs, class_balance_power=1.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        sum(parameter.sum() for parameter in source.parameters()).backward()
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        manifest = build_learner_manifest(source, source_config, "a" * 64, protocol_version=13, dataset_schema_version="v2", shard_schema_version="s2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            save_ai42_checkpoint(path, source, source_optimizer, manifest, step=9, epoch=2, extra={"batch_cursor": 17})
            target = AI42Actor(**kwargs)
            target_config = AI42LearnerConfig(model_kwargs=kwargs, class_balance_power=0.5)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
            target_manifest = build_learner_manifest(target, target_config, "a" * 64, protocol_version=13, dataset_schema_version="v2", shard_schema_version="s2")
            random.seed(13); np.random.seed(13); torch.manual_seed(13)
            before_rng = torch.get_rng_state().clone()
            state = load_ai42_model_warm_start(path, target, target_manifest)
            self.assertEqual(state.source_step, 9)
            self.assertFalse(target_optimizer.state_dict()["state"])
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)
            changed_dataset_manifest = build_learner_manifest(
                target, target_config, "b" * 64,
                protocol_version=13, dataset_schema_version="v2", shard_schema_version="s2",
            )
            with self.assertRaisesRegex(CheckpointError, "dataset_hash"):
                load_ai42_model_warm_start(path, target, changed_dataset_manifest)
            changed_state = load_ai42_model_warm_start(
                path, target, changed_dataset_manifest, allow_dataset_change=True,
            )
            self.assertEqual(changed_state.source_dataset_hash, "a" * 64)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            name = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][name] = payload["model_state_dict"][name] + 1
            torch.save(payload, path)
            with self.assertRaises(CheckpointError):
                load_ai42_model_warm_start(path, target, target_manifest)


if __name__ == "__main__":
    unittest.main()
