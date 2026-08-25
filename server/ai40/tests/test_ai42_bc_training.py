from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from tanat_ai40 import train_ai42_bc
from tanat_ai40.env import ACTION_DTYPE
from tanat_ai40.learner_ai42 import TeacherStatus
from tanat_ai40.model_ai42_actor import AI42Actor


class _TinyDataset:
    manifest = {
        "manifest_hash": "a" * 64,
        "dataset_schema_version": "AI42-dataset-v1",
        "shard_schema_version": "AI42-go-shard-v2",
    }

    def __len__(self) -> int:
        return 2

    @property
    def manifest_hash(self) -> str:
        return self.manifest["manifest_hash"]

    def split_match_ids(self) -> dict[str, tuple[str, ...]]:
        return {"train": ("train-000",), "validation": ("validation-000",)}

    def iter_matches(self, split: str):
        ticks = 1
        arrays = {
            "hero": np.zeros((ticks, 10, 32), dtype="<f4"),
            "abilities": np.zeros((ticks, 10, 4, 40), dtype="<f4"),
            "entities": np.zeros((ticks, 10, 96, 16), dtype="<f4"),
            "global": np.zeros((ticks, 10, 32), dtype="<f4"),
            "entity_mask": np.zeros((ticks, 10, 96), dtype="u1"),
            "kind_mask": np.ones((ticks, 10, 8), dtype="u1"),
            "target_mask": np.zeros((ticks, 10, 96), dtype="u1"),
            "skill_target_mask": np.zeros((ticks, 10, 4, 96), dtype="u1"),
            "teacher_status": np.full((ticks, 10), TeacherStatus.WAIT, dtype="u1"),
            "teacher_action": np.zeros((ticks, 10), dtype=ACTION_DTYPE),
            "step": np.asarray([0], dtype="<u4"),
            "done": np.asarray([1], dtype="u1"),
        }
        arrays["entity_mask"][..., 0] = 1
        arrays["target_mask"][..., 0] = 1
        arrays["skill_target_mask"][..., 0, 0] = 1
        yield f"{split}-000", arrays


class _StratifiedDataset(_TinyDataset):
    manifest = {
        "manifest_hash": "b" * 64,
        "dataset_schema_version": "AI42-dataset-v1",
        "shard_schema_version": "AI42-go-shard-v2",
        "matches": [
            {"match_id": "validation-a", "scenario": "alpha"},
            {"match_id": "validation-b", "scenario": "beta"},
            {"match_id": "validation-c", "scenario": "alpha"},
            {"match_id": "validation-d", "scenario": "beta"},
            {"match_id": "train-a", "scenario": "alpha"},
        ],
    }

    def split_match_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": ("train-a",),
            "validation": ("validation-a", "validation-b", "validation-c", "validation-d"),
        }


class _MixedControllerDataset(_TinyDataset):
    """Starts with AI20 UNAVAILABLE-only slots, then supplies WAIT rows."""

    def iter_matches(self, split: str):
        ticks = 3
        arrays = {
            "hero": np.zeros((ticks, 10, 32), dtype="<f4"),
            "abilities": np.zeros((ticks, 10, 4, 40), dtype="<f4"),
            "entities": np.zeros((ticks, 10, 96, 16), dtype="<f4"),
            "global": np.zeros((ticks, 10, 32), dtype="<f4"),
            "entity_mask": np.zeros((ticks, 10, 96), dtype="u1"),
            "kind_mask": np.ones((ticks, 10, 8), dtype="u1"),
            "target_mask": np.zeros((ticks, 10, 96), dtype="u1"),
            "skill_target_mask": np.zeros((ticks, 10, 4, 96), dtype="u1"),
            "teacher_status": np.full((ticks, 10), TeacherStatus.WAIT, dtype="u1"),
            "teacher_action": np.zeros((ticks, 10), dtype=ACTION_DTYPE),
            "step": np.arange(ticks, dtype="<u4"),
            "done": np.asarray([0, 0, 1], dtype="u1"),
        }
        arrays["teacher_status"][0, :] = TeacherStatus.UNAVAILABLE
        arrays["entity_mask"][..., 0] = 1
        arrays["target_mask"][..., 0] = 1
        arrays["skill_target_mask"][..., 0, 0] = 1
        yield f"{split}-000", arrays


def _args(output: Path, *extra: str):
    return train_ai42_bc.build_parser().parse_args([
        "--execute", "--device", "cpu", "--dataset", "dummy", "--output", str(output),
        "--hidden-size", "8", "--model-width", "8", "--entity-layers", "1",
        "--num-heads", "2", "--ff-multiplier", "1", "--timing-bins", "2",
        "--sequence-length", "1", "--batch-size", "1", "--epochs", "1",
        "--validation-batches", "1", *extra,
    ])


def _mock_probe_summary(loss: float, control_loss: float | None = None) -> train_ai42_bc.ProbeSummary:
    """Complete v2 evidence fixture for tests that exercise publication."""

    quality = max(0.0, min(1.0, 1.0 - loss / 2.0))
    control = {
        "micro_accuracy": quality,
        "supported_macro_f1": quality,
        "balanced_accuracy": quality,
        "per_class": {
            str(index): {"support": 1 if index == 0 else 0, "recall": quality}
            for index in range(4)
        },
        "count": 1,
    }
    heads = {
        "control": control,
        "kind": {
            "count": 1,
            "micro_accuracy": quality,
            "per_class": {
                str(index): {
                    "support": 1 if index == 1 else 0,
                    "recall": quality if index == 1 else 0.0,
                }
                for index in range(8)
            },
        },
        "target": {"count": 1, "micro_accuracy": quality},
        "anchor": {"count": 1, "micro_accuracy": quality},
    }
    metrics = {
        "heads": heads,
        "action": {"count": 1, "end_to_end_accuracy": quality},
        "offset": {"count": 1, "mean_manhattan_grid_distance": 1.0 - quality},
    }
    return train_ai42_bc.ProbeSummary(
        loss, 1, 1, 1, 1,
        {"control": loss if control_loss is None else control_loss},
        head_accuracies={name: quality for name in ("control", "kind", "target", "anchor")},
        head_denominators={name: 1 for name in ("control", "kind", "target", "anchor")},
        metrics=metrics,
    )


class AI42BCTrainingTests(unittest.TestCase):
    def test_promotion_rejects_zero_recall_for_supported_action_kind(self) -> None:
        baseline = _mock_probe_summary(1.0)
        candidate = _mock_probe_summary(0.5)
        baseline.metrics["heads"]["kind"]["per_class"]["1"]["support"] = 0
        baseline.metrics["heads"]["kind"]["per_class"]["2"]["support"] = 1
        baseline.metrics["heads"]["kind"]["per_class"]["2"]["recall"] = 0.0
        candidate.metrics["heads"]["kind"]["per_class"]["1"]["support"] = 0
        candidate.metrics["heads"]["kind"]["per_class"]["2"]["support"] = 1
        candidate.metrics["heads"]["kind"]["per_class"]["2"]["recall"] = 0.0
        gate = train_ai42_bc.promotion_gate(baseline, candidate)
        self.assertFalse(gate["accepted"])
        self.assertIn("kind_recall_2_coverage", gate["failed"])

    def test_budget_has_hard_cap_and_fake_clock_stops_before_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory), "--max-optimizer-seconds", "300")
            clock_values = iter((0.0, 300.0, 300.0))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()):
                report = train_ai42_bc.train(args, clock=lambda: next(clock_values, 300.0))
            self.assertTrue(report["deadline_reached"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["elapsed_optimizer_seconds"], 300.0)

            with self.assertRaisesRegex(train_ai42_bc.AI42TrainingError, r"\(0, 300\]"):
                train_ai42_bc.train(_args(Path(directory), "--max-optimizer-seconds", "300.01"), clock=lambda: 0.0)

    def test_parameter_updates_and_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)):
                report = train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            self.assertTrue(report["accepted"])
            self.assertTrue((output / "accepted.pt").is_file())
            first_accepted = (output / "accepted.pt").read_bytes()
            payload = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            self.assertEqual(payload["step"], 1)
            self.assertGreaterEqual(report["optimizer_steps"], 1)
            self.assertTrue((output / "best.pt").is_file())
            torch.manual_seed(4242)
            initial = AI42Actor(hidden_size=8, model_width=8, entity_layers=1, num_heads=2, ff_multiplier=1, timing_bins=2)
            self.assertTrue(any(
                not torch.equal(payload["model_state_dict"][name], value)
                for name, value in initial.state_dict().items()
            ))

            summaries = iter((
                _mock_probe_summary(0.4), _mock_probe_summary(0.4),
                _mock_probe_summary(0.6), _mock_probe_summary(0.6),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)):
                rejected = train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            self.assertFalse(rejected["accepted"])
            self.assertEqual(first_accepted, (output / "accepted.pt").read_bytes())

    def test_class_weight_provenance_is_bound_to_manifest_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = _args(output, "--max-steps", "1")
            args.class_weight_overrides = {"control": [0.0, 1.0, 0.0, 0.0]}
            args.head_weights = {
                "control": 1.0, "kind": 1.5, "target": 1.0, "offset": 2.0, "anchor": 1.0,
            }
            summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)):
                report = train_ai42_bc.train(args, clock=lambda: 0.0)

            payload = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            manifest = payload["manifest"]
            extra = payload["extra"]
            self.assertEqual(manifest["class_weight_overrides"], {"control": [0.0, 1.0, 0.0, 0.0]})
            self.assertEqual(manifest["class_weight_overrides_hash"], report["class_weight_overrides_hash"])
            self.assertEqual(manifest["class_weights"], report["class_weights"])
            self.assertEqual(extra["class_weight_overrides_hash"], report["class_weight_overrides_hash"])
            self.assertEqual(extra["class_weights"], report["class_weights"])
            self.assertEqual(extra["class_weight_provenance"]["final"], report["class_weights"])
            self.assertEqual(manifest["head_weights"], args.head_weights)
            self.assertEqual(manifest["head_weights_hash"], report["head_weights_hash"])
            self.assertEqual(extra["head_weights"], args.head_weights)
            self.assertEqual(extra["head_weights_hash"], report["head_weights_hash"])
            self.assertEqual(report["head_weights_hash"], train_ai42_bc.head_weights_hash(args.head_weights))
            self.assertEqual(report["manifest_digest"], train_ai42_bc.manifest_digest(manifest))

    def test_resume_restores_step_and_atomic_report_failure_preserves_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()):
                first = train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
                resumed = train_ai42_bc.train(
                    _args(output, "--max-steps", "2", "--resume", str(output / "latest.pt")),
                    clock=lambda: 0.0,
                )
            self.assertEqual(first["global_step"], 1)
            self.assertEqual(resumed["resume"]["start_step"], 1)
            self.assertEqual(resumed["global_step"], 2)

            report = output / "atomic.json"
            report.write_text("old", encoding="utf-8")
            with mock.patch.object(train_ai42_bc.os, "replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    train_ai42_bc.atomic_write_json(report, {"new": True})
            self.assertEqual(report.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(output.glob(".atomic.json.*.tmp")), [])

    def test_resumed_training_matches_uninterrupted_weights_and_persists_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            partial = root / "partial"
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()):
                full_report = train_ai42_bc.train(
                    _args(full, "--max-steps", "2", "--checkpoint-interval", "1"), clock=lambda: 0.0,
                )
                first_report = train_ai42_bc.train(
                    _args(partial, "--max-steps", "1", "--checkpoint-interval", "1"), clock=lambda: 0.0,
                )
                resumed_report = train_ai42_bc.train(
                    _args(
                        partial, "--max-steps", "2", "--checkpoint-interval", "1",
                        "--resume", str(partial / "latest.pt"),
                    ),
                    clock=lambda: 0.0,
                )
            self.assertEqual(first_report["batch_plan"]["batch_cursor"], 1)
            self.assertEqual(resumed_report["resume"]["start_step"], 1)
            self.assertEqual(resumed_report["batch_plan"]["batch_cursor"], 2)
            self.assertGreaterEqual(first_report["periodic_checkpoint_count"], 1)
            self.assertGreaterEqual(resumed_report["periodic_checkpoint_count"], 1)
            self.assertGreaterEqual(resumed_report["periodic_checkpoint_seconds"], 0.0)
            full_payload = torch.load(full / "latest.pt", map_location="cpu", weights_only=True)
            resumed_payload = torch.load(partial / "latest.pt", map_location="cpu", weights_only=True)
            self.assertEqual(full_report["global_step"], resumed_report["global_step"])
            self.assertEqual(set(full_payload["model_state_dict"]), set(resumed_payload["model_state_dict"]))
            for name in full_payload["model_state_dict"]:
                torch.testing.assert_close(
                    full_payload["model_state_dict"][name], resumed_payload["model_state_dict"][name],
                    atol=0, rtol=0,
                )
            self.assertEqual(
                full_payload["artifact_hashes"]["optimizer"],
                resumed_payload["artifact_hashes"]["optimizer"],
            )

    def test_mixed_controller_skips_empty_batches_and_resume_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _MixedControllerDataset()
            args = _args(root / "probe")
            raw_batches = list(train_ai42_bc.iter_ai42_dataset_batches(
                dataset, split="train", sequence_length=1, batch_size=1,
            ))
            eligible_batches = list(train_ai42_bc._iter_plan_batches(
                dataset, ("train-000",), "train", args, torch.device("cpu"),
            ))
            self.assertTrue(raw_batches[0].supervision_mask.sum().item() == 0)
            self.assertGreater(len(raw_batches), len(eligible_batches))
            self.assertTrue(all(bool(batch.supervision_mask.any()) for batch in eligible_batches))

            full = root / "full"
            partial = root / "partial"
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=dataset):
                full_report = train_ai42_bc.train(
                    _args(full, "--max-steps", "2", "--checkpoint-interval", "1"), clock=lambda: 0.0,
                )
                first_report = train_ai42_bc.train(
                    _args(partial, "--max-steps", "1", "--checkpoint-interval", "1"), clock=lambda: 0.0,
                )
                resumed_report = train_ai42_bc.train(
                    _args(
                        partial, "--max-steps", "2", "--checkpoint-interval", "1",
                        "--resume", str(partial / "latest.pt"),
                    ),
                    clock=lambda: 0.0,
                )
            self.assertGreater(full_report["pre_validation"]["batches"], 0)
            self.assertGreater(full_report["post_validation"]["batches"], 0)
            self.assertEqual(first_report["batch_plan"]["batch_cursor"], 1)
            self.assertEqual(resumed_report["batch_plan"]["batch_cursor"], 2)
            full_payload = torch.load(full / "latest.pt", map_location="cpu", weights_only=True)
            resumed_payload = torch.load(partial / "latest.pt", map_location="cpu", weights_only=True)
            self.assertEqual(full_report["global_step"], resumed_report["global_step"])
            for name in full_payload["model_state_dict"]:
                torch.testing.assert_close(
                    full_payload["model_state_dict"][name], resumed_payload["model_state_dict"][name],
                    atol=0, rtol=0,
                )
            self.assertEqual(
                full_payload["artifact_hashes"]["optimizer"], resumed_payload["artifact_hashes"]["optimizer"],
            )

    def test_validation_probe_is_seeded_hash_ranked_and_stratified(self) -> None:
        dataset = _StratifiedDataset()
        first = train_ai42_bc._ranked_match_ids(dataset, "validation", 4242, limit=4)
        second = train_ai42_bc._ranked_match_ids(dataset, "validation", 4242, limit=4)
        changed = train_ai42_bc._ranked_match_ids(dataset, "validation", 4244, limit=4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, ("validation-a", "validation-b", "validation-c", "validation-d"))
        scenarios = {entry["match_id"]: entry["scenario"] for entry in dataset.manifest["matches"]}
        self.assertEqual({scenarios[match_id] for match_id in first[:2]}, {"alpha", "beta"})
        self.assertNotEqual(first, changed)

    def test_report_failure_does_not_replace_accepted_or_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            accepted_summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(accepted_summaries)):
                train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            accepted_before = (output / "accepted.pt").read_bytes()
            best_before = (output / "best.pt").read_bytes()
            pointer_before = (output / train_ai42_bc.ACCEPTED_POINTER_FILENAME).read_bytes()
            improved_summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.4), _mock_probe_summary(0.4),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(improved_summaries)), \
                 mock.patch.object(train_ai42_bc, "_atomic_write_json", side_effect=OSError("injected report failure")):
                with self.assertRaises(OSError):
                    train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            self.assertEqual(accepted_before, (output / "accepted.pt").read_bytes())
            self.assertEqual(best_before, (output / "best.pt").read_bytes())
            self.assertEqual(pointer_before, (output / train_ai42_bc.ACCEPTED_POINTER_FILENAME).read_bytes())

    def test_corrupt_prior_generation_with_stale_payload_hashes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            accepted_summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(accepted_summaries)):
                train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            pointer_path = output / train_ai42_bc.ACCEPTED_POINTER_FILENAME
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            generation_path = output / pointer["checkpoint"]
            payload = torch.load(generation_path, map_location="cpu", weights_only=True)
            tensor_name = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][tensor_name] = payload["model_state_dict"][tensor_name] + 1
            torch.save(payload, generation_path)
            # Refresh the pointer's file digest so the failure exercises the
            # serialized payload/artifact digests, not only the pointer guard.
            pointer["sha256"] = train_ai42_bc._sha256_file(generation_path)
            pointer["bytes"] = generation_path.stat().st_size
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)):
                report = train_ai42_bc.train(_args(output, "--max-steps", "1"), clock=lambda: 0.0)
            self.assertTrue(report["accepted"])
            self.assertTrue(report["improvement"]["prior_accepted_present"])
            self.assertFalse(report["improvement"]["prior_accepted_compatible"])

    def test_second_alias_promotion_failure_keeps_pointer_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.5), _mock_probe_summary(0.5),
            ))
            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)):
                train_ai42_bc.train(_args(output), clock=lambda: 0.0)
            before = json.loads((output / train_ai42_bc.ACCEPTED_POINTER_FILENAME).read_text(encoding="utf-8"))

            summaries = iter((
                _mock_probe_summary(1.0), _mock_probe_summary(1.0),
                _mock_probe_summary(0.4), _mock_probe_summary(0.4),
            ))
            original_copy = train_ai42_bc._atomic_copy_file
            calls = []

            def fail_second(source, destination):
                calls.append(destination.name)
                if len(calls) == 2:
                    raise OSError("injected second promotion failure")
                return original_copy(source, destination)

            with mock.patch.object(train_ai42_bc, "load_dataset", return_value=_TinyDataset()), \
                 mock.patch.object(train_ai42_bc, "evaluate_probe", side_effect=lambda *_: next(summaries)), \
                 mock.patch.object(train_ai42_bc, "_atomic_copy_file", side_effect=fail_second):
                report = train_ai42_bc.train(_args(output), clock=lambda: 0.0)
            after = json.loads((output / train_ai42_bc.ACCEPTED_POINTER_FILENAME).read_text(encoding="utf-8"))
            self.assertTrue(report["accepted"])
            self.assertEqual(calls, ["accepted.pt", "best.pt"])
            self.assertGreater(after["generation"], before["generation"])
            self.assertEqual(report["promotion"]["compatibility_aliases"]["errors"].keys(), {"best"})
            self.assertEqual(after["sha256"], train_ai42_bc._sha256_file(output / after["checkpoint"]))
            self.assertEqual(after["bytes"], (output / after["checkpoint"]).stat().st_size)

    def test_probe_total_is_weighted_sum_of_head_micro_losses(self) -> None:
        class FakeLearner:
            actor = torch.nn.Linear(1, 1)
            config = SimpleNamespace(head_weights={"control": 2.0, "kind": 3.0})

            def __init__(self):
                self.results = iter((
                    SimpleNamespace(
                        loss=torch.tensor(0.0), head_losses={"control": torch.tensor(0.0)},
                        metrics={"supervised_count": 0, "action_count": 0, "control_count": 0,
                                 "heads": {"control": {"count": 0}}},
                    ),
                    SimpleNamespace(
                        loss=torch.tensor(99.0),
                        head_losses={"control": torch.tensor(1.0), "kind": torch.tensor(10.0)},
                        metrics={"supervised_count": 4, "action_count": 3, "control_count": 1,
                                 "heads": {"control": {"count": 1}, "kind": {"count": 3}}},
                    ),
                    SimpleNamespace(
                        loss=torch.tensor(99.0),
                        head_losses={"control": torch.tensor(3.0), "kind": torch.tensor(2.0)},
                        metrics={"supervised_count": 4, "action_count": 1, "control_count": 3,
                                 "heads": {"control": {"count": 3}, "kind": {"count": 1}}},
                    ),
                ))

            def loss(self, batch):
                return next(self.results)

        result = train_ai42_bc.evaluate_probe(FakeLearner(), (object(), object(), object()))
        self.assertEqual(result.head_denominators, {"control": 4, "kind": 4})
        self.assertAlmostEqual(result.head_losses["control"], 2.5)
        self.assertAlmostEqual(result.head_losses["kind"], 8.0)
        self.assertAlmostEqual(result.loss, 2.0 * 2.5 + 3.0 * 8.0)

        class EmptyLearner(FakeLearner):
            def __init__(self):
                self.results = iter((SimpleNamespace(
                    loss=torch.tensor(0.0), head_losses={"control": torch.tensor(0.0)},
                    metrics={"supervised_count": 0, "heads": {"control": {"count": 0}}},
                ),))

        with self.assertRaisesRegex(train_ai42_bc.AI42TrainingError, "probe is empty"):
            train_ai42_bc.evaluate_probe(EmptyLearner(), (object(),))

    def test_production_batch_default_is_eight(self) -> None:
        self.assertEqual(train_ai42_bc.build_parser().parse_args([]).batch_size, 8)


if __name__ == "__main__":
    unittest.main()
