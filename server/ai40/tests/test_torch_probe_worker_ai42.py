from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency-specific CI gate
    raise unittest.SkipTest(f"AI-42 Torch probe worker tests require torch: {exc}")

from tanat_ai40.learner_ai42 import AI42LearnerConfig, build_learner_manifest, save_ai42_checkpoint
from tanat_ai40.model_ai42_actor import AI42Actor
from tanat_ai40 import torch_probe_worker_ai42 as worker
from tanat_ai40.torch_probe_worker_ai42 import PROTOCOL, process_request
from tanat_ai40.trajectory_ai42 import canonical_json_bytes


MODEL = {
    "hidden_size": 8,
    "model_width": 8,
    "entity_layers": 1,
    "num_heads": 2,
    "ff_multiplier": 1,
}
LEARNER = {
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "class_balance_power": 1.0,
    "offset_coordinate_loss_weight": 0.5,
    "max_gradient_norm": 1.0,
}


def _batch() -> dict[str, object]:
    batch, steps, entities = 1, 2, 96
    return {
        "observations": {
            "hero": [[[0.0] * 32 for _ in range(steps)]],
            "abilities": [[[[0.0] * 40 for _ in range(4)] for _ in range(steps)]],
            "entities": [[[[0.0] * 16 for _ in range(entities)] for _ in range(steps)]],
            "global_state": [[[0.0] * 32 for _ in range(steps)]],
        },
        "masks": {
            "entity_mask": [[[True] * 4 + [False] * (entities - 4) for _ in range(steps)]],
            "kind_mask": [[[True] * 8 for _ in range(steps)]],
            "target_mask": [[[True] * 4 + [False] * (entities - 4) for _ in range(steps)]],
            "skill_target_mask": [[[[True] * 4 + [False] * (entities - 4) for _ in range(4)] for _ in range(steps)]],
        },
        "labels": {
            "teacher_actions": [[ [1, 0, 0, 0] for _ in range(steps) ]],
            "teacher_status": [[1 for _ in range(steps)]],
        },
        "reset_mask": [[True, False]],
        "time_indices": [[0, 1]],
    }


def _request(checkpoint: Path, checkpoint_hash: str, batch: dict[str, object]) -> dict[str, object]:
    batch_bytes = canonical_json_bytes(batch)
    request: dict[str, object] = {
        "protocol": PROTOCOL,
        "request_sha256": "0" * 64,
        "seed": 4242,
        "device": "cpu",
        "model": MODEL,
        "learner": LEARNER,
        "warm_start": {
            "path": str(checkpoint), "sha256": checkpoint_hash,
            "dataset_hash": "a" * 64, "allow_dataset_change": False,
        },
        "batch": {"kind": "inline", "sha256": hashlib.sha256(batch_bytes).hexdigest(), "value": batch},
    }
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    request["request_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return request


def _rebind(request: dict[str, object]) -> None:
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    request["request_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


class TorchAI42ProbeWorkerTests(unittest.TestCase):
    def _checkpoint(self, directory: Path, *, name: str = "checkpoint.pt", seed: int = 7) -> tuple[Path, str]:
        torch.manual_seed(seed)
        actor = AI42Actor(**MODEL)
        config = AI42LearnerConfig(model_kwargs=MODEL)
        manifest = build_learner_manifest(actor, config, "a" * 64, protocol_version=13)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)
        path = directory / name
        save_ai42_checkpoint(path, actor, optimizer, manifest, step=9, epoch=2, extra={"batch_cursor": 17})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def test_worker_runs_masked_recurrent_probe_without_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint, checkpoint_hash = self._checkpoint(Path(directory_name))
            request = _request(checkpoint, checkpoint_hash, _batch())
            response = json.loads(process_request(canonical_json_bytes(request)))

        self.assertTrue(response["ok"])
        result = response["result"]
        self.assertTrue(all(
            value for name, value in result["invariants"].items()
            if name not in {"optimizer_step_called", "optimizer_authorized", "final_report_published"}
        ))
        self.assertFalse(result["invariants"]["optimizer_step_called"])
        self.assertFalse(result["invariants"]["optimizer_authorized"])
        self.assertEqual(result["batch"]["supervised_count"], 2)
        self.assertEqual(result["warm_start"]["source_step"], 9)
        self.assertEqual(result["warm_start"]["source_cursor"], 17)
        self.assertEqual(result["hashes"]["model_after_warm_start"], result["hashes"]["model_after"])

    def test_worker_requires_explicit_dataset_change_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint, checkpoint_hash = self._checkpoint(Path(directory_name))
            denied = _request(checkpoint, checkpoint_hash, _batch())
            denied["warm_start"]["dataset_hash"] = "b" * 64
            _rebind(denied)
            denied_response = json.loads(process_request(canonical_json_bytes(denied)))
            self.assertFalse(denied_response["ok"])

            allowed = copy.deepcopy(denied)
            allowed["warm_start"]["allow_dataset_change"] = True
            _rebind(allowed)
            allowed_response = json.loads(process_request(canonical_json_bytes(allowed)))

        self.assertTrue(allowed_response["ok"])
        self.assertTrue(allowed_response["result"]["warm_start"]["dataset_changed"])
        self.assertTrue(allowed_response["result"]["warm_start"]["dataset_change_allowed"])
        self.assertTrue(allowed_response["result"]["invariants"]["optimizer_not_restored"])
        self.assertTrue(allowed_response["result"]["invariants"]["rng_not_restored"])

    def test_worker_rejects_noncanonical_unknown_and_hash_bound_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint, checkpoint_hash = self._checkpoint(Path(directory_name))
            request = _request(checkpoint, checkpoint_hash, _batch())
            canonical = canonical_json_bytes(request)

            noncanonical = json.dumps(request, indent=2).encode("utf-8")
            response = json.loads(process_request(noncanonical))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "noncanonical_json")

            unknown = dict(request)
            unknown["unexpected"] = True
            response = json.loads(process_request(canonical_json_bytes(unknown)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "schema_error")

            tampered = dict(request)
            tampered["request_sha256"] = "f" * 64
            response = json.loads(process_request(canonical_json_bytes(tampered)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "hash_mismatch")

            bundle_path = Path(directory_name) / "batch.json"
            bundle_bytes = canonical_json_bytes(_batch())
            bundle_path.write_bytes(bundle_bytes)
            bundle_request = _request(checkpoint, checkpoint_hash, _batch())
            bundle_request["batch"] = {
                "kind": "bundle",
                "path": str(bundle_path),
                "sha256": "0" * 64,
            }
            _rebind(bundle_request)
            response = json.loads(process_request(canonical_json_bytes(bundle_request)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "hash_mismatch")

    def test_worker_rejects_nonfinite_json_before_model_work(self) -> None:
        response = json.loads(process_request(b'{"value":NaN}'))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_json")

    def test_dataset_batch_kind_is_rejected_before_worker_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint, checkpoint_hash = self._checkpoint(Path(directory_name))
            request = _request(checkpoint, checkpoint_hash, _batch())
            request["batch"] = {
                "kind": "dataset",
                "path": "must-not-be-read",
                "manifest_sha256": "0" * 64,
                "split": "train",
                "sequence_length": 1,
                "batch_size": 1,
            }
            _rebind(request)
            with mock.patch.object(worker, "_stage_verified_checkpoint") as stage, \
                 mock.patch.object(worker, "_load_batch") as load_batch, \
                 mock.patch.object(worker, "AI42Actor") as actor_constructor:
                response = json.loads(process_request(canonical_json_bytes(request)))

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "schema_error")
        stage.assert_not_called()
        load_batch.assert_not_called()
        actor_constructor.assert_not_called()

    def test_checkpoint_swap_uses_staged_bytes_and_fails_source_reverification(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            checkpoint_a, hash_a = self._checkpoint(root, name="accepted-a.pt", seed=7)
            checkpoint_b, _ = self._checkpoint(root, name="accepted-b.pt", seed=8)
            request = _request(checkpoint_a, hash_a, _batch())
            real_inspect = worker.inspect_ai42_checkpoint
            swapped = False

            def swap_after_stage(path, *args, **kwargs):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    checkpoint_a.write_bytes(checkpoint_b.read_bytes())
                return real_inspect(path, *args, **kwargs)

            with mock.patch.object(worker, "inspect_ai42_checkpoint", side_effect=swap_after_stage):
                response = json.loads(process_request(canonical_json_bytes(request)))

        self.assertTrue(swapped)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "source_changed")

    def test_deterministic_debug_warn_mode_is_restored_on_success_and_failure(self) -> None:
        original_mode = torch.get_deterministic_debug_mode()
        try:
            torch.set_deterministic_debug_mode(1)
            with tempfile.TemporaryDirectory() as directory_name:
                checkpoint, checkpoint_hash = self._checkpoint(Path(directory_name))
                request = _request(checkpoint, checkpoint_hash, _batch())
                success = json.loads(process_request(canonical_json_bytes(request)))
                self.assertTrue(success["ok"])
                self.assertEqual(torch.get_deterministic_debug_mode(), 1)

                failed_request = copy.deepcopy(request)
                failed_request["warm_start"]["sha256"] = "0" * 64
                _rebind(failed_request)
                failure = json.loads(process_request(canonical_json_bytes(failed_request)))
                self.assertFalse(failure["ok"])
                self.assertEqual(torch.get_deterministic_debug_mode(), 1)
        finally:
            torch.set_deterministic_debug_mode(original_mode)

    def test_hard_bounds_fail_before_model_or_batch_tensor_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            checkpoint, checkpoint_hash = self._checkpoint(root)
            base = _request(checkpoint, checkpoint_hash, _batch())

            with mock.patch.object(worker, "MAX_CHECKPOINT_BYTES", checkpoint.stat().st_size - 1), \
                 mock.patch.object(worker, "AI42Actor") as actor_constructor:
                response = json.loads(process_request(canonical_json_bytes(base)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "size_limit")
            actor_constructor.assert_not_called()

            bundle_path = root / "batch.json"
            bundle_bytes = canonical_json_bytes(_batch())
            bundle_path.write_bytes(bundle_bytes)
            bundle_request = copy.deepcopy(base)
            bundle_request["batch"] = {
                "kind": "bundle",
                "path": str(bundle_path),
                "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            }
            _rebind(bundle_request)
            with mock.patch.object(worker, "MAX_BUNDLE_BYTES", len(bundle_bytes) - 1), \
                 mock.patch.object(worker, "AI42Actor") as actor_constructor:
                response = json.loads(process_request(canonical_json_bytes(bundle_request)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "size_limit")
            actor_constructor.assert_not_called()

            model_request = copy.deepcopy(base)
            model_request["model"] = {
                "hidden_size": 4096,
                "model_width": 4096,
                "entity_layers": 4096,
                "num_heads": 8,
                "ff_multiplier": 4096,
            }
            _rebind(model_request)
            with mock.patch.object(worker, "AI42Actor") as actor_constructor:
                response = json.loads(process_request(canonical_json_bytes(model_request)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "bounds_error")
            actor_constructor.assert_not_called()

            batch_request = copy.deepcopy(base)
            oversized_batch = _batch()
            oversized_batch["observations"]["hero"] = [
                oversized_batch["observations"]["hero"][0]
                for _ in range(worker.MAX_BATCH_SIZE + 1)
            ]
            batch_request["batch"] = {
                "kind": "inline",
                "sha256": hashlib.sha256(canonical_json_bytes(oversized_batch)).hexdigest(),
                "value": oversized_batch,
            }
            _rebind(batch_request)
            with mock.patch.object(worker, "AI42Actor") as actor_constructor, \
                 mock.patch.object(worker.AI42Batch, "from_mapping") as batch_constructor:
                response = json.loads(process_request(canonical_json_bytes(batch_request)))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "bounds_error")
            actor_constructor.assert_not_called()
            batch_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
