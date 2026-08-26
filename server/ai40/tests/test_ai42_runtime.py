from __future__ import annotations

import inspect
import hashlib
import unittest
from unittest import mock

import torch

from tanat_ai40.env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
)
from tanat_ai40.model_ai42_actor import (
    AI42Actor,
    CONTROL_CANCEL,
    CONTROL_HOLD,
    CONTROL_ISSUE,
    CONTROL_WAIT,
)
from tanat_ai40.runtime_ai42 import (
    AI42InferenceAdapter,
    AI42Manifest,
    ActionMaskError,
    CheckpointCompatibilityError,
    HeroLifecycle,
    RecurrentStateStore,
    checkpoint_compatibility_report,
    load_ai42_checkpoint,
    load_compatible_checkpoint,
    manifest_mismatches,
)


def manifest(seed: str = "a") -> AI42Manifest:
    def digest(name: str) -> str:
        return hashlib.sha256(f"{name}-{seed}".encode()).hexdigest()

    return AI42Manifest(
        model_hash=digest("model"),
        config_hash=digest("config"),
        checkpoint_hash=digest("checkpoint"),
        observation_hash=digest("observation"),
        action_hash=digest("action"),
        trajectory_hash=digest("trajectory"),
    )


class AI42RuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.actor = AI42Actor(
            hidden_size=8,
            model_width=8,
            entity_layers=1,
            num_heads=2,
            ff_multiplier=1,
        ).eval()
        with torch.no_grad():
            self.actor.control_head.boundary.weight.zero_()
            self.actor.control_head.boundary.bias.fill_(-100.0)
            self.actor.control_head.boundary.bias[CONTROL_ISSUE] = 100.0

    @staticmethod
    def observation(entity_slots: int = 5) -> dict[str, torch.Tensor]:
        torch.manual_seed(17)
        return {
            "hero": torch.randn(HERO_FEATURES),
            "abilities": torch.randn(ABILITY_COUNT, ABILITY_FEATURES),
            "entities": torch.randn(entity_slots, ENTITY_FEATURES),
            "global_state": torch.randn(GLOBAL_FEATURES),
            "entity_mask": torch.tensor([True, True, False, False, False])[:entity_slots],
        }

    @staticmethod
    def masks(entity_slots: int = 5) -> dict[str, torch.Tensor]:
        return {
            "kind": torch.tensor([False, False, True, False, False, False, False, False]),
            "target": torch.tensor([[False, True] + [False] * (entity_slots - 2)] * 8),
            "offset": torch.tensor([[False] * 5 + [True] + [False] * 75] * 8),
            "anchor": torch.tensor([[False, False, False, True] + [False] * 11] * 8),
        }

    def test_runtime_import_is_actor_only_and_has_no_live_profile_wiring(self) -> None:
        import tanat_ai40.runtime_ai42 as runtime

        source = inspect.getsource(runtime)
        self.assertNotIn("critic_ai42", source)
        self.assertNotIn("full_state", source)
        self.assertFalse(runtime.LIVE_PROFILE_WIRING)

    def test_manifest_is_immutable_and_canonical(self) -> None:
        first = manifest()
        reordered = AI42Manifest.from_dict({key: first.to_dict()[key] for key in reversed(first.to_dict())})
        self.assertEqual(first, reordered)
        self.assertEqual(first.canonical_json(), reordered.canonical_json())
        self.assertEqual(first.canonical_bytes(), first.canonical_json().encode())
        with self.assertRaises(AttributeError):
            first.model_hash = "changed"  # type: ignore[misc]
        differences = manifest_mismatches(first, manifest("b"))
        self.assertEqual(set(differences), {
            "model_hash", "config_hash", "checkpoint_hash", "observation_hash",
            "action_hash", "trajectory_hash",
        })

    def test_manifest_decode_requires_complete_strict_canonical_fields(self) -> None:
        valid = manifest().to_dict()
        missing = dict(valid)
        missing.pop("trajectory_hash")
        with self.assertRaisesRegex(ValueError, "missing"):
            AI42Manifest.from_dict(missing)
        unknown = dict(valid, extra="forbidden")
        with self.assertRaisesRegex(ValueError, "unknown"):
            AI42Manifest.from_dict(unknown)
        for bad_hash in ("0" * 63, "A" * 64, "g" * 64):
            malformed = dict(valid, model_hash=bad_hash)
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                AI42Manifest.from_dict(malformed)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            AI42Manifest.from_dict(dict(valid, model_version="   "))

    def test_recurrent_store_has_ten_stable_slots_and_lifecycle_resets(self) -> None:
        store = RecurrentStateStore(4)
        self.assertEqual(store.slot_count, HERO_COUNT)
        store.set(7, torch.ones(4), torch.full((4,), 2.0))
        store.on_death(7)
        torch.testing.assert_close(store.get(7)[0], torch.zeros(4))
        store.set(7, torch.ones(4), torch.ones(4))
        store.on_respawn(7)
        torch.testing.assert_close(store.get(7)[0], torch.zeros(4))
        store.set(0, torch.ones(4), torch.ones(4))
        store.apply_event(None, HeroLifecycle.MATCH_RESET)
        self.assertTrue(bool(torch.equal(store.get(0)[0], torch.zeros(4))))
        with self.assertRaises(IndexError):
            store.get(HERO_COUNT)

    def test_recurrent_snapshot_restore_validates_shape_and_device(self) -> None:
        store = RecurrentStateStore(4)
        store.set(2, torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32) + 1)
        snapshot = store.snapshot()
        store.reset_match()
        store.restore(snapshot)
        torch.testing.assert_close(store.get(2)[0], torch.arange(4, dtype=torch.float32))
        invalid = dict(snapshot.to_dict())
        invalid["h"] = torch.zeros(HERO_COUNT - 1, 4)
        with self.assertRaises(ValueError):
            store.restore(invalid)
        invalid = dict(snapshot.to_dict())
        invalid["device"] = "cuda"
        with self.assertRaises(ValueError):
            store.restore(invalid)

    def test_masks_are_enforced_and_invalid_outputs_fallback_with_telemetry(self) -> None:
        adapter = AI42InferenceAdapter(self.actor)
        masks = self.masks()
        result = adapter.infer(0, self.observation(), masks)
        self.assertFalse(result.used_fallback)
        self.assertEqual(dict(result.action)["kind"], 2)
        self.assertEqual(dict(result.action)["target"], 1)
        self.assertEqual(dict(result.action)["offset"], 5)
        self.assertEqual(dict(result.action)["anchor"], 3)

        bad = self.observation()
        bad["hero"][0] = float("nan")
        fallback = adapter.infer(1, bad)
        self.assertTrue(fallback.used_fallback)
        self.assertEqual(fallback.reason, "schema_mismatch")
        self.assertEqual(adapter.telemetry_snapshot()["fallback_reasons"]["schema_mismatch"], 1)

        with mock.patch.object(self.actor, "forward", side_effect=RuntimeError("boom")):
            fallback = adapter.infer(2, self.observation(), self.masks())
        self.assertEqual(fallback.reason, "exception")
        self.assertEqual(adapter.telemetry_snapshot()["calls"], 3)

    def test_control_is_selected_before_parameters_and_non_issue_is_explicit(self) -> None:
        original_forward = self.actor.forward
        for control, name in (
            (CONTROL_ISSUE, "issue"),
            (CONTROL_WAIT, "wait"),
            (CONTROL_HOLD, "hold"),
            (CONTROL_CANCEL, "cancel"),
        ):
            with self.subTest(control=name):
                adapter = AI42InferenceAdapter(self.actor)

                def forced_forward(*args, _control=control, **kwargs):
                    output = original_forward(*args, **kwargs)
                    control_logits = output["control"].new_full(output["control"].shape, -10.0)
                    control_logits[:, _control] = 10.0
                    output["control"] = control_logits
                    return output

                with mock.patch.object(self.actor, "forward", side_effect=forced_forward):
                    result = adapter.infer(0, self.observation(), self.masks())

                self.assertFalse(result.used_fallback)
                self.assertEqual(result["control"], control)
                self.assertEqual(result["control_name"], name)
                self.assertTrue(result["valid"])
                self.assertEqual(result["issued"], control == CONTROL_ISSUE)
                self.assertIsNotNone(result.raw_output)
                torch.testing.assert_close(
                    adapter.state_store.get(0)[0], result.raw_output["h"][0],  # type: ignore[index]
                )
                if control == CONTROL_ISSUE:
                    self.assertEqual(result["kind"], 2)
                    self.assertEqual(result["target"], 1)
                    self.assertEqual(result["offset"], 5)
                    self.assertEqual(result["anchor"], 3)
                else:
                    for field in (
                        "kind", "target", "offset", "anchor",
                    ):
                        self.assertEqual(result[field], -1)

    def test_malformed_control_output_falls_back_without_advancing_state(self) -> None:
        original_forward = self.actor.forward
        for malformed in ("shape", "non_finite"):
            with self.subTest(malformed=malformed):
                adapter = AI42InferenceAdapter(self.actor)
                before = adapter.snapshot_state()
                masks = self.masks()
                masks["kind"] = torch.zeros_like(masks["kind"])

                def malformed_forward(*args, _malformed=malformed, **kwargs):
                    output = original_forward(*args, **kwargs)
                    control_logits = output["control"].clone()
                    if _malformed == "shape":
                        output["control"] = control_logits[:, :3]
                    else:
                        control_logits[0, 0] = float("nan")
                        output["control"] = control_logits
                    return output

                with mock.patch.object(self.actor, "forward", side_effect=malformed_forward):
                    result = adapter.infer(0, self.observation(), masks)

                self.assertTrue(result.used_fallback)
                self.assertEqual(result.reason, "schema_mismatch")
                self.assertFalse(result["valid"])
                self.assertFalse(result["issued"])
                torch.testing.assert_close(adapter.snapshot_state().h, before.h)
                torch.testing.assert_close(adapter.snapshot_state().c, before.c)

    def test_masks_fail_closed_and_target_always_intersects_entity_visibility(self) -> None:
        adapter = AI42InferenceAdapter(self.actor)
        with mock.patch.object(self.actor, "forward", wraps=self.actor.forward) as forward:
            missing = adapter.infer(0, self.observation(), {"kind": self.masks()["kind"]})
        forward.assert_not_called()
        self.assertTrue(missing.used_fallback)
        self.assertEqual(missing.reason, "mask_invalid")
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["kind"], -1)
        self.assertEqual(missing["target"], -1)
        self.assertEqual(missing["offset"], -1)

        hidden_target_masks = self.masks()
        hidden_target_masks["target"] = torch.tensor(
            [[False, False, False, False, True]] * 8,
        )
        hidden = adapter.infer(1, self.observation(), hidden_target_masks)
        self.assertTrue(hidden.used_fallback)
        self.assertEqual(hidden.reason, "mask_invalid")
        self.assertEqual(hidden["target"], -1)
        self.assertFalse(hidden["valid"])

    def test_latency_telemetry_is_bounded_and_counts_fallbacks(self) -> None:
        readings = iter((10.0, 10.1, 20.0, 20.3, 30.0, 30.2))
        adapter = AI42InferenceAdapter(
            self.actor,
            clock=lambda: next(readings),
            latency_sample_size=2,
            latency_threshold_seconds=0.15,
        )
        adapter.infer(0, self.observation(), self.masks())
        adapter.infer(1, self.observation(), {})
        adapter.infer(2, self.observation(), self.masks())
        telemetry = adapter.telemetry_snapshot()
        self.assertEqual(telemetry["calls"], 3)
        self.assertEqual(telemetry["fallbacks"], 1)
        self.assertAlmostEqual(telemetry["last_latency_seconds"], 0.2)
        self.assertAlmostEqual(telemetry["total_latency_seconds"], 0.6)
        self.assertAlmostEqual(telemetry["max_latency_seconds"], 0.3)
        self.assertEqual(telemetry["latency_sample_count"], 2)
        self.assertEqual(telemetry["latency_sample_capacity"], 2)
        self.assertEqual(telemetry["latency_threshold_exceeded"], 2)

    def test_manifest_mismatch_falls_back_without_calling_actor(self) -> None:
        adapter = AI42InferenceAdapter(
            self.actor,
            expected_manifest=manifest("expected"),
            model_manifest=manifest("actual"),
        )
        with mock.patch.object(self.actor, "forward", side_effect=AssertionError("must not run")):
            result = adapter.infer(0, self.observation())
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "manifest_mismatch")

    def test_exact_checkpoint_load_and_no_partial_load(self) -> None:
        expected = manifest()
        state = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        checkpoint = {"manifest": expected.to_dict(), "state_dict": state}
        report = checkpoint_compatibility_report(self.actor, checkpoint, expected)
        self.assertTrue(report.compatible)
        self.assertEqual(set(report.loaded), set(state))
        exact = load_compatible_checkpoint(self.actor, checkpoint, expected)
        self.assertTrue(exact.exact)

        before = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        partial = dict(state)
        partial.pop(next(iter(partial)))
        bad = {"manifest": expected.to_dict(), "state_dict": partial}
        rejected = load_ai42_checkpoint(self.actor, bad, expected)
        self.assertFalse(rejected.compatible)
        self.assertTrue(rejected.mismatched)
        for name, value in self.actor.state_dict().items():
            torch.testing.assert_close(value, before[name])
        with self.assertRaises(CheckpointCompatibilityError):
            load_ai42_checkpoint(self.actor, bad, expected, strict=True)

    def test_non_finite_checkpoint_is_rejected_before_any_load(self) -> None:
        expected = manifest()
        before = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        floating_name = next(name for name, value in before.items() if value.is_floating_point())
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(non_finite=non_finite):
                state = {name: value.detach().clone() for name, value in before.items()}
                state[floating_name].reshape(-1)[0] = non_finite
                checkpoint = {"manifest": expected.to_dict(), "state_dict": state}
                with mock.patch.object(
                    self.actor, "load_state_dict", wraps=self.actor.load_state_dict,
                ) as load:
                    report = load_ai42_checkpoint(self.actor, checkpoint, expected)
                self.assertFalse(report.compatible)
                self.assertIn(f"state_dict:{floating_name}:non_finite", report.mismatched)
                load.assert_not_called()
                for name, value in self.actor.state_dict().items():
                    torch.testing.assert_close(value, before[name])

    def test_injected_state_store_must_match_complete_actor_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden size"):
            AI42InferenceAdapter(self.actor, state_store=RecurrentStateStore(7))
        with self.assertRaisesRegex(ValueError, "dtype"):
            AI42InferenceAdapter(
                self.actor,
                state_store=RecurrentStateStore(8, dtype=torch.float64),
            )
        wrong_device = RecurrentStateStore(8)
        wrong_device.device = torch.device("meta")
        with self.assertRaisesRegex(ValueError, "device"):
            AI42InferenceAdapter(self.actor, state_store=wrong_device)
        wrong_slots = RecurrentStateStore(8)
        wrong_slots.slot_count = HERO_COUNT - 1
        with self.assertRaisesRegex(ValueError, "exactly"):
            AI42InferenceAdapter(self.actor, state_store=wrong_slots)

    def test_migration_requires_explicit_report_and_complete_state(self) -> None:
        expected = manifest("new")
        old = manifest("old")
        state = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        checkpoint = {"manifest": old.to_dict(), "state_dict": state}
        rejected = load_ai42_checkpoint(self.actor, checkpoint, expected)
        self.assertFalse(rejected.compatible)
        self.assertFalse(rejected.migrated)

        migrated = load_ai42_checkpoint(
            self.actor,
            checkpoint,
            expected,
            migration=lambda source, _old, _new: dict(source),
        )
        self.assertTrue(migrated.compatible)
        self.assertTrue(migrated.migrated)
        self.assertIn("manifest:model_hash", migrated.mismatched)
        self.assertEqual(migrated.skipped, ())


if __name__ == "__main__":
    unittest.main()
