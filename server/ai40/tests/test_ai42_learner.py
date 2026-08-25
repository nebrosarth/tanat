from __future__ import annotations

import copy
import hashlib
import io
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - dependency-specific CI gate
    raise unittest.SkipTest(f"AI-42 learner tests require torch and numpy: {exc}")

from tanat_ai40.learner_ai42 import (
    AI42Batch,
    AI42Learner,
    AI42LearnerConfig,
    AI42LearnerError,
    CheckpointError,
    NonFiniteError,
    TeacherStatus,
    build_learner_manifest,
    class_balance_weights,
    forward_batch,
    iter_ai42_dataset_batches,
    load_ai42_checkpoint,
    save_ai42_checkpoint,
)
from tanat_ai40.env import ACTION_DTYPE
from tanat_ai40.model_ai42_actor import AI42Actor
import tanat_ai40.train_ai42_bc as train_ai42_bc
from tanat_ai40.train_ai42_bc import main


class AI42LearnerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.actor = AI42Actor(
            hidden_size=8,
            model_width=8,
            entity_layers=1,
            num_heads=2,
            ff_multiplier=1,
            timing_bins=2,
        )

    def batch(self, *, death: bool = True, invalid_target: bool = False) -> AI42Batch:
        batch_size, sequence_length = 2, 4
        torch.manual_seed(11)
        hero = torch.randn(batch_size, sequence_length, 32)
        abilities = torch.randn(batch_size, sequence_length, 4, 40)
        entities = torch.randn(batch_size, sequence_length, 96, 16)
        global_state = torch.randn(batch_size, sequence_length, 32)
        entity_mask = torch.zeros(batch_size, sequence_length, 96, dtype=torch.bool)
        entity_mask[..., :4] = True
        actions = torch.tensor([
            [[1, 0, 2, 1], [2, 1, 0, 0], [3, 2, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [4, 2, 5, 0], [7, 0, 0, 0], [1, 0, 1, 2]],
        ])
        statuses = torch.tensor([
            [TeacherStatus.ACTION, TeacherStatus.ACTION, TeacherStatus.ACTION, TeacherStatus.WAIT],
            [TeacherStatus.HOLD, TeacherStatus.ACTION, TeacherStatus.CANCEL, TeacherStatus.NONE],
        ])
        kind_mask = torch.ones(batch_size, sequence_length, 8, dtype=torch.bool)
        target_mask = entity_mask.clone()
        skill_target_mask = entity_mask.unsqueeze(2).expand(batch_size, sequence_length, 4, 96).clone()
        if invalid_target:
            target_mask[0, 1, 1] = False
        padding_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        padding_mask[1, 3] = True
        death_mask = torch.zeros_like(padding_mask)
        if death:
            death_mask[0, 2] = True
        return AI42Batch(
            hero, abilities, entities, global_state, entity_mask,
            teacher_actions=actions,
            teacher_status=statuses,
            kind_mask=kind_mask,
            target_mask=target_mask,
            skill_target_mask=skill_target_mask,
            padding_mask=padding_mask,
            death_mask=death_mask,
            time_indices=torch.arange(sequence_length).expand(batch_size, -1),
        )

    def test_recurrent_boundaries_and_padding_are_applied(self) -> None:
        batch = self.batch()
        output = forward_batch(self.actor, batch)
        self.assertEqual(tuple(output["kind"].shape), (2, 4, 8))
        self.assertEqual(tuple(output["target"].shape), (2, 4, 8, 96))

        fresh = AI42Batch(
            batch.hero[:1, 2:3], batch.abilities[:1, 2:3], batch.entities[:1, 2:3],
            batch.global_state[:1, 2:3], batch.entity_mask[:1, 2:3],
            teacher_actions=batch.teacher_actions[:1, 2:3],
            teacher_status=batch.teacher_status[:1, 2:3],
            kind_mask=batch.kind_mask[:1, 2:3],
            target_mask=batch.target_mask[:1, 2:3],
            skill_target_mask=batch.skill_target_mask[:1, 2:3],
        )
        fresh_output = forward_batch(self.actor, fresh)
        torch.testing.assert_close(output["kind"][:1, 2], fresh_output["kind"][:, 0], atol=0, rtol=0)
        torch.testing.assert_close(output["h"][:1, 2], fresh_output["h"][:, 0], atol=0, rtol=0)
        # A padded row cannot advance the carried recurrent state.
        torch.testing.assert_close(output["final_h"][1], output["h"][1, 2], atol=0, rtol=0)

    def test_status_masks_heads_and_backward_without_optimizer_step(self) -> None:
        learner = AI42Learner(self.actor, AI42LearnerConfig(max_gradient_norm=0.5))
        before = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        result = learner.backward(self.batch())
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(result.metrics["action_count"], 4)
        self.assertEqual(result.metrics["control_count"], 7)
        self.assertEqual(result.metrics["action_parameter_count"], 4)
        self.assertEqual(result.metrics["supervised_count"], 7)
        self.assertEqual(result.control_counts, {"issue": 4, "wait": 1, "hold": 1, "cancel": 1})
        self.assertTrue(result.metrics["timing"]["excluded"])
        self.assertEqual(set(result.class_counts), {"control", "kind", "target", "offset", "anchor"})
        self.assertEqual(set(result.skill_metrics), {"1", "2", "3", "4"})
        learner.clip_gradients()
        for name, value in self.actor.state_dict().items():
            torch.testing.assert_close(value, before[name])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA batch-device gate")
    def test_batch_to_cuda_moves_every_tensor_field(self) -> None:
        moved = self.batch().to("cuda")
        for field_name in moved.__dataclass_fields__:
            value = getattr(moved, field_name)
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cuda", field_name)

    def test_action_mask_rejection_and_non_finite_input_fail_closed(self) -> None:
        with self.assertRaises(AI42LearnerError):
            AI42Learner(self.actor).loss(self.batch(invalid_target=True))
        bad = self.batch()
        bad_hero = bad.hero.clone()
        bad_hero[0, 0, 0] = float("nan")
        with self.assertRaises(NonFiniteError):
            AI42Batch(
                bad_hero, bad.abilities, bad.entities, bad.global_state, bad.entity_mask,
                teacher_actions=bad.teacher_actions, teacher_status=bad.teacher_status,
                kind_mask=bad.kind_mask, target_mask=bad.target_mask,
                skill_target_mask=bad.skill_target_mask,
            )
        with self.assertRaisesRegex(AI42LearnerError, "integer labels"):
            AI42Batch(
                bad.hero, bad.abilities, bad.entities, bad.global_state, bad.entity_mask,
                teacher_actions=bad.teacher_actions.float(), teacher_status=bad.teacher_status,
                kind_mask=bad.kind_mask, target_mask=bad.target_mask,
                skill_target_mask=bad.skill_target_mask,
            )
        with self.assertRaisesRegex(AI42LearnerError, "bool or uint8"):
            AI42Batch(
                bad.hero, bad.abilities, bad.entities, bad.global_state, bad.entity_mask,
                teacher_actions=bad.teacher_actions, teacher_status=bad.teacher_status,
                kind_mask=bad.kind_mask.float(), target_mask=bad.target_mask,
                skill_target_mask=bad.skill_target_mask,
            )

    def test_class_balance_and_protocol_adapter(self) -> None:
        weights = class_balance_weights((8, 2, 0), power=1.0)
        self.assertAlmostEqual(float(weights[0]), 0.4, places=5)
        self.assertAlmostEqual(float(weights[1]), 1.6, places=5)
        payload = {
            "observations": {
                "hero": self.batch().hero,
                "abilities": self.batch().abilities,
                "entities": self.batch().entities,
                "global": self.batch().global_state,
            },
            "masks": {
                "entity_mask": self.batch().entity_mask,
                "kind_mask": self.batch().kind_mask,
                "target_mask": self.batch().target_mask,
                "skill_target_mask": self.batch().skill_target_mask,
            },
            "labels": {"actions": self.batch().teacher_actions, "status": self.batch().teacher_status},
        }
        adapted = AI42Batch.from_mapping(payload)
        self.assertEqual((adapted.batch_size, adapted.sequence_length), (2, 4))

    def test_durable_dataset_windows_never_cross_match_or_death_boundary(self) -> None:
        ticks, heroes = 3, 10
        arrays = {
            "hero": np.zeros((ticks, heroes, 32), dtype="<f4"),
            "abilities": np.zeros((ticks, heroes, 4, 40), dtype="<f4"),
            "entities": np.zeros((ticks, heroes, 96, 16), dtype="<f4"),
            "global": np.zeros((ticks, heroes, 32), dtype="<f4"),
            "entity_mask": np.zeros((ticks, heroes, 96), dtype="u1"),
            "kind_mask": np.ones((ticks, heroes, 8), dtype="u1"),
            "target_mask": np.zeros((ticks, heroes, 96), dtype="u1"),
            "skill_target_mask": np.zeros((ticks, heroes, 4, 96), dtype="u1"),
            "teacher_status": np.full((ticks, heroes), TeacherStatus.WAIT, dtype="u1"),
            "teacher_action": np.zeros((ticks, heroes), dtype=ACTION_DTYPE),
            "step": np.arange(7, 10, dtype="<u4"),
            "done": np.asarray([0, 0, 1], dtype="u1"),
        }
        arrays["hero"][:, :, 0] = np.arange(heroes, dtype="<f4") / 100
        arrays["hero"][1, 0, 9] = 1

        class FakeDataset:
            def iter_matches(self, split: str):
                self.split = split
                yield "match-a", arrays

        batches = list(iter_ai42_dataset_batches(
            FakeDataset(), split="train", sequence_length=2, batch_size=20,
        ))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual((batch.batch_size, batch.sequence_length), (20, 2))
        self.assertTrue(batch.reset_mask[:, 0].all())
        self.assertTrue(batch.padding_mask[1::2, 1].all())
        self.assertTrue(batch.death_mask[0, 1])
        self.assertTrue(batch.death_mask[1, 0])
        self.assertTrue(torch.equal(batch.time_indices[0], torch.tensor([7, 8])))
        self.assertTrue(torch.equal(batch.time_indices[1], torch.tensor([9, 10])))

    def test_manifest_exact_atomic_checkpoint_and_rng_restore(self) -> None:
        config = AI42LearnerConfig(model_kwargs={"hidden_size": 8, "model_width": 8, "entity_layers": 1, "num_heads": 2, "ff_multiplier": 1, "timing_bins": 2})
        optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        manifest = build_learner_manifest(self.actor, config, hashlib.sha256(b"dataset").hexdigest())
        before = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learner.pt"
            save_ai42_checkpoint(path, self.actor, optimizer, manifest, step=4, epoch=2)
            self.assertTrue(path.is_file())
            resumed = load_ai42_checkpoint(path, self.actor, optimizer, manifest)
            self.assertEqual((resumed.step, resumed.epoch), (4, 2))
            for name, value in self.actor.state_dict().items():
                torch.testing.assert_close(value, before[name])
            with self.assertRaises(CheckpointError):
                load_ai42_checkpoint(path, self.actor, optimizer, dict(manifest, dataset_hash="different"))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            first_key = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][first_key] = payload["model_state_dict"][first_key] + 1
            torch.save(payload, path)
            with self.assertRaisesRegex(CheckpointError, "artifact digest"):
                load_ai42_checkpoint(path, self.actor, optimizer, manifest)

    def test_cli_requires_execute_and_directs_to_native_preflight(self) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            self.assertEqual(main([]), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("requires --execute", error.getvalue())
        self.assertIn("run-ai42-bc-preflight", error.getvalue())
        self.assertNotIn("preflight", train_ai42_bc.__all__)
        self.assertFalse(hasattr(train_ai42_bc, "preflight"))
        self.assertEqual(main(["--execute"]), 2)


if __name__ == "__main__":
    unittest.main()
