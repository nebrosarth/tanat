from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tanat_ai40.cycle_ai42_ppo import run_cycle
from tanat_ai40.ppo_ai42 import AI42PPOConfig


class AI42CycleTests(unittest.TestCase):
    def test_cycle_recommends_but_never_applies_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.pt"
            candidate.touch()
            training = {"final_checkpoint": str(candidate)}
            paired = {
                "score_rate": 0.6,
                "mean_candidate_reward": 2.0,
                "mean_champion_reward": 1.0,
                "candidate_invalid_actions": 0,
                "champion_invalid_actions": 0,
            }
            anchor_candidate = {"score_rate": 0.2}
            anchor_champion = {"score_rate": 0.1}
            with (
                mock.patch("tanat_ai40.cycle_ai42_ppo.train_self_play", return_value=training),
                mock.patch("tanat_ai40.cycle_ai42_ppo.evaluate_pair", return_value=paired),
                mock.patch(
                    "tanat_ai40.cycle_ai42_ppo.evaluate_vs_ai30",
                    side_effect=[anchor_candidate, anchor_champion],
                ),
            ):
                result = run_cycle(
                    root / "champion.pt", root / "config.json", root / "env.exe",
                    root / "cycle", device=torch.device("cpu"), train_seconds=1,
                    workers=2, rollout_steps=2, max_match_steps=10, seed=1,
                    eval_matches=2, eval_workers=2, ppo_config=AI42PPOConfig(),
                    past_opponent_fraction=0.2,
                )
            self.assertTrue(result["promotion_recommended"])
            self.assertFalse(result["promotion_applied"])
            self.assertTrue((root / "cycle" / "cycle-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
