from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .env import ENTITY_FEATURES, GLOBAL_FEATURES, HERO_FEATURES, MAX_ENTITIES, REWARD_HASH, SCHEMA_HASH
from .model import AI40Policy


class ExportWrapper(nn.Module):
    def __init__(self, policy: AI40Policy):
        super().__init__()
        self.policy = policy

    def forward(self, hero, entities, global_state, entity_mask, h, c):
        out = self.policy(hero, entities, global_state, entity_mask, h, c)
        return out["kind"], out["target"], out["direction"], out["distance"], out["value"], out["h"], out["c"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ai40.onnx"))
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_hash") != SCHEMA_HASH.hex():
        raise RuntimeError("checkpoint schema hash does not match AssaultObservationV1")
    if saved.get("reward_hash") != REWARD_HASH.hex():
        raise RuntimeError("checkpoint reward hash does not match AssaultRewardV2")
    policy = AI40Policy()
    policy.load_state_dict(saved["model"])
    policy.eval()
    batch = 10
    inputs = (
        torch.zeros(batch, HERO_FEATURES), torch.zeros(batch, MAX_ENTITIES, ENTITY_FEATURES),
        torch.zeros(batch, GLOBAL_FEATURES), torch.ones(batch, MAX_ENTITIES),
        torch.zeros(batch, policy.hidden_size), torch.zeros(batch, policy.hidden_size),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(policy).eval()
    torch.onnx.export(wrapper, inputs, args.output,
                      input_names=["hero", "entities", "global", "entity_mask", "h", "c"],
                      output_names=["kind", "target", "direction", "distance", "value", "next_h", "next_c"],
                      dynamic_axes={name: {0: "heroes"} for name in
                                    ("hero", "entities", "global", "entity_mask", "h", "c", "kind", "target", "direction", "distance", "value", "next_h", "next_c")},
                      opset_version=18)
    model_id = "AI-40-v3:" + hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]
    manifest = {"version": "AI-40-v3", "schema_hash": SCHEMA_HASH.hex(), "reward_hash": REWARD_HASH.hex(),
                "model_id": model_id,
                "model": args.output.name, "inputs": ["hero", "entities", "global", "entity_mask", "h", "c"],
                "outputs": ["kind", "target", "direction", "distance", "value", "next_h", "next_c"],
                "recurrent_size": policy.hidden_size, "opset": 18}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
