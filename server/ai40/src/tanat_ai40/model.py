from __future__ import annotations

import warnings
import torch
from torch import nn

from .env import ACTION_KINDS, ENTITY_FEATURES, GLOBAL_FEATURES, HERO_FEATURES, MAX_ENTITIES


class AI40Policy(nn.Module):
    """Shared recurrent hero policy; recurrent state remains per hero."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.hero_encoder = nn.Sequential(nn.Linear(HERO_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128))
        self.entity_encoder = nn.Sequential(nn.Linear(ENTITY_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128))
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128))
        self.core = nn.LSTMCell(384, hidden_size)
        self.kind_head = nn.Linear(hidden_size, ACTION_KINDS)
        self.direction_head = nn.Linear(hidden_size, 16)
        self.distance_head = nn.Linear(hidden_size, 3)
        self.target_query = nn.Linear(hidden_size, 128)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def forward(self, hero, entities, global_state, entity_mask, h, c):
        hero_embedding = self.hero_encoder(hero)
        entity_embedding = self.entity_encoder(entities)
        mask = entity_mask.bool().unsqueeze(-1)
        pooled = entity_embedding.masked_fill(~mask, -1e9).amax(dim=1)
        pooled = torch.where(entity_mask.bool().any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        global_embedding = self.global_encoder(global_state)
        h, c = self.core(torch.cat((hero_embedding, pooled, global_embedding), dim=-1), (h, c))
        query = self.target_query(h).unsqueeze(1)
        target_logits = (entity_embedding * query).sum(dim=-1) / (128.0 ** 0.5)
        return {
            "kind": self.kind_head(h), "target": target_logits,
            "direction": self.direction_head(h), "distance": self.distance_head(h),
            "value": self.value_head(h).squeeze(-1), "h": h, "c": c,
        }


class PolicyRunner:
    """Optional torch.compile wrapper that falls back to eager execution."""

    def __init__(
        self,
        policy: AI40Policy,
        compile_model: bool = False,
        *,
        mode: str = "default",
        dynamic: bool = True,
        fullgraph: bool = False,
    ):
        self.policy = policy
        self.compiled = None
        if compile_model and hasattr(torch, "compile"):
            try:
                self.compiled = torch.compile(
                    policy, mode=mode, dynamic=dynamic, fullgraph=fullgraph,
                )
            except Exception as exc:  # pragma: no cover - backend/platform specific
                warnings.warn(f"torch.compile setup failed; using eager policy: {exc}")

    def __call__(self, *args, **kwargs):
        if self.compiled is not None:
            try:
                return self.compiled(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - backend/platform specific
                warnings.warn(f"torch.compile execution failed; using eager policy: {exc}")
                self.compiled = None
        return self.policy(*args, **kwargs)

    @property
    def active(self) -> bool:
        return self.compiled is not None


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask.bool(), -1e9))
