from __future__ import annotations

import torch
from torch import nn

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_FEATURES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)


class AI41Policy(nn.Module):
    """Ability-aware recurrent policy with action-conditioned parameter heads."""

    uses_abilities = True
    navigation_actions = False

    def __init__(
        self,
        hidden_size: int = 256,
        hero_vocab_size: int = 128,
        *,
        direction_bins: int = 16,
        distance_bins: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # Names and shapes shared with AI40Policy are intentional: the migration
        # utility can retain the mature encoders, recurrent core and base heads.
        self.hero_encoder = nn.Sequential(
            nn.Linear(HERO_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(ENTITY_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.hero_id_embedding = nn.Embedding(hero_vocab_size, 128)
        self.ability_encoder = nn.Sequential(
            nn.Linear(ABILITY_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.ability_hero = nn.Linear(128, 128)
        self.core = nn.LSTMCell(384, hidden_size)
        self.kind_head = nn.Linear(hidden_size, ACTION_KINDS)

        # Every action kind gets its own context. Skill1..Skill4 additionally
        # receive the encoded record for their corresponding ability slot.
        self.action_kind_embedding = nn.Embedding(ACTION_KINDS, hidden_size)
        self.ability_action = nn.Linear(128, hidden_size)
        self.target_query = nn.Linear(hidden_size, 128)
        self.direction_head = nn.Linear(hidden_size, direction_bins)
        self.distance_head = nn.Linear(hidden_size, distance_bins)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def forward(self, hero, abilities, entities, global_state, entity_mask, h, c):
        hero_ids = torch.clamp(torch.round(hero[:, 0] * 100).long(), 0,
                               self.hero_id_embedding.num_embeddings - 1)
        ability_embedding = self.ability_encoder(abilities)
        ability_pool = ability_embedding.amax(dim=1)
        hero_embedding = (
            self.hero_encoder(hero) + self.hero_id_embedding(hero_ids) +
            self.ability_hero(ability_pool)
        )
        entity_embedding = self.entity_encoder(entities)
        mask = entity_mask.bool().unsqueeze(-1)
        pooled = entity_embedding.masked_fill(~mask, -1e9).amax(dim=1)
        pooled = torch.where(
            entity_mask.bool().any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled),
        )
        global_embedding = self.global_encoder(global_state)
        h, c = self.core(
            torch.cat((hero_embedding, pooled, global_embedding), dim=-1), (h, c),
        )

        batch = hero.shape[0]
        action_context = h.unsqueeze(1) + self.action_kind_embedding.weight.unsqueeze(0)
        skill_context = torch.zeros(
            batch, ACTION_KINDS, self.hidden_size,
            dtype=action_context.dtype, device=action_context.device,
        )
        skill_context[:, 3:3 + ABILITY_COUNT] = self.ability_action(ability_embedding)
        action_context = action_context + skill_context

        queries = self.target_query(action_context)
        target_logits = torch.einsum("bkd,bed->bke", queries, entity_embedding) / (128.0 ** 0.5)
        return {
            "kind": self.kind_head(h),
            "target": target_logits,
            "direction": self.direction_head(action_context),
            "distance": self.distance_head(action_context),
            "value": self.value_head(h).squeeze(-1),
            "h": h,
            "c": c,
        }


class AI41NavigationPolicy(AI41Policy):
    """AI-41 with an OpenAI-Five-style 9x9 grid and global map anchors."""

    navigation_actions = True

    def __init__(self, hidden_size: int = 256, hero_vocab_size: int = 128):
        super().__init__(
            hidden_size, hero_vocab_size,
            direction_bins=NAVIGATION_OFFSETS,
            distance_bins=NAVIGATION_ANCHORS,
        )


def selected_action_logits(output: dict[str, torch.Tensor], kinds: torch.Tensor) -> dict[str, torch.Tensor]:
    """Select kind-conditioned parameter logits for sampled/replayed actions."""
    rows = torch.arange(kinds.shape[0], device=kinds.device)
    return {
        "kind": output["kind"],
        "target": output["target"][rows, kinds],
        "direction": output["direction"][rows, kinds],
        "distance": output["distance"][rows, kinds],
        "value": output["value"],
        "h": output["h"],
        "c": output["c"],
    }


def migrate_ai40_state(policy: AI41Policy, ai40_state: dict[str, torch.Tensor]) -> list[str]:
    """Copy shape-compatible AI-40 weights and neutralize new residual branches."""
    current = policy.state_dict()
    copied: list[str] = []
    for name, value in ai40_state.items():
        if name in current and current[name].shape == value.shape:
            current[name] = value
            copied.append(name)
    policy.load_state_dict(current)
    with torch.no_grad():
        # New contexts start as zero residuals, preserving AI-40 behavior before
        # AI-41 training teaches hero/ability specialization.
        policy.hero_id_embedding.weight.zero_()
        policy.ability_hero.weight.zero_()
        policy.ability_hero.bias.zero_()
        policy.action_kind_embedding.weight.zero_()
        policy.ability_action.weight.zero_()
        policy.ability_action.bias.zero_()
    return copied
