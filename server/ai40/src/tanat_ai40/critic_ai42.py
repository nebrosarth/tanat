"""Training-only centralized multi-head critic for the reserved AI-42 model.

The critic consumes privileged learner state and is deliberately independent
of the actor, protocol, exporter, and runtime modules.  Entity slots have no
positional parameters and are reduced only after permutation-equivariant
self-attention.  Hero slots likewise share all parameters, so a simultaneous
permutation of every per-hero input only permutes the per-hero value rows.
"""

from __future__ import annotations

import torch
from torch import nn

from .env import (
    ABILITY_COUNT,
    ABILITY_FEATURES,
    ACTION_KINDS,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HERO_COUNT,
    HERO_FEATURES,
)


MACRO_TEAM_COUNT = 2
MACRO_FEATURES = 6
EXECUTED_ACTION_FEATURES = 4
VALUE_HEAD_NAMES = (
    "total",
    "win",
    "structures",
    "economy",
    "survival",
    "teamfight",
)


class _Projection(nn.Sequential):
    """Bias-bearing nonlinear projection for one typed input."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )


class _EntityAttentionBlock(nn.Module):
    """Pre-norm entity attention with finite empty-set semantics."""

    def __init__(self, width: int, heads: int, feedforward_width: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.SiLU(),
            nn.Linear(feedforward_width, width),
        )

    def forward(self, tokens: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        valid = entity_mask.unsqueeze(-1)
        tokens = torch.where(valid, tokens, torch.zeros_like(tokens))

        # MultiheadAttention has undefined softmax rows when every key is
        # masked.  Expose one zero dummy key for those rows, then clear all
        # invalid outputs again before the next block or pooling operation.
        empty = ~entity_mask.any(dim=1)
        dummy_slot = torch.cat(
            (
                torch.ones_like(entity_mask[:, :1]),
                torch.zeros_like(entity_mask[:, 1:]),
            ),
            dim=1,
        )
        safe_mask = torch.where(empty.unsqueeze(1), dummy_slot, entity_mask)

        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = torch.where(valid, tokens, torch.zeros_like(tokens))

        tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return torch.where(valid, tokens, torch.zeros_like(tokens))


class _HeroAttentionBlock(nn.Module):
    """Shared self-attention block over the ten unordered hero slots."""

    def __init__(self, width: int, heads: int, feedforward_width: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.SiLU(),
            nn.Linear(feedforward_width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.norm_feedforward(tokens))


class _ValueHead(nn.Sequential):
    """Independent scalar value head applied to every hero slot."""

    def __init__(self, width: int):
        super().__init__(
            nn.LayerNorm(width),
            nn.Linear(width, max(width // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(width // 2, 1), 1),
        )


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Choose a valid attention-head count for an integer test width."""

    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class AI42CentralizedCritic(nn.Module):
    """Centralized full-state critic used only by a learner.

    Inputs have the following shapes:

    - ``heroes``: ``[batch, 10, HERO_FEATURES]``;
    - ``abilities``: ``[batch, 10, 4, ABILITY_FEATURES]``;
    - ``entities`` and ``entity_mask``: ``[batch, N, ENTITY_FEATURES]`` and
      ``[batch, N]``;
    - ``global_state``: ``[batch, 2, GLOBAL_FEATURES]``;
    - ``macro_plans``: ``[batch, 2, 6]``;
    - ``assignments``: ``[batch, 10, ACTION_KINDS]``;
    - ``executed_actions``: ``[batch, 10, 4]``.

    Every returned value is ``[batch, 10]``.  ``test_size`` can be ``True``
    for a small deterministic configuration or a positive integer to select
    its model width; explicit size arguments remain available for tests.
    """

    training_only = True
    has_value_heads = True
    accepts_privileged_state = True

    def __init__(
        self,
        *,
        model_width: int | None = None,
        entity_layers: int | None = None,
        hero_layers: int | None = None,
        num_heads: int | None = None,
        ff_multiplier: int | None = None,
        test_size: bool | int = False,
    ):
        if isinstance(test_size, bool):
            test_width = 32 if test_size else 256
            defaults = (1, 1, 4, 2) if test_size else (2, 2, 8, 4)
        elif isinstance(test_size, int):
            if test_size < 1:
                raise ValueError("test_size must be positive")
            test_width = test_size
            defaults = (1, 1, 4, 2)
        else:
            raise TypeError("test_size must be bool or int")

        model_width = test_width if model_width is None else model_width
        entity_layers = defaults[0] if entity_layers is None else entity_layers
        hero_layers = defaults[1] if hero_layers is None else hero_layers
        num_heads = defaults[2] if num_heads is None else num_heads
        ff_multiplier = defaults[3] if ff_multiplier is None else ff_multiplier

        if isinstance(test_size, int) and not isinstance(test_size, bool):
            if num_heads > model_width or model_width % num_heads:
                num_heads = _largest_divisor_at_most(model_width, num_heads)
        if model_width < 1:
            raise ValueError("model_width must be positive")
        if entity_layers < 1 or hero_layers < 1:
            raise ValueError("attention layer counts must be positive")
        if num_heads < 1 or model_width % num_heads:
            raise ValueError("model_width must be divisible by num_heads")
        if ff_multiplier < 1:
            raise ValueError("ff_multiplier must be positive")

        super().__init__()
        self.model_width = model_width
        self.entity_layers = entity_layers
        self.hero_layers = hero_layers
        self.num_heads = num_heads
        self.ff_multiplier = ff_multiplier
        self.test_size = test_size

        self.hero_encoder = _Projection(HERO_FEATURES, model_width)
        self.ability_encoder = _Projection(ABILITY_FEATURES, model_width)
        self.ability_slot_embedding = nn.Embedding(ABILITY_COUNT, model_width)
        self.ability_to_hero = nn.Linear(model_width, model_width)
        self.assignment_encoder = _Projection(ACTION_KINDS, model_width)
        self.executed_action_encoder = _Projection(
            EXECUTED_ACTION_FEATURES, model_width,
        )

        self.entity_encoder = _Projection(ENTITY_FEATURES, model_width)
        self.entity_attention = nn.ModuleList([
            _EntityAttentionBlock(
                model_width,
                num_heads,
                model_width * ff_multiplier,
            )
            for _ in range(entity_layers)
        ])
        self.entity_summary_encoder = _Projection(model_width, model_width)
        self.entity_count_encoder = _Projection(1, model_width)

        self.global_encoder = _Projection(GLOBAL_FEATURES, model_width)
        self.macro_encoder = _Projection(MACRO_FEATURES, model_width)
        self.hero_input_norm = nn.LayerNorm(model_width)
        self.hero_attention = nn.ModuleList([
            _HeroAttentionBlock(
                model_width,
                num_heads,
                model_width * ff_multiplier,
            )
            for _ in range(hero_layers)
        ])

        self.value_heads = nn.ModuleDict({
            name: _ValueHead(model_width) for name in VALUE_HEAD_NAMES
        })

    @staticmethod
    def _validate_shapes(
        heroes: torch.Tensor,
        abilities: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        global_state: torch.Tensor,
        macro_plans: torch.Tensor,
        assignments: torch.Tensor,
        executed_actions: torch.Tensor,
    ) -> None:
        tensors = {
            "heroes": (heroes, 3),
            "abilities": (abilities, 4),
            "entities": (entities, 3),
            "entity_mask": (entity_mask, 2),
            "global_state": (global_state, 3),
            "macro_plans": (macro_plans, 3),
            "assignments": (assignments, 3),
            "executed_actions": (executed_actions, 3),
        }
        for name, (value, dimensions) in tensors.items():
            if value.ndim != dimensions:
                raise ValueError(
                    f"{name} must have rank {dimensions}, got {value.ndim}"
                )

        batch = heroes.shape[0]
        entity_slots = entities.shape[1]
        expected = {
            "heroes": (batch, HERO_COUNT, HERO_FEATURES),
            "abilities": (batch, HERO_COUNT, ABILITY_COUNT, ABILITY_FEATURES),
            "entities": (batch, entity_slots, ENTITY_FEATURES),
            "entity_mask": (batch, entity_slots),
            "global_state": (batch, MACRO_TEAM_COUNT, GLOBAL_FEATURES),
            "macro_plans": (batch, MACRO_TEAM_COUNT, MACRO_FEATURES),
            "assignments": (batch, HERO_COUNT, ACTION_KINDS),
            "executed_actions": (batch, HERO_COUNT, EXECUTED_ACTION_FEATURES),
        }
        actual = {
            "heroes": tuple(heroes.shape),
            "abilities": tuple(abilities.shape),
            "entities": tuple(entities.shape),
            "entity_mask": tuple(entity_mask.shape),
            "global_state": tuple(global_state.shape),
            "macro_plans": tuple(macro_plans.shape),
            "assignments": tuple(assignments.shape),
            "executed_actions": tuple(executed_actions.shape),
        }
        for name, wanted in expected.items():
            if actual[name] != wanted:
                raise ValueError(f"{name} must have shape {wanted}, got {actual[name]}")
        if entity_slots < 1:
            raise ValueError("entities must contain at least one slot")

    def forward(
        self,
        heroes: torch.Tensor,
        abilities: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        global_state: torch.Tensor,
        macro_plans: torch.Tensor,
        assignments: torch.Tensor,
        executed_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return independent per-hero value estimates for learner updates."""

        self._validate_shapes(
            heroes,
            abilities,
            entities,
            entity_mask,
            global_state,
            macro_plans,
            assignments,
            executed_actions,
        )
        mask = entity_mask.to(device=entities.device, dtype=torch.bool)
        valid = mask.unsqueeze(-1)

        # Clear padded entity data before learned projections.  This also
        # makes all-masked rows finite if their padding contains NaNs.
        safe_entities = torch.where(valid, entities, torch.zeros_like(entities))
        entity_tokens = self.entity_encoder(safe_entities)
        entity_tokens = torch.where(valid, entity_tokens, torch.zeros_like(entity_tokens))
        for block in self.entity_attention:
            entity_tokens = block(entity_tokens, mask)

        count = mask.sum(dim=1, keepdim=True).to(dtype=entity_tokens.dtype)
        entity_summary = entity_tokens.sum(dim=1) / count.clamp_min(1)
        entity_summary = self.entity_summary_encoder(entity_summary)
        has_entities = count.gt(0)
        entity_summary = torch.where(
            has_entities,
            entity_summary,
            torch.zeros_like(entity_summary),
        )
        count_context = self.entity_count_encoder(
            count / max(entities.shape[1], 1),
        )
        count_context = torch.where(
            has_entities,
            count_context,
            torch.zeros_like(count_context),
        )
        entity_context = entity_summary + count_context

        ability_tokens = self.ability_encoder(abilities)
        ability_tokens = ability_tokens + self.ability_slot_embedding.weight.view(
            1, 1, ABILITY_COUNT, self.model_width,
        )
        ability_summary = ability_tokens.mean(dim=2)

        global_context = self.global_encoder(global_state).mean(dim=1)
        macro_context = self.macro_encoder(macro_plans).mean(dim=1)
        shared_context = global_context + macro_context + entity_context

        # All operations below use shared weights and no hero-slot embedding.
        # Therefore the hero stack is permutation equivariant by construction.
        hero_tokens = self.hero_encoder(heroes)
        hero_tokens = hero_tokens + self.ability_to_hero(ability_summary)
        hero_tokens = hero_tokens + self.assignment_encoder(assignments)
        hero_tokens = hero_tokens + self.executed_action_encoder(executed_actions)
        hero_tokens = hero_tokens + shared_context.unsqueeze(1)
        hero_tokens = self.hero_input_norm(hero_tokens)
        for block in self.hero_attention:
            hero_tokens = block(hero_tokens)

        return {
            name: head(hero_tokens).squeeze(-1)
            for name, head in self.value_heads.items()
        }


AI42Critic = AI42CentralizedCritic
CentralizedMultiHeadCritic = AI42CentralizedCritic


def parameter_count(module: nn.Module) -> int:
    """Return the number of parameters in a critic instance."""

    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "AI42CentralizedCritic",
    "AI42Critic",
    "CentralizedMultiHeadCritic",
    "EXECUTED_ACTION_FEATURES",
    "MACRO_FEATURES",
    "MACRO_TEAM_COUNT",
    "VALUE_HEAD_NAMES",
    "parameter_count",
]
