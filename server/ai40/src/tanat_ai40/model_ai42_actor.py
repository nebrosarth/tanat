"""Standalone entity-centric actor for the reserved AI-42 contract.

The actor deliberately keeps the AI-41 tensor boundary while changing the
entity encoder.  Entity slots have no positional parameters: self-attention
is therefore permutation equivariant, and the masked mean/count summary used
by the recurrent core is permutation invariant.  Recurrent state is supplied
by the caller, so one shared module can keep an independent state per hero.

The actor remains independent from the training-only critic and accepts only
actor-side observation tensors.  Learner, native-runtime, and optional export
adapters consume its explicit public output contract; no value head or
privileged full-state input is present here.
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
    HERO_FEATURES,
    NAVIGATION_ANCHORS,
    NAVIGATION_OFFSETS,
)


# Control is deliberately separate from the v13 action-kind vocabulary.  It
# is the first recurrent decision: ISSUE exposes the existing action heads,
# while WAIT, HOLD, and CANCEL do not require fabricated action parameters.
CONTROL_ISSUE = 0
CONTROL_WAIT = 1
CONTROL_HOLD = 2
CONTROL_CANCEL = 3
CONTROL_CLASSES = 4
CONTROL_NAMES = ("issue", "wait", "hold", "cancel")


class _Projection(nn.Sequential):
    """Small bias-bearing projection shared by observation token types."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )


class _EntityAttentionBlock(nn.Module):
    """Pre-norm self-attention block with safe padding semantics.

    ``MultiheadAttention`` can produce NaNs when every key is padded.  For an
    all-masked row we temporarily expose a zero dummy key at slot zero, then
    mask every row again before returning.  This keeps valid rows unchanged,
    keeps invalid entity values out of the computation, and makes the empty
    set a finite, deterministic input to the actor.
    """

    def __init__(self, width: int, heads: int, feedforward_width: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True,
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

        # A fully padded key set is undefined for softmax.  The exposed dummy
        # contains zero because masked tokens were cleared above.
        safe_mask = entity_mask.clone()
        empty = ~entity_mask.any(dim=1)
        # Boolean-output ONNX Where is not implemented by all ORT execution
        # providers.  OR-ing the empty-row marker into slot zero is exactly the
        # same dummy-key rule and lowers to portable boolean operators.
        safe_mask = torch.cat(
            (safe_mask[:, :1] | empty.unsqueeze(1), safe_mask[:, 1:]), dim=1,
        )

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


class AI42Actor(nn.Module):
    """Entity-centric, recurrent micro actor for one controlled hero.

    The positional dimensions intentionally mirror :class:`AI41Policy`:

    ``hero`` is ``[batch, HERO_FEATURES]``;
    ``abilities`` is ``[batch, ABILITY_COUNT, ABILITY_FEATURES]``;
    ``entities`` is ``[batch, entity_slots, ENTITY_FEATURES]``;
    ``global_state`` is the actor-visible ``[batch, GLOBAL_FEATURES]``
    observation context (not a centralized critic/full-state tensor).

    ``entity_mask`` is the only entity validity mask consumed by this module.
    Action masks remain a caller concern until the AI-42 protocol/runtime
    adapter is implemented.  Output heads are already factorized by action
    kind, which makes autoregressive sampling or teacher replay direct.
    """

    uses_abilities = True
    navigation_actions = True
    has_value_head = False
    control_classes = CONTROL_CLASSES

    def __init__(
        self,
        hidden_size: int = 384,
        hero_vocab_size: int = 128,
        *,
        model_width: int = 384,
        entity_layers: int = 4,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        timing_bins: int = 4,
        ability_action_start: int = 3,
    ):
        super().__init__()
        if hidden_size < 1 or model_width < 1:
            raise ValueError("hidden_size and model_width must be positive")
        if hero_vocab_size < 1:
            raise ValueError("hero_vocab_size must be positive")
        if entity_layers < 1:
            raise ValueError("entity_layers must be positive")
        if num_heads < 1 or model_width % num_heads:
            raise ValueError("model_width must be divisible by num_heads")
        if ff_multiplier < 1:
            raise ValueError("ff_multiplier must be positive")
        if timing_bins < 1:
            raise ValueError("timing_bins must be positive")
        if not 0 <= ability_action_start <= ACTION_KINDS - ABILITY_COUNT:
            raise ValueError("ability action range does not fit in ACTION_KINDS")

        self.hidden_size = hidden_size
        self.model_width = model_width
        self.entity_layers = entity_layers
        self.num_heads = num_heads
        self.timing_bins = timing_bins
        self.ability_action_start = ability_action_start

        self.hero_encoder = _Projection(HERO_FEATURES, model_width)
        self.entity_encoder = _Projection(ENTITY_FEATURES, model_width)
        self.global_encoder = _Projection(GLOBAL_FEATURES, model_width)
        self.hero_id_embedding = nn.Embedding(hero_vocab_size, model_width)

        self.ability_encoder = _Projection(ABILITY_FEATURES, model_width)
        self.ability_slot_embedding = nn.Embedding(ABILITY_COUNT, model_width)
        self.ability_to_hero = nn.Linear(model_width, model_width)
        self.ability_to_action = nn.Linear(model_width, model_width)

        feedforward_width = model_width * ff_multiplier
        self.entity_attention = nn.ModuleList([
            _EntityAttentionBlock(model_width, num_heads, feedforward_width)
            for _ in range(entity_layers)
        ])
        self.entity_summary_encoder = _Projection(model_width, model_width)
        # Count is explicit rather than inferred from a mean, so one object
        # and a group of otherwise identical objects remain distinguishable.
        self.entity_count_encoder = _Projection(1, model_width)

        # Four observation contexts make the recurrent input explicit: hero,
        # entity set, actor-visible global context, and entity-count context.
        self.core = nn.LSTMCell(model_width * 4, hidden_size)
        self.state_to_width = _Projection(hidden_size, model_width)

        # Control is emitted directly from recurrent memory and gates the
        # action-parameter heads during teacher replay and inference.
        self.control_head = nn.Linear(hidden_size, CONTROL_CLASSES)
        self.kind_head = nn.Linear(hidden_size, ACTION_KINDS)
        self.action_kind_embedding = nn.Embedding(ACTION_KINDS, model_width)
        self.target_query = nn.Linear(model_width, model_width)
        self.entity_key = nn.Linear(model_width, model_width)
        self.offset_head = nn.Linear(model_width, NAVIGATION_OFFSETS)
        self.anchor_head = nn.Linear(model_width, NAVIGATION_ANCHORS)

        # These heads are reserved until the timing contract is defined.  They
        # are emitted for checkpoint/teacher compatibility but are not consumed
        # by the current action protocol or used as a value estimate.
        self.timing_head = nn.Linear(model_width, timing_bins)
        self.timing_aux_head = nn.Linear(model_width, timing_bins)

    @property
    def direction_head(self) -> nn.Linear:
        """AI-41 spelling for the 81-way local navigation head."""

        return self.offset_head

    @property
    def distance_head(self) -> nn.Linear:
        """AI-41 spelling for the 15-way navigation-anchor head."""

        return self.anchor_head

    def initial_state(
        self,
        batch: int,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an independent zero LSTM state for each hero row."""

        if batch < 1:
            raise ValueError("batch must be positive")
        parameter = next(self.parameters())
        if device is None:
            device = parameter.device
        if dtype is None:
            dtype = parameter.dtype
        shape = (batch, self.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def _hero_ids(self, hero: torch.Tensor, hero_ids: torch.Tensor | None) -> torch.Tensor:
        if hero_ids is None:
            # AI-41 encodes the stable hero id in normalized feature zero.
            hero_ids = torch.round(hero[:, 0] * 100).long()
        else:
            hero_ids = hero_ids.to(device=hero.device, dtype=torch.long).reshape(-1)
        return hero_ids.clamp(0, self.hero_id_embedding.num_embeddings - 1)

    @staticmethod
    def _validate_shapes(
        hero: torch.Tensor,
        abilities: torch.Tensor,
        entities: torch.Tensor,
        global_state: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> None:
        batch = hero.shape[0]
        expected = {
            "hero": (batch, HERO_FEATURES),
            "abilities": (batch, ABILITY_COUNT, ABILITY_FEATURES),
            "entities": (batch, entities.shape[1], ENTITY_FEATURES),
            "global_state": (batch, GLOBAL_FEATURES),
            "entity_mask": (batch, entities.shape[1]),
        }
        actual = {
            "hero": tuple(hero.shape),
            "abilities": tuple(abilities.shape),
            "entities": tuple(entities.shape),
            "global_state": tuple(global_state.shape),
            "entity_mask": tuple(entity_mask.shape),
        }
        for name, wanted in expected.items():
            if actual[name] != wanted:
                raise ValueError(f"{name} must have shape {wanted}, got {actual[name]}")
        if entities.shape[1] < 1:
            raise ValueError("entities must contain at least one slot")

    def forward(
        self,
        hero: torch.Tensor,
        abilities: torch.Tensor,
        entities: torch.Tensor,
        global_state: torch.Tensor,
        entity_mask: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        *,
        hero_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance one hero observation and return unmasked action logits.

        ``h`` and ``c`` are not shared or mutated in place.  Callers should
        retain the returned state independently for every controlled hero.
        """

        self._validate_shapes(hero, abilities, entities, global_state, entity_mask)
        mask = entity_mask.to(device=entities.device, dtype=torch.bool)
        valid = mask.unsqueeze(-1)

        # Clear invalid values before any learned projection.  ``where`` is
        # intentional: multiplying a NaN by zero would still leave a NaN.
        safe_entities = torch.where(valid, entities, torch.zeros_like(entities))
        entity_tokens = self.entity_encoder(safe_entities)
        entity_tokens = torch.where(valid, entity_tokens, torch.zeros_like(entity_tokens))
        for block in self.entity_attention:
            entity_tokens = block(entity_tokens, mask)

        count = mask.sum(dim=1, keepdim=True).to(dtype=entity_tokens.dtype)
        entity_summary = entity_tokens.sum(dim=1) / count.clamp_min(1)
        entity_summary = self.entity_summary_encoder(entity_summary)
        has_entities = count.gt(0)
        entity_summary = torch.where(has_entities, entity_summary, torch.zeros_like(entity_summary))
        count_context = self.entity_count_encoder(count / max(entities.shape[1], 1))
        count_context = torch.where(has_entities, count_context, torch.zeros_like(count_context))

        ability_tokens = self.ability_encoder(abilities)
        ability_tokens = ability_tokens + self.ability_slot_embedding.weight.unsqueeze(0)
        ability_summary = ability_tokens.mean(dim=1)
        hero_embedding = (
            self.hero_encoder(hero)
            + self.hero_id_embedding(self._hero_ids(hero, hero_ids))
            + self.ability_to_hero(ability_summary)
        )
        global_embedding = self.global_encoder(global_state)

        if h is None or c is None:
            h, c = self.initial_state(hero.shape[0], hero.device, dtype=hero.dtype)
        if h.shape != (hero.shape[0], self.hidden_size) or c.shape != h.shape:
            raise ValueError(
                f"recurrent state must have shape {(hero.shape[0], self.hidden_size)}"
            )
        recurrent_input = torch.cat(
            (hero_embedding, entity_summary, global_embedding, count_context), dim=-1,
        )
        h, c = self.core(recurrent_input, (h, c))
        state_context = self.state_to_width(h)

        action_context = (
            state_context.unsqueeze(1)
            + self.action_kind_embedding.weight.unsqueeze(0)
        )
        skill_context = self.ability_to_action(ability_tokens)
        prefix = action_context.new_zeros(
            hero.shape[0], self.ability_action_start, self.model_width,
        )
        suffix = action_context.new_zeros(
            hero.shape[0], ACTION_KINDS - self.ability_action_start - ABILITY_COUNT,
            self.model_width,
        )
        action_context = action_context + torch.cat((prefix, skill_context, suffix), dim=1)

        query = self.target_query(action_context)
        key = self.entity_key(entity_tokens)
        target_logits = torch.einsum("bkd,bnd->bkn", query, key)
        target_logits = target_logits / (self.model_width ** 0.5)
        # Invalid slots are never valid action targets, and zero is finite even
        # for the all-masked observation.  The runtime may apply richer action
        # masks after selecting a kind.
        target_logits = torch.where(
            mask.unsqueeze(1), target_logits, torch.zeros_like(target_logits),
        )

        control_logits = self.control_head(h)
        kind_logits = self.kind_head(h)
        offset_logits = self.offset_head(action_context)
        anchor_logits = self.anchor_head(action_context)
        timing_logits = self.timing_head(action_context)
        timing_aux_logits = self.timing_aux_head(action_context)
        return {
            "control": control_logits,
            "kind": kind_logits,
            "target": target_logits,
            "offset": offset_logits,
            "anchor": anchor_logits,
            "timing": timing_logits,
            "timing_aux": timing_aux_logits,
            # Preserve the AI-41 names while the AI-42 protocol is reserved.
            "direction": offset_logits,
            "distance": anchor_logits,
            "h": h,
            "c": c,
        }


# The policy alias makes the standalone actor easy to discover without
# implying compatibility with AI-40/41 checkpoints.
AI42Policy = AI42Actor
AI42MicroActor = AI42Actor


def selected_action_logits(
    output: dict[str, torch.Tensor],
    kinds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select all parameter heads for one sampled/replayed action kind."""

    rows = torch.arange(kinds.shape[0], device=kinds.device)
    selected = {
        "control": output["control"],
        "kind": output["kind"],
        "target": output["target"][rows, kinds],
        "offset": output["offset"][rows, kinds],
        "anchor": output["anchor"][rows, kinds],
        "timing": output["timing"][rows, kinds],
        "timing_aux": output["timing_aux"][rows, kinds],
        "h": output["h"],
        "c": output["c"],
    }
    selected["direction"] = selected["offset"]
    selected["distance"] = selected["anchor"]
    return selected


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in module.parameters())
