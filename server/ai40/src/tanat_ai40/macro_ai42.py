"""Shadow-only AI-42 low-frequency team macro policy.

The macro policy is deliberately independent from the AI-40/41 micro actor and
from the Go orchestrator.  It consumes only an actor-visible team observation:
five allied hero/readiness tokens, team context, objective tokens and a
sanitized previous plan.  It produces a bounded strategic assignment for each
of the five hero slots; movement coordinates, targets for abilities and skill
actions are outside this contract.

Hero token convention
---------------------
By default feature zero is ``alive`` (one means alive) and feature one is
``retreating`` (one means retreating).  Callers may provide explicit masks to
avoid relying on those convention fields.  Dead or retreating rows are
zeroed before encoding and are forced to ``recover``/``no commit`` at decode.

Objective convention
--------------------
The last objective-pointer class is a finite ``no objective`` sentinel.  The
objective mask applies only to the real objective slots; masked slots receive
the lowest finite representable logit and can never win greedy decoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
from torch import nn


ALLIED_HERO_COUNT: Final = 5
MACRO_MODES: Final = ("farm", "defend", "push", "gank", "group", "recover")
MODE_NAMES: Final = MACRO_MODES
MODE_COUNT: Final = len(MACRO_MODES)
RECOVER_MODE: Final = MODE_COUNT - 1
NO_OBJECTIVE: Final = -1
COMMITMENT_NO: Final = 0
COMMITMENT_YES: Final = 1

# These aliases document the relationship to the current legacy orchestrator
# without importing it or wiring this shadow policy into the live server.
ORCHESTRATOR_MODE_ALIASES: Final = {
    "farm": "lane",
    "defend": "base",
    "push": "push",
    "gank": "rally",
    "group": "altar",
    "recover": "recover",
}


class _Projection(nn.Sequential):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )


class MacroPolicyConfig:
    """Small, explicit dimensions for production and cheap unit-test models."""

    def __init__(
        self,
        *,
        hero_features: int = 12,
        team_features: int = 24,
        objective_features: int = 12,
        previous_plan_features: int = 12,
        model_width: int = 64,
        transformer_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        role_count: int = 5,
        lane_count: int = 3,
        commitment_bins: int = 2,
        horizon_bins: int = 4,
        alive_feature_index: int | None = 0,
        retreating_feature_index: int | None = 1,
    ):
        positive = {
            "hero_features": hero_features,
            "team_features": team_features,
            "objective_features": objective_features,
            "previous_plan_features": previous_plan_features,
            "model_width": model_width,
            "transformer_layers": transformer_layers,
            "num_heads": num_heads,
            "ff_multiplier": ff_multiplier,
            "role_count": role_count,
            "lane_count": lane_count,
            "commitment_bins": commitment_bins,
            "horizon_bins": horizon_bins,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if model_width % num_heads:
            raise ValueError("model_width must be divisible by num_heads")
        if commitment_bins < 2:
            raise ValueError("commitment_bins must contain no-commit and commit")
        for name, index, width in (
            ("alive_feature_index", alive_feature_index, hero_features),
            ("retreating_feature_index", retreating_feature_index, hero_features),
        ):
            if index is not None and not 0 <= index < width:
                raise ValueError(f"{name} must be None or an index in hero_features")

        self.hero_features = hero_features
        self.team_features = team_features
        self.objective_features = objective_features
        self.previous_plan_features = previous_plan_features
        self.model_width = model_width
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.ff_multiplier = ff_multiplier
        self.role_count = role_count
        self.lane_count = lane_count
        self.commitment_bins = commitment_bins
        self.horizon_bins = horizon_bins
        self.alive_feature_index = alive_feature_index
        self.retreating_feature_index = retreating_feature_index


class _TeamTransformerBlock(nn.Module):
    """Pre-norm entity/team attention with finite padding semantics."""

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

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_expanded = valid.unsqueeze(-1)
        tokens = torch.where(valid_expanded, tokens, torch.zeros_like(tokens))

        # The team token is always valid, but retain the dummy-key fallback so
        # this block remains safe if it is reused with a fully padded set.
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        dummy = torch.zeros_like(safe_valid)
        dummy[:, 0] = True
        safe_valid = torch.where(empty.unsqueeze(1), dummy, safe_valid)

        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = torch.where(valid_expanded, tokens, torch.zeros_like(tokens))
        tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return torch.where(valid_expanded, tokens, torch.zeros_like(tokens))


class AI42MacroPolicy(nn.Module):
    """Standalone MAT-style low-frequency team policy.

    ``forward`` returns logits.  ``greedy_decode`` performs the deterministic
    five-step autoregressive decode.  The order of positional inputs is:

    ``allied_heroes``: ``[batch, 5, hero_features]``;
    ``team_state``: ``[batch, team_features]``;
    ``objectives``: ``[batch, objective_slots, objective_features]``;
    ``objective_mask``: ``[batch, objective_slots]``;
    ``previous_plan``: ``[batch, plan_tokens, previous_plan_features]`` or
    ``[batch, previous_plan_features]`` for one team-level plan token.

    ``mode``/``objective``/``role``/``lane``/``commitment``/``horizon`` logits
    have one row per allied hero.  ``team_mode`` is an additional summary head;
    it does not replace the five assignment heads.
    """

    assignment_count = ALLIED_HERO_COUNT
    mode_names = MACRO_MODES
    no_objective = NO_OBJECTIVE

    def __init__(
        self,
        hero_features: int = 12,
        team_features: int = 24,
        objective_features: int = 12,
        previous_plan_features: int = 12,
        *,
        model_width: int = 64,
        transformer_layers: int = 2,
        entity_layers: int | None = None,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        role_count: int = 5,
        lane_count: int = 3,
        commitment_bins: int = 2,
        horizon_bins: int = 4,
        alive_feature_index: int | None = 0,
        retreating_feature_index: int | None = 1,
        config: MacroPolicyConfig | None = None,
    ):
        super().__init__()
        if config is None:
            config = MacroPolicyConfig(
                hero_features=hero_features,
                team_features=team_features,
                objective_features=objective_features,
                previous_plan_features=previous_plan_features,
                model_width=model_width,
                transformer_layers=transformer_layers if entity_layers is None else entity_layers,
                num_heads=num_heads,
                ff_multiplier=ff_multiplier,
                role_count=role_count,
                lane_count=lane_count,
                commitment_bins=commitment_bins,
                horizon_bins=horizon_bins,
                alive_feature_index=alive_feature_index,
                retreating_feature_index=retreating_feature_index,
            )
        elif entity_layers is not None and entity_layers != config.transformer_layers:
            raise ValueError("entity_layers conflicts with config.transformer_layers")
        self.config = config
        self.hero_features = config.hero_features
        self.team_features = config.team_features
        self.objective_features = config.objective_features
        self.previous_plan_features = config.previous_plan_features
        self.model_width = config.model_width
        self.role_count = config.role_count
        self.lane_count = config.lane_count
        self.commitment_bins = config.commitment_bins
        self.horizon_bins = config.horizon_bins

        width = config.model_width
        self.hero_encoder = _Projection(config.hero_features, width)
        self.team_encoder = _Projection(config.team_features, width)
        self.objective_encoder = _Projection(config.objective_features, width)
        self.previous_plan_encoder = _Projection(config.previous_plan_features, width)
        self.hero_type = nn.Parameter(torch.zeros(1, 1, width))
        self.team_type = nn.Parameter(torch.zeros(1, 1, width))
        self.objective_type = nn.Parameter(torch.zeros(1, 1, width))
        self.previous_plan_type = nn.Parameter(torch.zeros(1, 1, width))
        self.hero_slot_embedding = nn.Embedding(ALLIED_HERO_COUNT, width)

        feedforward_width = width * config.ff_multiplier
        self.entity_attention = nn.ModuleList([
            _TeamTransformerBlock(width, config.num_heads, feedforward_width)
            for _ in range(config.transformer_layers)
        ])
        self.team_summary = _Projection(width, width)
        self.objective_summary = _Projection(width, width)

        # The recurrent decoder is the MAT-style assignment chain.  Each step
        # reads one hero token, then the next state receives the full selected
        # assignment embedding before the following hero is decoded.
        self.assignment_input = _Projection(width * 3, width)
        self.assignment_decoder = nn.GRUCell(width, width)
        self.assignment_recurrence = nn.GRUCell(width, width)
        self.assignment_slot_embedding = nn.Embedding(ALLIED_HERO_COUNT, width)

        self.mode_head = nn.Linear(width, MODE_COUNT)
        self.team_mode_head = nn.Linear(width, MODE_COUNT)
        self.objective_query = nn.Linear(width, width)
        self.objective_null_head = nn.Linear(width, 1)
        self.role_head = nn.Linear(width, config.role_count)
        self.lane_head = nn.Linear(width, config.lane_count)
        self.commitment_head = nn.Linear(width, config.commitment_bins)
        self.horizon_head = nn.Linear(width, config.horizon_bins)

        self.mode_embedding = nn.Embedding(MODE_COUNT, width)
        self.role_embedding = nn.Embedding(config.role_count, width)
        self.lane_embedding = nn.Embedding(config.lane_count, width)
        self.commitment_embedding = nn.Embedding(config.commitment_bins, width)
        self.horizon_embedding = nn.Embedding(config.horizon_bins, width)
        self.no_objective_embedding = nn.Parameter(torch.zeros(width))

    @staticmethod
    def _finite(value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _require_shape(name: str, value: torch.Tensor, wanted: tuple[int, ...]) -> None:
        if tuple(value.shape) != wanted:
            raise ValueError(f"{name} must have shape {wanted}, got {tuple(value.shape)}")

    def sanitize_hero_tokens(
        self,
        allied_heroes: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None = None,
        retreating_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite hero tokens and the rows that require recovery."""

        if allied_heroes.ndim != 3 or allied_heroes.shape[1] != ALLIED_HERO_COUNT:
            raise ValueError(
                f"allied_heroes must have shape [batch, {ALLIED_HERO_COUNT}, features]"
            )
        heroes = self._finite(allied_heroes)
        batch = heroes.shape[0]
        wanted = (batch, ALLIED_HERO_COUNT)
        if dead_mask is None:
            alive_index = self.config.alive_feature_index
            dead = (
                torch.zeros(wanted, dtype=torch.bool, device=heroes.device)
                if alive_index is None
                else heroes[..., alive_index].le(0.5)
            )
        else:
            self._require_shape("dead_mask", dead_mask, wanted)
            dead = dead_mask.to(device=heroes.device, dtype=torch.bool)
        if retreating_mask is None:
            retreat_index = self.config.retreating_feature_index
            retreating = (
                torch.zeros(wanted, dtype=torch.bool, device=heroes.device)
                if retreat_index is None
                else heroes[..., retreat_index].gt(0.5)
            )
        else:
            self._require_shape("retreating_mask", retreating_mask, wanted)
            retreating = retreating_mask.to(device=heroes.device, dtype=torch.bool)
        recovery = dead | retreating
        heroes = torch.where(recovery.unsqueeze(-1), torch.zeros_like(heroes), heroes)
        return heroes, recovery

    def sanitize_previous_plan(self, previous_plan: torch.Tensor) -> torch.Tensor:
        """Make a previous-plan tensor finite without adding hidden state."""

        if previous_plan.ndim not in (2, 3):
            raise ValueError("previous_plan must be [batch, features] or [batch, tokens, features]")
        if previous_plan.shape[-1] != self.previous_plan_features:
            raise ValueError(
                "previous_plan feature width does not match previous_plan_features: "
                f"{previous_plan.shape[-1]} != {self.previous_plan_features}"
            )
        return self._finite(previous_plan)

    def _validate_inputs(
        self,
        allied_heroes: torch.Tensor,
        team_state: torch.Tensor,
        objectives: torch.Tensor,
        objective_mask: torch.Tensor,
        previous_plan: torch.Tensor,
    ) -> tuple[int, int]:
        if allied_heroes.ndim != 3 or allied_heroes.shape[1:] != (
            ALLIED_HERO_COUNT,
            self.hero_features,
        ):
            raise ValueError(
                "allied_heroes must have shape "
                f"[batch, {ALLIED_HERO_COUNT}, {self.hero_features}]"
            )
        batch = allied_heroes.shape[0]
        self._require_shape("team_state", team_state, (batch, self.team_features))
        if objectives.ndim != 3 or objectives.shape[0] != batch or objectives.shape[2] != self.objective_features:
            raise ValueError(
                "objectives must have shape "
                f"[batch, objective_slots, {self.objective_features}]"
            )
        objective_slots = objectives.shape[1]
        if objective_slots < 1:
            raise ValueError("objectives must provide at least one masked slot")
        self._require_shape("objective_mask", objective_mask, (batch, objective_slots))
        if previous_plan.ndim == 2:
            self._require_shape(
                "previous_plan", previous_plan, (batch, self.previous_plan_features)
            )
        elif previous_plan.ndim == 3:
            if previous_plan.shape[0] != batch or previous_plan.shape[2] != self.previous_plan_features:
                raise ValueError(
                    "previous_plan must have shape "
                    f"[batch, plan_tokens, {self.previous_plan_features}]"
                )
        else:
            raise ValueError("previous_plan must be [batch, features] or [batch, tokens, features]")
        return batch, objective_slots

    def _encode(
        self,
        heroes: torch.Tensor,
        team_state: torch.Tensor,
        objectives: torch.Tensor,
        objective_mask: torch.Tensor,
        previous_plan: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = heroes.shape[0]
        device = heroes.device
        if previous_plan.ndim == 2:
            previous_plan = previous_plan.unsqueeze(1)

        hero_slots = torch.arange(ALLIED_HERO_COUNT, device=device)
        hero_tokens = self.hero_encoder(heroes)
        hero_tokens = hero_tokens + self.hero_type + self.hero_slot_embedding(hero_slots).unsqueeze(0)
        team_tokens = self.team_encoder(team_state).unsqueeze(1) + self.team_type
        objective_mask = objective_mask.to(device=device, dtype=torch.bool)
        safe_objectives = torch.where(
            objective_mask.unsqueeze(-1), self._finite(objectives), torch.zeros_like(objectives)
        )
        objective_tokens = self.objective_encoder(safe_objectives) + self.objective_type
        previous_tokens = self.previous_plan_encoder(previous_plan) + self.previous_plan_type

        previous_valid = torch.ones(
            (batch, previous_tokens.shape[1]), dtype=torch.bool, device=device
        )
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=device),
                previous_valid,
                torch.ones((batch, ALLIED_HERO_COUNT), dtype=torch.bool, device=device),
                objective_mask,
            ),
            dim=1,
        )
        tokens = torch.cat((team_tokens, previous_tokens, hero_tokens, objective_tokens), dim=1)
        for block in self.entity_attention:
            tokens = block(tokens, valid)
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        cursor = 1
        previous_end = cursor + previous_tokens.shape[1]
        hero_end = previous_end + ALLIED_HERO_COUNT
        hero_memory = tokens[:, previous_end:hero_end]
        objective_memory = tokens[:, hero_end:]
        team_memory = tokens[:, 0]
        team_context = self.team_summary(team_memory)
        objective_count = objective_mask.sum(dim=1, keepdim=True).to(objective_memory.dtype)
        objective_mean = objective_memory.sum(dim=1) / objective_count.clamp_min(1)
        has_objective = objective_count.gt(0)
        objective_mean = torch.where(
            has_objective, objective_mean, torch.zeros_like(objective_mean)
        )
        team_context = team_context + self.objective_summary(objective_mean)
        return team_context, hero_memory, objective_memory, objective_mask

    @staticmethod
    def _force_class(logits: torch.Tensor, class_index: int, force: torch.Tensor) -> torch.Tensor:
        low = torch.full_like(logits, torch.finfo(logits.dtype).min)
        forced = low.clone()
        forced[..., class_index] = 0
        return torch.where(force.unsqueeze(-1), forced, logits)

    def _head_step(
        self,
        hidden: torch.Tensor,
        objective_memory: torch.Tensor,
        objective_mask: torch.Tensor,
        recovery: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mode = self.mode_head(hidden)
        mode = self._force_class(mode, RECOVER_MODE, recovery)
        role = self.role_head(hidden)
        lane = self.lane_head(hidden)
        commitment = self.commitment_head(hidden)
        commitment = self._force_class(commitment, COMMITMENT_NO, recovery)
        horizon = self.horizon_head(hidden)

        query = self.objective_query(hidden)
        objective = torch.einsum("bd,bod->bo", query, objective_memory)
        objective = objective / (self.model_width ** 0.5)
        objective = torch.where(
            objective_mask, objective, torch.full_like(objective, torch.finfo(objective.dtype).min)
        )
        null_objective = self.objective_null_head(hidden)
        objective = torch.cat((objective, null_objective), dim=-1)
        force_no_objective = recovery | ~objective_mask.any(dim=1)
        null_only = torch.full_like(objective, torch.finfo(objective.dtype).min)
        null_only[..., -1] = 0
        objective = torch.where(force_no_objective.unsqueeze(-1), null_only, objective)
        return {
            "mode": mode,
            "objective": objective,
            "role": role,
            "lane": lane,
            "commitment": commitment,
            "horizon": horizon,
        }

    def _assignment_embedding(
        self,
        selections: Mapping[str, torch.Tensor],
        objective_memory: torch.Tensor,
    ) -> torch.Tensor:
        mode = selections["mode"]
        objective = selections["objective"]
        role = selections["role"]
        lane = selections["lane"]
        commitment = selections["commitment"]
        horizon = selections["horizon"]
        if objective_memory.ndim != 3:
            raise ValueError(
                "objective_memory must have shape [batch, objective_slots, width]"
            )
        if objective.ndim != 1 or objective.shape[0] != objective_memory.shape[0]:
            raise ValueError(
                "selected objective must have shape [batch] matching objective_memory"
            )
        objective_count = objective_memory.shape[1]
        safe_objective = objective.to(
            device=objective_memory.device, dtype=torch.long
        ).clamp(0, max(objective_count - 1, 0))
        gather_index = safe_objective.view(-1, 1, 1).expand(
            -1, 1, objective_memory.shape[-1]
        )
        selected = objective_memory.gather(
            1, gather_index
        ).squeeze(1)
        no_objective = objective.eq(NO_OBJECTIVE).unsqueeze(-1)
        selected = torch.where(
            no_objective, self.no_objective_embedding.view(1, -1), selected
        )
        return (
            self.mode_embedding(mode)
            + selected
            + self.role_embedding(role)
            + self.lane_embedding(lane)
            + self.commitment_embedding(commitment)
            + self.horizon_embedding(horizon)
        )

    @staticmethod
    def _teacher_value(
        teacher_assignments: Mapping[str, torch.Tensor] | None,
        name: str,
    ) -> torch.Tensor | None:
        if teacher_assignments is None:
            return None
        aliases = {"objective": ("objective", "objectives"), "commitment": ("commitment", "commit")}
        for key in aliases.get(name, (name,)):
            if key in teacher_assignments:
                return teacher_assignments[key]
        return None

    def _run_decoder(
        self,
        team_context: torch.Tensor,
        hero_memory: torch.Tensor,
        objective_memory: torch.Tensor,
        objective_mask: torch.Tensor,
        recovery: torch.Tensor,
        *,
        teacher_assignments: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = team_context.shape[0]
        hidden = torch.zeros_like(team_context)
        logits: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("mode", "objective", "role", "lane", "commitment", "horizon")
        }
        selections: dict[str, list[torch.Tensor]] = {key: [] for key in logits}
        for slot in range(ALLIED_HERO_COUNT):
            slot_context = torch.cat(
                (
                    team_context,
                    hero_memory[:, slot],
                    self.assignment_slot_embedding.weight[slot].expand(batch, -1),
                ),
                dim=-1,
            )
            decoder_input = self.assignment_input(slot_context)
            hidden = self.assignment_decoder(decoder_input, hidden)
            step_logits = self._head_step(hidden, objective_memory, objective_mask, recovery[:, slot])
            for key, value in step_logits.items():
                logits[key].append(value)

            step_selection: dict[str, torch.Tensor] = {}
            for key, value in step_logits.items():
                teacher = self._teacher_value(teacher_assignments, key)
                if teacher is not None:
                    if tuple(teacher.shape) != (batch, ALLIED_HERO_COUNT):
                        raise ValueError(
                            f"teacher_assignments[{key!r}] must have shape "
                            f"[{batch}, {ALLIED_HERO_COUNT}]"
                        )
                    selected = teacher[:, slot].to(device=value.device, dtype=torch.long)
                else:
                    selected = value.argmax(dim=-1)
                if key == "objective":
                    valid_objective = selected.lt(objective_memory.shape[1])
                    selected = torch.where(valid_objective, selected, torch.full_like(selected, NO_OBJECTIVE))
                    selected = torch.where(
                        objective_mask.gather(
                            1,
                            selected.clamp(0, objective_memory.shape[1] - 1).unsqueeze(1),
                        ).squeeze(1) & valid_objective,
                        selected,
                        torch.full_like(selected, NO_OBJECTIVE),
                    )
                    selected = torch.where(recovery[:, slot], torch.full_like(selected, NO_OBJECTIVE), selected)
                elif key == "mode":
                    selected = torch.where(recovery[:, slot], torch.full_like(selected, RECOVER_MODE), selected)
                    selected = selected.clamp(0, MODE_COUNT - 1)
                elif key == "commitment":
                    selected = torch.where(recovery[:, slot], torch.full_like(selected, COMMITMENT_NO), selected)
                    selected = selected.clamp(0, self.commitment_bins - 1)
                elif key == "role":
                    selected = selected.clamp(0, self.role_count - 1)
                elif key == "lane":
                    selected = selected.clamp(0, self.lane_count - 1)
                else:
                    selected = selected.clamp(0, self.horizon_bins - 1)
                step_selection[key] = selected
                selections[key].append(selected)
            assignment_embedding = self._assignment_embedding(step_selection, objective_memory)
            hidden = self.assignment_recurrence(assignment_embedding, hidden)

        stacked_logits = {key: torch.stack(value, dim=1) for key, value in logits.items()}
        stacked_selections = {key: torch.stack(value, dim=1) for key, value in selections.items()}
        return stacked_logits, stacked_selections

    def _prepare(
        self,
        allied_heroes: torch.Tensor,
        team_state: torch.Tensor,
        objectives: torch.Tensor,
        objective_mask: torch.Tensor,
        previous_plan: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None,
        retreating_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(allied_heroes, team_state, objectives, objective_mask, previous_plan)
        heroes, recovery = self.sanitize_hero_tokens(
            allied_heroes, dead_mask=dead_mask, retreating_mask=retreating_mask
        )
        team_state = self._finite(team_state)
        previous_plan = self.sanitize_previous_plan(previous_plan)
        return (
            heroes,
            team_state,
            self._finite(objectives),
            objective_mask,
            previous_plan,
            recovery,
        )

    def forward(
        self,
        allied_heroes: torch.Tensor,
        team_state: torch.Tensor,
        objectives: torch.Tensor,
        objective_mask: torch.Tensor,
        previous_plan: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None = None,
        retreating_mask: torch.Tensor | None = None,
        teacher_assignments: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode an observation and return finite, unmasked macro logits."""

        heroes, team_state, objectives, objective_mask, previous_plan, recovery = self._prepare(
            allied_heroes,
            team_state,
            objectives,
            objective_mask,
            previous_plan,
            dead_mask=dead_mask,
            retreating_mask=retreating_mask,
        )
        team_context, hero_memory, objective_memory, objective_mask = self._encode(
            heroes, team_state, objectives, objective_mask, previous_plan
        )
        assignment_logits, _ = self._run_decoder(
            team_context,
            hero_memory,
            objective_memory,
            objective_mask,
            recovery,
            teacher_assignments=teacher_assignments,
        )
        team_mode = self.team_mode_head(team_context)
        team_mode = self._force_class(team_mode, RECOVER_MODE, recovery.any(dim=1))
        return {
            # Per-hero mode is the autoregressive assignment head.
            "mode": assignment_logits["mode"],
            "objective": assignment_logits["objective"],
            "role": assignment_logits["role"],
            "lane": assignment_logits["lane"],
            "commitment": assignment_logits["commitment"],
            "horizon": assignment_logits["horizon"],
            "team_mode": team_mode,
            "assignment_mode": assignment_logits["mode"],
            "mode_logits": assignment_logits["mode"],
            "objective_logits": assignment_logits["objective"],
            "role_logits": assignment_logits["role"],
            "lane_logits": assignment_logits["lane"],
            "commitment_logits": assignment_logits["commitment"],
            "horizon_logits": assignment_logits["horizon"],
            "team_mode_logits": team_mode,
            "recovery_mask": recovery,
        }

    @torch.no_grad()
    def greedy_decode(
        self,
        allied_heroes: torch.Tensor,
        team_state: torch.Tensor,
        objectives: torch.Tensor,
        objective_mask: torch.Tensor,
        previous_plan: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None = None,
        retreating_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return exactly five deterministic greedy assignments."""

        heroes, team_state, objectives, objective_mask, previous_plan, recovery = self._prepare(
            allied_heroes,
            team_state,
            objectives,
            objective_mask,
            previous_plan,
            dead_mask=dead_mask,
            retreating_mask=retreating_mask,
        )
        team_context, hero_memory, objective_memory, objective_mask = self._encode(
            heroes, team_state, objectives, objective_mask, previous_plan
        )
        _, selected = self._run_decoder(
            team_context, hero_memory, objective_memory, objective_mask, recovery
        )
        team_mode = self.team_mode_head(team_context).argmax(dim=-1)
        team_mode = torch.where(
            recovery.any(dim=1), torch.full_like(team_mode, RECOVER_MODE), team_mode
        )
        return {
            "mode": selected["mode"],
            "objective": selected["objective"],
            "role": selected["role"],
            "lane": selected["lane"],
            "commitment": selected["commitment"],
            "horizon": selected["horizon"],
            "team_mode": team_mode,
            "assignment_mode": selected["mode"],
            "recovery_mask": recovery,
        }

    decode = greedy_decode
    deterministic_decode = greedy_decode


AI42MacroModel = AI42MacroPolicy


def parameter_count(module: nn.Module) -> int:
    """Return total model parameter count for test/configuration reporting."""

    return sum(parameter.numel() for parameter in module.parameters())
