"""Training tools for the shared recurrent AI-40 policy."""

from typing import TYPE_CHECKING

from .env import (
    AI40_ROSTER,
    AI40_SELF_PLAY_CONTROLLERS,
    AssaultEnvProcess,
    AssaultVectorEnv,
    HeroAction,
    REWARD_HASH,
    SCHEMA_HASH,
    StepResult,
    self_play_rosters,
)
if TYPE_CHECKING:
    from .model_ai42_actor import AI42Actor, AI42MicroActor, AI42Policy

__all__ = [
    "AI40_ROSTER",
    "AI40_SELF_PLAY_CONTROLLERS",
    "AssaultEnvProcess",
    "AssaultVectorEnv",
    "HeroAction",
    "StepResult",
    "SCHEMA_HASH",
    "REWARD_HASH",
    "self_play_rosters",
    "AI42Actor",
    "AI42MicroActor",
    "AI42Policy",
]


def __getattr__(name: str):
    """Lazily expose model classes without importing model code at package load."""

    if name in {"AI42Actor", "AI42MicroActor", "AI42Policy"}:
        from .model_ai42_actor import AI42Actor, AI42MicroActor, AI42Policy

        exports = {
            "AI42Actor": AI42Actor,
            "AI42MicroActor": AI42MicroActor,
            "AI42Policy": AI42Policy,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = exports[name]
    globals()[name] = value
    return value
