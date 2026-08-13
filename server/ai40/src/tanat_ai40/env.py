from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import struct
import subprocess
from typing import BinaryIO, Iterable

import numpy as np

HERO_COUNT = 10
MAX_ENTITIES = 96
HERO_FEATURES = 32
ENTITY_FEATURES = 16
GLOBAL_FEATURES = 32
ACTION_KINDS = 8
CONTROLLER_EXTERNAL = 0
CONTROLLER_AI20 = 1
CONTROLLER_AI30 = 2
CONTROLLER_AI40 = 3
AI40_SELF_PLAY_CONTROLLERS = (CONTROLLER_AI40,) * HERO_COUNT
AI40_ROSTER = np.asarray([7, 10, 13, 17, 18, 22, 23, 35, 36, 38], dtype=np.int32)
PROTOCOL_VERSION = 3
MAGIC = b"TANT"
SCHEMA = (
    "tanat.assault.v1|hero=10x32|entities=10x96x16|global=10x32|"
    "action=kind,target,direction,distance|mask=8+96+4x96"
)
SCHEMA_HASH = hashlib.sha256(SCHEMA.encode()).digest()
REWARD_SCHEMA = (
    "tanat.assault.reward.v2|xp=.002|money_gain=.04|money_spend=.004|hp=2|mana=.75|"
    "death=-1|hero_kill=-.6|creep_last_hit=-.16|structure=two_thirds_damage+"
    "one_third_destroy|win=5|zero_sum=1|team_spirit=.2"
)
REWARD_HASH = hashlib.sha256(REWARD_SCHEMA.encode()).digest()
OBSERVATION_DTYPE = np.dtype([
    ("hero", "<f4", (HERO_FEATURES,)),
    ("entities", "<f4", (MAX_ENTITIES, ENTITY_FEATURES)),
    ("global_state", "<f4", (GLOBAL_FEATURES,)),
    ("entity_mask", "u1", (MAX_ENTITIES,)),
    ("kind_mask", "u1", (ACTION_KINDS,)),
    ("target_mask", "u1", (MAX_ENTITIES,)),
    ("skill_target_mask", "u1", (4, MAX_ENTITIES)),
], align=False)
ACTION_DTYPE = np.dtype([
    ("kind", "u1"), ("target", "<u2"),
    ("direction", "u1"), ("distance", "u1"),
], align=False)


def self_play_rosters(seed_rng: np.random.Generator, count: int) -> list[list[int]]:
    """Shuffle the complete hero pool so every avatar trains on both sides."""
    if count < 0:
        raise ValueError("roster count cannot be negative")
    return [seed_rng.permutation(AI40_ROSTER).tolist() for _ in range(count)]


@dataclass(slots=True)
class HeroAction:
    kind: int = 0
    target: int = 0
    direction: int = 0
    distance: int = 0


@dataclass(slots=True)
class StepResult:
    step: int
    elapsed: float
    done: bool
    winner: int
    invalid: np.ndarray
    rewards: np.ndarray
    hero: np.ndarray
    entities: np.ndarray
    global_state: np.ndarray
    entity_mask: np.ndarray
    kind_mask: np.ndarray
    target_mask: np.ndarray
    skill_target_mask: np.ndarray


class AssaultProtocolError(RuntimeError):
    pass


class AssaultEnvProcess:
    """One synchronous Go environment process; create one per rollout worker."""

    def __init__(self, executable: str | os.PathLike[str]):
        self.executable = Path(executable)
        self.process = subprocess.Popen(
            [str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        assert self.process.stdin and self.process.stdout
        self._in: BinaryIO = self.process.stdin
        self._out: BinaryIO = self.process.stdout

    def reset(
        self,
        seed: int,
        max_steps: int = 4_500,
        roster: Iterable[int] | None = None,
        controllers: Iterable[int] | None = None,
    ) -> StepResult:
        roster_values = list(roster or [0] * HERO_COUNT)
        controller_values = list(controllers or [0] * HERO_COUNT)
        if len(roster_values) != HERO_COUNT or len(controller_values) != HERO_COUNT:
            raise ValueError("roster and controllers must contain exactly 10 values")
        payload = struct.pack("<qI10i10B", seed, max_steps, *roster_values, *controller_values)
        self._write(1, payload)
        return self._read_result()

    def step(self, actions: Iterable[HeroAction]) -> StepResult:
        if isinstance(actions, np.ndarray):
            values = np.asarray(actions)
            if values.shape != (HERO_COUNT, 4):
                raise ValueError("action array must have shape (10, 4)")
            packed = np.empty(HERO_COUNT, dtype=ACTION_DTYPE)
            packed["kind"] = values[:, 0]
            packed["target"] = values[:, 1]
            packed["direction"] = values[:, 2]
            packed["distance"] = values[:, 3]
            payload = packed.tobytes()
            self._write(2, payload)
            return self._read_result()
        values = list(actions)
        if len(values) != HERO_COUNT:
            raise ValueError("actions must contain exactly 10 HeroAction values")
        payload = b"".join(struct.pack("<B H B B", a.kind, a.target, a.direction, a.distance) for a in values)
        self._write(2, payload)
        return self._read_result()

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_pipes()
            return
        try:
            self._write(3, b"")
            self.process.wait(timeout=3)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def __enter__(self) -> "AssaultEnvProcess":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write(self, command: int, payload: bytes) -> None:
        body = MAGIC + struct.pack("<HH", PROTOCOL_VERSION, command) + payload
        self._in.write(struct.pack("<I", len(body)) + body)
        self._in.flush()

    def _read_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._out.read(size - len(chunks))
            if not chunk:
                error = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
                raise AssaultProtocolError(f"assaultenv closed its output: {error}")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_result(self) -> StepResult:
        size = struct.unpack("<I", self._read_exact(4))[0]
        body = self._read_exact(size)
        if body[:4] != MAGIC:
            raise AssaultProtocolError("bad response magic")
        version, response = struct.unpack_from("<HH", body, 4)
        if version != PROTOCOL_VERSION:
            raise AssaultProtocolError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
        if response == 255:
            raise AssaultProtocolError(body[8:].decode(errors="replace"))
        if response != 100:
            raise AssaultProtocolError(f"unknown response {response}")
        if body[8:40] != SCHEMA_HASH:
            raise AssaultProtocolError("schema hash mismatch")
        if body[40:72] != REWARD_HASH:
            raise AssaultProtocolError("reward hash mismatch")
        step, elapsed, done, winner = struct.unpack_from("<IfB3xi", body, 72)
        offset = 88
        invalid = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
        offset += HERO_COUNT
        rewards = np.frombuffer(body, dtype="<f4", count=HERO_COUNT, offset=offset)
        offset += HERO_COUNT * 4
        observations = np.frombuffer(
            body, dtype=OBSERVATION_DTYPE, count=HERO_COUNT, offset=offset,
        )
        hero = observations["hero"]
        entities = observations["entities"]
        global_state = observations["global_state"]
        entity_mask = observations["entity_mask"]
        kind_mask = observations["kind_mask"]
        target_mask = observations["target_mask"]
        skill_target_mask = observations["skill_target_mask"]
        offset += HERO_COUNT * OBSERVATION_DTYPE.itemsize
        if offset != len(body):
            raise AssaultProtocolError(f"result has {len(body)-offset} trailing bytes")
        return StepResult(step, elapsed, bool(done), winner, invalid, rewards, hero, entities,
                          global_state, entity_mask, kind_mask, target_mask, skill_target_mask)


class AssaultVectorEnv:
    """Synchronous vector of independent Go rollout processes."""

    def __init__(self, executable: str | os.PathLike[str], workers: int):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.envs = [AssaultEnvProcess(executable) for _ in range(workers)]
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="assault-rollout")

    def reset(
        self, seeds: Iterable[int], max_steps: int = 4_500,
        controllers: Iterable[int] | None = None,
        controller_sets: Iterable[Iterable[int]] | None = None,
        rosters: Iterable[Iterable[int]] | None = None,
    ) -> list[StepResult]:
        values = list(seeds)
        if len(values) != len(self.envs):
            raise ValueError("one seed is required per worker")
        if controllers is not None and controller_sets is not None:
            raise ValueError("pass controllers or controller_sets, not both")
        default_controllers = None if controllers is None else list(controllers)
        controller_values = ([default_controllers] * len(self.envs) if controller_sets is None
                             else [list(worker_controllers) for worker_controllers in controller_sets])
        if len(controller_values) != len(self.envs):
            raise ValueError("one controller set is required per worker")
        roster_values = ([None] * len(self.envs) if rosters is None
                         else [list(roster) for roster in rosters])
        if len(roster_values) != len(self.envs):
            raise ValueError("one roster is required per worker")
        futures = [self._pool.submit(env.reset, seed, max_steps, roster, worker_controllers)
                   for env, seed, roster, worker_controllers in
                   zip(self.envs, values, roster_values, controller_values)]
        return [future.result() for future in futures]

    def reset_indices(
        self, indices: Iterable[int], seeds: Iterable[int], max_steps: int,
        controllers: Iterable[int] | None = None,
        controller_sets: Iterable[Iterable[int]] | None = None,
        rosters: Iterable[Iterable[int]] | None = None,
    ) -> dict[int, StepResult]:
        pairs = list(zip(indices, seeds))
        if controllers is not None and controller_sets is not None:
            raise ValueError("pass controllers or controller_sets, not both")
        default_controllers = None if controllers is None else list(controllers)
        controller_values = ([default_controllers] * len(pairs) if controller_sets is None
                             else [list(worker_controllers) for worker_controllers in controller_sets])
        if len(controller_values) != len(pairs):
            raise ValueError("one controller set is required per reset index")
        roster_values = ([None] * len(pairs) if rosters is None
                         else [list(roster) for roster in rosters])
        if len(roster_values) != len(pairs):
            raise ValueError("one roster is required per reset index")
        futures = {
            index: self._pool.submit(
                self.envs[index].reset, seed, max_steps, roster, worker_controllers,
            )
            for (index, seed), roster, worker_controllers in
            zip(pairs, roster_values, controller_values)
        }
        return {index: future.result() for index, future in futures.items()}

    def step(self, actions: Iterable[Iterable[HeroAction]]) -> list[StepResult]:
        if isinstance(actions, np.ndarray):
            values = [worker_actions for worker_actions in actions]
        else:
            values = [list(worker_actions) for worker_actions in actions]
        if len(values) != len(self.envs):
            raise ValueError("one action batch is required per worker")
        futures = [self._pool.submit(env.step, worker_actions) for env, worker_actions in zip(self.envs, values)]
        return [future.result() for future in futures]

    def close(self) -> None:
        futures = [self._pool.submit(env.close) for env in self.envs]
        for future in futures:
            future.result()
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "AssaultVectorEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
