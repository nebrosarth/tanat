from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import threading
from typing import BinaryIO, Iterable

import numpy as np

HERO_COUNT = 10
MAX_ENTITIES = 96
HERO_FEATURES = 32
ENTITY_FEATURES = 16
GLOBAL_FEATURES = 32
ABILITY_COUNT = 4
ABILITY_FEATURES = 40
ACTION_KINDS = 8
NAVIGATION_OFFSETS = 81
NAVIGATION_ANCHORS = 15
CONTROLLER_EXTERNAL = 0
CONTROLLER_AI20 = 1
CONTROLLER_AI30 = 2
CONTROLLER_AI40 = 3
AI40_SELF_PLAY_CONTROLLERS = (CONTROLLER_AI40,) * HERO_COUNT
# Same alternating draft as the live Assault bot spawner: both sides receive
# a mixed five-hero roster when a fixed evaluation roster is used.
AI40_ROSTER = np.asarray([7, 13, 18, 23, 36, 10, 17, 22, 35, 38], dtype=np.int32)
PROTOCOL_VERSION = 4
AI41_LEGACY_PROTOCOL_VERSION = 5
AI41_PROTOCOL_VERSION = 6
AI41_EVALUATION_PROTOCOL_VERSION = 7
AI41_NAVIGATION_PROTOCOL_VERSION = 8
AI41_NAVIGATION_EVALUATION_PROTOCOL_VERSION = 9
AI41_STRATEGIC_PROTOCOL_VERSION = 10
AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION = 11
AI41_TEACHER_PROTOCOL_VERSION = 12
# AI-42 is an explicit opt-in append-only v12 actor observation contract.
AI42_PROTOCOL_VERSION = 13
AI42_RESERVED_PROTOCOL_VERSION = AI42_PROTOCOL_VERSION
AI42_EVALUATION_PROTOCOL_VERSION = 14
AI42_DAGGER_PROTOCOL_VERSION = 15
MAGIC = b"TANT"
COMMAND_RESET = 1
COMMAND_STEP = 2
COMMAND_CLOSE = 3
COMMAND_VECTOR_RESET = 4
COMMAND_VECTOR_STEP = 5
COMMAND_VECTOR_RESET_INDICES = 6
RESPONSE_RESULT = 100
RESPONSE_VECTOR_RESULT = 101
RESPONSE_ERROR = 255
SCHEMA = (
    "tanat.assault.v1|hero=10x32|entities=10x96x16|global=10x32|"
    "action=kind,target,direction,distance|mask=8+96+4x96"
)
SCHEMA_HASH = hashlib.sha256(SCHEMA.encode()).digest()
AI41_LEGACY_SCHEMA = (
    "tanat.assault.v2|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|"
    "action=kind,target,direction,distance|conditioned=kind|mask=8+96+4x96"
)
AI41_LEGACY_SCHEMA_HASH = hashlib.sha256(AI41_LEGACY_SCHEMA.encode()).digest()
AI41_SCHEMA = (
    "tanat.assault.v3|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|"
    "global_lane=active,onehot3,wrong|action=kind,target,direction,distance|"
    "conditioned=kind|mask=8+96+4x96"
)
AI41_SCHEMA_HASH = hashlib.sha256(AI41_SCHEMA.encode()).digest()
AI41_NAVIGATION_SCHEMA = (
    "tanat.assault.v4|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|"
    "global_lane=active,onehot3,wrong|action=kind,target,offset81,anchor15|"
    "conditioned=kind|anchors=local,bases,lanes3x4|mask=8+96+4x96"
)
AI41_NAVIGATION_SCHEMA_HASH = hashlib.sha256(AI41_NAVIGATION_SCHEMA.encode()).digest()
REWARD_SCHEMA = (
    "tanat.assault.reward.v2|xp=.002|money_gain=.04|money_spend=.004|hp=2|mana=.75|"
    "death=-1|hero_kill=-.6|creep_last_hit=-.16|structure=two_thirds_damage+"
    "one_third_destroy|win=5|zero_sum=1|team_spirit=.2"
)
REWARD_HASH = hashlib.sha256(REWARD_SCHEMA.encode()).digest()
AI41_REWARD_SCHEMA = (
    REWARD_SCHEMA + "|wrong_lane=-.15_per_second|lane_assignment=2-1-2|"
    "lane_until=360-600|lane_corridor=30"
)
AI41_REWARD_HASH = hashlib.sha256(AI41_REWARD_SCHEMA.encode()).digest()
AI41_STRATEGIC_REWARD_SCHEMA_V4 = (
    AI41_REWARD_SCHEMA +
    "|shaping_time_weight=.6^(elapsed/600s)|draw_timeout=-2_post_zero_sum"
)
AI41_STRATEGIC_REWARD_HASH_V4 = hashlib.sha256(
    AI41_STRATEGIC_REWARD_SCHEMA_V4.encode()
).digest()
AI41_STRATEGIC_REWARD_SCHEMA = (
    AI41_STRATEGIC_REWARD_SCHEMA_V4 +
    "|tanat_creep_last_hit_bonus=.24|standard_wave_last_hit_mean=.4"
)
AI41_STRATEGIC_REWARD_HASH = hashlib.sha256(AI41_STRATEGIC_REWARD_SCHEMA.encode()).digest()
AI42_SCHEMA = ""
AI42_SCHEMA_HASH = b""
AI42_EVALUATION_SCHEMA = ""
AI42_EVALUATION_SCHEMA_HASH = b""
AI42_DAGGER_SCHEMA = ""
AI42_DAGGER_SCHEMA_HASH = b""
# v13 deliberately uses the established strategic V5 reward semantics/hash.
AI42_REWARD_SCHEMA = AI41_STRATEGIC_REWARD_SCHEMA
AI42_REWARD_HASH = AI41_STRATEGIC_REWARD_HASH


def _protocol_has_abilities(protocol_version: int) -> bool:
    return protocol_version in (
        AI41_LEGACY_PROTOCOL_VERSION,
        AI41_PROTOCOL_VERSION,
        AI41_EVALUATION_PROTOCOL_VERSION,
        AI41_NAVIGATION_PROTOCOL_VERSION,
        AI41_NAVIGATION_EVALUATION_PROTOCOL_VERSION,
        AI41_STRATEGIC_PROTOCOL_VERSION,
        AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
        AI41_TEACHER_PROTOCOL_VERSION,
        AI42_PROTOCOL_VERSION,
        AI42_EVALUATION_PROTOCOL_VERSION,
        AI42_DAGGER_PROTOCOL_VERSION,
    )


def _protocol_schema_hash(protocol_version: int) -> bytes:
    if protocol_version == AI42_PROTOCOL_VERSION:
        return AI42_SCHEMA_HASH
    if protocol_version == AI42_EVALUATION_PROTOCOL_VERSION:
        return AI42_EVALUATION_SCHEMA_HASH
    if protocol_version == AI42_DAGGER_PROTOCOL_VERSION:
        return AI42_DAGGER_SCHEMA_HASH
    if protocol_version in (
        AI41_NAVIGATION_PROTOCOL_VERSION,
        AI41_NAVIGATION_EVALUATION_PROTOCOL_VERSION,
        AI41_STRATEGIC_PROTOCOL_VERSION,
        AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
        AI41_TEACHER_PROTOCOL_VERSION,
    ):
        return AI41_NAVIGATION_SCHEMA_HASH
    if protocol_version in (AI41_PROTOCOL_VERSION, AI41_EVALUATION_PROTOCOL_VERSION):
        return AI41_SCHEMA_HASH
    if protocol_version == AI41_LEGACY_PROTOCOL_VERSION:
        return AI41_LEGACY_SCHEMA_HASH
    return SCHEMA_HASH


def _protocol_reward_hash(protocol_version: int) -> bytes:
    if protocol_version in (
        AI41_STRATEGIC_PROTOCOL_VERSION,
        AI41_STRATEGIC_EVALUATION_PROTOCOL_VERSION,
        AI41_TEACHER_PROTOCOL_VERSION,
        AI42_PROTOCOL_VERSION,
        AI42_EVALUATION_PROTOCOL_VERSION,
        AI42_DAGGER_PROTOCOL_VERSION,
    ):
        return AI41_STRATEGIC_REWARD_HASH
    return (AI41_REWARD_HASH if protocol_version in (
        AI41_PROTOCOL_VERSION,
        AI41_EVALUATION_PROTOCOL_VERSION,
        AI41_NAVIGATION_PROTOCOL_VERSION,
        AI41_NAVIGATION_EVALUATION_PROTOCOL_VERSION,
    ) else REWARD_HASH)
OBSERVATION_DTYPE = np.dtype([
    ("hero", "<f4", (HERO_FEATURES,)),
    ("entities", "<f4", (MAX_ENTITIES, ENTITY_FEATURES)),
    ("global_state", "<f4", (GLOBAL_FEATURES,)),
    ("entity_mask", "u1", (MAX_ENTITIES,)),
    ("kind_mask", "u1", (ACTION_KINDS,)),
    ("target_mask", "u1", (MAX_ENTITIES,)),
    ("skill_target_mask", "u1", (4, MAX_ENTITIES)),
], align=False)
AI41_OBSERVATION_DTYPE = np.dtype([
    ("hero", "<f4", (HERO_FEATURES,)),
    ("abilities", "<f4", (ABILITY_COUNT, ABILITY_FEATURES)),
    ("entities", "<f4", (MAX_ENTITIES, ENTITY_FEATURES)),
    ("global_state", "<f4", (GLOBAL_FEATURES,)),
    ("entity_mask", "u1", (MAX_ENTITIES,)),
    ("kind_mask", "u1", (ACTION_KINDS,)),
    ("target_mask", "u1", (MAX_ENTITIES,)),
    ("skill_target_mask", "u1", (ABILITY_COUNT, MAX_ENTITIES)),
], align=False)
ACTION_DTYPE = np.dtype([
    ("kind", "u1"), ("target", "<u2"),
    ("direction", "u1"), ("distance", "u1"),
], align=False)
CONTROLLED_ACTION_DTYPE = np.dtype([
    ("control", "u1"), ("kind", "u1"), ("target", "<u2"),
    ("direction", "u1"), ("distance", "u1"),
], align=False)
DAGGER_ACTION_DTYPE = np.dtype([
    ("control", "u1"), ("kind", "u1"), ("target", "<u2"),
    ("direction", "u1"), ("distance", "u1"), ("intervention", "u1"),
], align=False)
RESET_PAYLOAD_SIZE = 8 + 4 + HERO_COUNT * 4 + HERO_COUNT
VECTOR_RESULT_HEADER_SIZE = 76
MAX_FRAME_SIZE = 64 << 20
STDERR_TAIL_BYTES = 65_536


def _layout_has_abilities(protocol_version: int) -> bool:
    return _protocol_has_abilities(protocol_version)


def _layout_has_teacher(protocol_version: int) -> bool:
    return protocol_version == AI41_TEACHER_PROTOCOL_VERSION or protocol_version == AI42_PROTOCOL_VERSION


def _layout_is_ai42(protocol_version: int) -> bool:
    return protocol_version in (AI42_PROTOCOL_VERSION, AI42_DAGGER_PROTOCOL_VERSION)


@dataclass(frozen=True, slots=True)
class _FrameLayout:
    size: int
    fields: tuple[tuple[str, int, int], ...]


def _result_layout(protocol_version: int, count: int = 1, vector: bool = False) -> _FrameLayout:
    if count < 1:
        raise ValueError("frame layout count must be positive")
    offset = 0
    fields: list[tuple[str, int, int]] = []

    def add(name: str, size: int) -> None:
        nonlocal offset
        fields.append((name, offset, size))
        offset += size

    add("frame.magic", 4)
    add("frame.version", 2)
    add("frame.response", 2)
    add("result.schema_hash", 32)
    add("result.reward_hash", 32)
    if vector:
        add("result.count", 4)
        add("result.steps", count * 4)
        add("result.elapsed", count * 4)
        add("result.done", count)
        add("result.winner", count * 4)
        actors = count * HERO_COUNT
        add("result.invalid", actors)
        add("result.reward", actors * 4)
        add("result.hero", actors * HERO_FEATURES * 4)
        if _layout_has_abilities(protocol_version):
            add("result.abilities", actors * ABILITY_COUNT * ABILITY_FEATURES * 4)
        add("result.entities", actors * MAX_ENTITIES * ENTITY_FEATURES * 4)
        add("result.global", actors * GLOBAL_FEATURES * 4)
        add("result.entity_mask", actors * MAX_ENTITIES)
        add("result.kind_mask", actors * ACTION_KINDS)
        add("result.target_mask", actors * MAX_ENTITIES)
        add("result.skill_target_mask", actors * ABILITY_COUNT * MAX_ENTITIES)
        if protocol_version == AI41_TEACHER_PROTOCOL_VERSION:
            add("result.teacher_actions", actors * ACTION_DTYPE.itemsize)
            add("result.teacher_valid", actors)
        elif _layout_is_ai42(protocol_version):
            add("result.teacher_intent", actors * ACTION_DTYPE.itemsize)
            add("result.teacher_status", actors)
            add("result.executed_actions", actors * ACTION_DTYPE.itemsize)
            add("result.executed_valid", actors)
            add("result.rejection_reason", actors)
            if protocol_version == AI42_DAGGER_PROTOCOL_VERSION:
                add("result.active_order", actors)
        elif protocol_version == AI42_EVALUATION_PROTOCOL_VERSION:
            add("result.active_order", actors)
    else:
        add("result.step", 4)
        add("result.elapsed", 4)
        add("result.done_padding", 4)
        add("result.winner", 4)
        add("result.invalid", HERO_COUNT)
        add("result.reward", HERO_COUNT * 4)
        for hero in range(HERO_COUNT):
            prefix = f"result.observation[{hero}]."
            add(prefix + "hero", HERO_FEATURES * 4)
            if _layout_has_abilities(protocol_version):
                add(prefix + "abilities", ABILITY_COUNT * ABILITY_FEATURES * 4)
            add(prefix + "entities", MAX_ENTITIES * ENTITY_FEATURES * 4)
            add(prefix + "global", GLOBAL_FEATURES * 4)
            add(prefix + "entity_mask", MAX_ENTITIES)
            add(prefix + "kind_mask", ACTION_KINDS)
            add(prefix + "target_mask", MAX_ENTITIES)
            add(prefix + "skill_target_mask", ABILITY_COUNT * MAX_ENTITIES)
        if protocol_version == AI41_TEACHER_PROTOCOL_VERSION:
            add("result.teacher_actions", HERO_COUNT * ACTION_DTYPE.itemsize)
            add("result.teacher_valid", HERO_COUNT)
        elif _layout_is_ai42(protocol_version):
            add("result.teacher_intent", HERO_COUNT * ACTION_DTYPE.itemsize)
            add("result.teacher_status", HERO_COUNT)
            add("result.executed_actions", HERO_COUNT * ACTION_DTYPE.itemsize)
            add("result.executed_valid", HERO_COUNT)
            add("result.rejection_reason", HERO_COUNT)
            if protocol_version == AI42_DAGGER_PROTOCOL_VERSION:
                add("result.active_order", HERO_COUNT)
        elif protocol_version == AI42_EVALUATION_PROTOCOL_VERSION:
            add("result.active_order", HERO_COUNT)
    return _FrameLayout(offset, tuple(fields))


def _layout_fields_text(layout: _FrameLayout) -> str:
    return ",".join(
        f"{name}@{field_offset}:{field_size}"
        for name, field_offset, field_size in layout.fields
    )


def _ai42_schema_text() -> str:
    scalar = _result_layout(AI42_PROTOCOL_VERSION)
    vector = _result_layout(AI42_PROTOCOL_VERSION, count=1, vector=True)
    return (
        "tanat.assault.ai42.v13|frame=little-endian|"
        "action=kind,target,offset81,anchor15|"
        "skill_navigation=offset81_only|"
        "teacher_status=none,action,wait,hold,cancel,unavailable|"
        "executed_reason=none,masked,invalid,server_rejected,safety,timeout,policy_error,unknown255|"
        f"scalar.body={scalar.size}|scalar.fields={_layout_fields_text(scalar)}|"
        f"vector.body={vector.size}|vector.fields={_layout_fields_text(vector)}"
    )


# Keep this immediately after the layout builder: the hash is a function of
# the implemented offsets/sizes, not a separately copied fixture string.
AI42_SCHEMA = _ai42_schema_text()
AI42_SCHEMA_HASH = hashlib.sha256(AI42_SCHEMA.encode("utf-8")).digest()


def _ai42_evaluation_schema_text() -> str:
    scalar = _result_layout(AI42_EVALUATION_PROTOCOL_VERSION)
    vector = _result_layout(AI42_EVALUATION_PROTOCOL_VERSION, count=1, vector=True)
    return (
        "tanat.assault.ai42.evaluation.v14|frame=little-endian|"
        "input=control,kind,target,offset81,anchor15|control=issue,hold,idle|"
        "skill_navigation=offset81_only|"
        f"scalar.body={scalar.size}|scalar.fields={_layout_fields_text(scalar)}|"
        f"vector.body={vector.size}|vector.fields={_layout_fields_text(vector)}"
    )


AI42_EVALUATION_SCHEMA = _ai42_evaluation_schema_text()
AI42_EVALUATION_SCHEMA_HASH = hashlib.sha256(
    AI42_EVALUATION_SCHEMA.encode("utf-8")
).digest()


def _ai42_dagger_schema_text() -> str:
    scalar = _result_layout(AI42_DAGGER_PROTOCOL_VERSION)
    vector = _result_layout(AI42_DAGGER_PROTOCOL_VERSION, count=1, vector=True)
    return (
        "tanat.assault.ai42.dagger.v15|frame=little-endian|"
        "input=control,kind,target,offset81,anchor15,intervention01|control=issue,hold,idle|"
        "teacher_status=none,action,wait,hold,cancel,unavailable|"
        "skill_navigation=offset81_only|"
        f"scalar.body={scalar.size}|scalar.fields={_layout_fields_text(scalar)}|"
        f"vector.body={vector.size}|vector.fields={_layout_fields_text(vector)}"
    )


AI42_DAGGER_SCHEMA = _ai42_dagger_schema_text()
AI42_DAGGER_SCHEMA_HASH = hashlib.sha256(AI42_DAGGER_SCHEMA.encode("utf-8")).digest()


def _layout_size_error(layout: _FrameLayout, got: int) -> "AssaultProtocolError | None":
    if got == layout.size:
        return None
    if got > layout.size:
        return AssaultProtocolError(
            f"field=result.trailing offset={layout.size}: got {got} bytes, want {layout.size}"
        )
    for name, field_offset, field_size in layout.fields:
        if got < field_offset + field_size:
            return AssaultProtocolError(
                f"field={name} offset={field_offset}: truncated, frame ends at {got}, need {field_offset + field_size}"
            )
    return AssaultProtocolError(f"field=result.body offset={got}: got {got} bytes, want {layout.size}")


def _validate_runtime_protocol(protocol_version: int) -> None:
    if protocol_version != PROTOCOL_VERSION and not _protocol_has_abilities(protocol_version):
        raise ValueError(f"unsupported runtime protocol version {protocol_version}")


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
    abilities: np.ndarray | None = None
    teacher_actions: np.ndarray | None = None
    teacher_valid: np.ndarray | None = None
    teacher_intent: np.ndarray | None = None
    teacher_status: np.ndarray | None = None
    executed_actions: np.ndarray | None = None
    executed_valid: np.ndarray | None = None
    rejection_reason: np.ndarray | None = None
    schema_hash: bytes | None = None
    reward_hash: bytes | None = None
    active_order: np.ndarray | None = None


class VectorStepResults(list[StepResult]):
    """List-compatible results with zero-copy structure-of-arrays observations."""

    def __init__(self, values: Iterable[StepResult], batched_observations: dict[str, np.ndarray]):
        super().__init__(values)
        self.batched_observations: dict[str, np.ndarray] | None = batched_observations

    def __setitem__(self, key, value) -> None:
        # A partial match reset replaces one record from a different response
        # buffer, so the original contiguous batch is no longer authoritative.
        self.batched_observations = None
        super().__setitem__(key, value)


class AssaultProtocolError(RuntimeError):
    pass


class AssaultEnvProcess:
    """One synchronous Go environment process; create one per rollout worker."""

    def __init__(self, executable: str | os.PathLike[str],
                 protocol_version: int = PROTOCOL_VERSION):
        _validate_runtime_protocol(protocol_version)
        self.executable = Path(executable)
        self.protocol_version = protocol_version
        self.observation_dtype = (AI41_OBSERVATION_DTYPE
                                  if _protocol_has_abilities(protocol_version) else OBSERVATION_DTYPE)
        self.schema_hash = _protocol_schema_hash(protocol_version)
        self.reward_hash = _protocol_reward_hash(protocol_version)
        self.process = subprocess.Popen(
            [str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        assert self.process.stdin and self.process.stdout
        self._in: BinaryIO = self.process.stdin
        self._out: BinaryIO = self.process.stdout
        # The simulation is a long-running child and can emit Go diagnostics at
        # any time.  Leaving stderr unread eventually fills the OS pipe and
        # makes a healthy child block inside a tick, indistinguishable from a
        # hung rollout.  Keep a bounded tail for a useful protocol error while
        # continuously draining the pipe.
        self._stderr_tail: deque[bytes] = deque()
        self._stderr_size = 0
        self._stderr_lock = threading.Lock()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="assaultenv-stderr", daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        pipe = self.process.stderr
        if pipe is None:
            return
        try:
            read = getattr(pipe, "read1", pipe.read)
            while chunk := read(4096):
                with self._stderr_lock:
                    self._stderr_tail.append(chunk)
                    self._stderr_size += len(chunk)
                    while self._stderr_size > STDERR_TAIL_BYTES and self._stderr_tail:
                        self._stderr_size -= len(self._stderr_tail.popleft())
        except (OSError, ValueError):
            # Shutdown closes the pipe concurrently with this daemon reader.
            pass

    def _stderr_text(self) -> str:
        with self._stderr_lock:
            return b"".join(self._stderr_tail).decode(errors="replace")

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
        self._write(COMMAND_RESET, payload)
        return self._read_result()

    def step(self, actions: Iterable[HeroAction]) -> StepResult:
        if isinstance(actions, np.ndarray):
            values = np.asarray(actions)
            dagger = self.protocol_version == AI42_DAGGER_PROTOCOL_VERSION
            controlled = self.protocol_version == AI42_EVALUATION_PROTOCOL_VERSION or dagger
            columns = 6 if dagger else (5 if controlled else 4)
            if values.shape != (HERO_COUNT, columns):
                raise ValueError(f"action array must have shape (10, {columns})")
            packed = np.empty(
                HERO_COUNT, dtype=DAGGER_ACTION_DTYPE if dagger else (CONTROLLED_ACTION_DTYPE if controlled else ACTION_DTYPE),
            )
            offset = 0
            if controlled:
                packed["control"] = values[:, 0]
                offset = 1
            packed["kind"] = values[:, offset]
            packed["target"] = values[:, offset + 1]
            packed["direction"] = values[:, offset + 2]
            packed["distance"] = values[:, offset + 3]
            if dagger:
                packed["intervention"] = values[:, offset + 4]
            payload = packed.tobytes()
            self._write(COMMAND_STEP, payload)
            return self._read_result()
        if self.protocol_version in (AI42_EVALUATION_PROTOCOL_VERSION, AI42_DAGGER_PROTOCOL_VERSION):
            columns = 6 if self.protocol_version == AI42_DAGGER_PROTOCOL_VERSION else 5
            raise TypeError(f"AI-42 controlled actions must be a (10, {columns}) NumPy array")
        values = list(actions)
        if len(values) != HERO_COUNT:
            raise ValueError("actions must contain exactly 10 HeroAction values")
        payload = b"".join(struct.pack("<B H B B", a.kind, a.target, a.direction, a.distance) for a in values)
        self._write(COMMAND_STEP, payload)
        return self._read_result()

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_pipes()
            return
        try:
            self._write(COMMAND_CLOSE, b"")
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
        body = MAGIC + struct.pack("<HH", self.protocol_version, command) + payload
        self._in.write(struct.pack("<I", len(body)) + body)
        self._in.flush()

    def _read_exact(self, size: int, field: str = "frame.body", offset: int = 0) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._out.read(size - len(chunks))
            if not chunk:
                error = self._stderr_text()
                raise AssaultProtocolError(
                    f"field={field} offset={offset + len(chunks)}: truncated, "
                    f"got {len(chunks)} of {size} bytes; {error}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_result(self) -> StepResult:
        size = struct.unpack("<I", self._read_exact(4, "frame.length"))[0]
        if size < 8 or size > MAX_FRAME_SIZE:
            raise AssaultProtocolError(f"field=frame.length offset=0: invalid frame length {size}")
        body = self._read_exact(size, "frame.body")
        if body[:4] != MAGIC:
            raise AssaultProtocolError("field=frame.magic offset=0: bad response magic")
        version, response = struct.unpack_from("<HH", body, 4)
        if version != self.protocol_version:
            raise AssaultProtocolError(
                f"field=frame.version offset=4: got {version}, expected {self.protocol_version}"
            )
        if response == RESPONSE_ERROR:
            raise AssaultProtocolError(body[8:].decode(errors="replace"))
        if response != RESPONSE_RESULT:
            raise AssaultProtocolError(f"field=frame.response offset=6: unknown response {response}")
        layout = _result_layout(self.protocol_version)
        size_error = _layout_size_error(layout, len(body))
        if size_error is not None:
            raise size_error
        if body[8:40] != self.schema_hash:
            raise AssaultProtocolError("field=result.schema_hash offset=8: schema hash mismatch")
        if body[40:72] != self.reward_hash:
            raise AssaultProtocolError("field=result.reward_hash offset=40: reward hash mismatch")
        step, elapsed, done, winner = struct.unpack_from("<IfB3xi", body, 72)
        offset = 88
        invalid = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
        offset += HERO_COUNT
        rewards = np.frombuffer(body, dtype="<f4", count=HERO_COUNT, offset=offset)
        offset += HERO_COUNT * 4
        observations = np.frombuffer(
            body, dtype=self.observation_dtype, count=HERO_COUNT, offset=offset,
        )
        hero = observations["hero"]
        entities = observations["entities"]
        global_state = observations["global_state"]
        entity_mask = observations["entity_mask"]
        kind_mask = observations["kind_mask"]
        target_mask = observations["target_mask"]
        skill_target_mask = observations["skill_target_mask"]
        abilities = observations["abilities"] if "abilities" in observations.dtype.names else None
        offset += HERO_COUNT * self.observation_dtype.itemsize
        teacher_actions = teacher_valid = None
        teacher_intent = teacher_status = None
        executed_actions = executed_valid = rejection_reason = None
        active_order = None
        if self.protocol_version == AI41_TEACHER_PROTOCOL_VERSION:
            teacher_actions = np.frombuffer(body, dtype=ACTION_DTYPE, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT * ACTION_DTYPE.itemsize
            teacher_valid = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT
        elif _layout_is_ai42(self.protocol_version):
            teacher_intent = np.frombuffer(body, dtype=ACTION_DTYPE, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT * ACTION_DTYPE.itemsize
            teacher_status = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT
            executed_actions = np.frombuffer(body, dtype=ACTION_DTYPE, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT * ACTION_DTYPE.itemsize
            executed_valid = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT
            rejection_reason = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
            offset += HERO_COUNT
            if self.protocol_version == AI42_DAGGER_PROTOCOL_VERSION:
                active_order = np.frombuffer(body, dtype=np.uint8, count=HERO_COUNT, offset=offset)
                offset += HERO_COUNT
        elif self.protocol_version == AI42_EVALUATION_PROTOCOL_VERSION:
            active_order = np.frombuffer(
                body, dtype=np.uint8, count=HERO_COUNT, offset=offset,
            )
            offset += HERO_COUNT
        if offset != len(body):
            raise AssaultProtocolError(f"field=result.trailing offset={offset}: got {len(body)-offset} bytes")
        return StepResult(step, elapsed, bool(done), winner, invalid, rewards, hero, entities,
                          global_state, entity_mask, kind_mask, target_mask, skill_target_mask,
                          abilities, teacher_actions, teacher_valid, teacher_intent,
                          teacher_status, executed_actions, executed_valid, rejection_reason,
                          bytes(body[8:40]), bytes(body[40:72]), active_order)


def _reset_values(
    seed: int,
    max_steps: int,
    roster: Iterable[int] | None,
    controllers: Iterable[int] | None,
) -> tuple[int, ...]:
    roster_values = tuple(roster or [0] * HERO_COUNT)
    controller_values = tuple(controllers or [0] * HERO_COUNT)
    if len(roster_values) != HERO_COUNT or len(controller_values) != HERO_COUNT:
        raise ValueError("roster and controllers must contain exactly 10 values")
    return (seed, max_steps, *roster_values, *controller_values)


class AssaultVectorProcess:
    """One Go process that owns and advances a fixed vector of matches."""

    def __init__(self, executable: str | os.PathLike[str], workers: int,
                 protocol_version: int = PROTOCOL_VERSION):
        if workers < 1:
            raise ValueError("workers must be positive")
        _validate_runtime_protocol(protocol_version)
        self.executable = Path(executable)
        self.workers = workers
        self.protocol_version = protocol_version
        self.schema_hash = _protocol_schema_hash(protocol_version)
        self.reward_hash = _protocol_reward_hash(protocol_version)
        self.process = subprocess.Popen(
            [str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        assert self.process.stdin and self.process.stdout
        self._in: BinaryIO = self.process.stdin
        self._out: BinaryIO = self.process.stdout
        # The first full reset allocates the maximum response buffer. Later
        # full-step responses reuse it without resizing exported NumPy views.
        self._result_buffer = bytearray()
        # Partial resets cannot overwrite the full response: observations for
        # matches that did not reset still reference the previous full buffer.
        self._partial_result_buffer = bytearray()
        # Vector workers are long-lived too; continuously drain diagnostics so
        # a noisy Go child cannot fill stderr and block its next response.
        self._stderr_tail: deque[bytes] = deque()
        self._stderr_size = 0
        self._stderr_lock = threading.Lock()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="assaultenv-vector-stderr", daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        pipe = self.process.stderr
        if pipe is None:
            return
        try:
            read = getattr(pipe, "read1", pipe.read)
            while chunk := read(4096):
                with self._stderr_lock:
                    self._stderr_tail.append(chunk)
                    self._stderr_size += len(chunk)
                    while self._stderr_size > STDERR_TAIL_BYTES and self._stderr_tail:
                        self._stderr_size -= len(self._stderr_tail.popleft())
        except (OSError, ValueError):
            # Shutdown closes the pipe concurrently with this daemon reader.
            pass

    def _stderr_text(self) -> str:
        with self._stderr_lock:
            return b"".join(self._stderr_tail).decode(errors="replace")

    def _write(self, command: int, payload: bytes | bytearray | memoryview) -> None:
        body_size = 8 + len(payload)
        self._in.write(struct.pack("<I4sHH", body_size, MAGIC, self.protocol_version, command))
        if payload:
            self._in.write(payload)
        self._in.flush()

    def _read_exact(self, size: int, field: str = "frame.body", offset: int = 0) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._out.read(size - len(chunks))
            if not chunk:
                error = self._stderr_text()
                raise AssaultProtocolError(
                    f"field={field} offset={offset + len(chunks)}: truncated, "
                    f"got {len(chunks)} of {size} bytes; {error}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_vector_results(self, expected: int, partial: bool = False) -> VectorStepResults:
        size = struct.unpack("<I", self._read_exact(4, "frame.length"))[0]
        if size < 8 or size > MAX_FRAME_SIZE:
            raise AssaultProtocolError(f"field=frame.length offset=0: invalid frame length {size}")
        buffer = self._partial_result_buffer if partial else self._result_buffer
        if size > len(buffer):
            buffer = bytearray(size)
            if partial:
                self._partial_result_buffer = buffer
            else:
                self._result_buffer = buffer
        view = memoryview(buffer)[:size]
        offset = 0
        while offset < size:
            read = self._out.readinto(view[offset:])
            if not read:
                raise AssaultProtocolError(
                    f"field=frame.body offset={offset}: truncated, got {offset} of {size} bytes"
                )
            offset += read
        if size < 8 or view[:4] != MAGIC:
            raise AssaultProtocolError("field=frame.magic offset=0: bad vector response magic")
        version, response = struct.unpack_from("<HH", view, 4)
        if version != self.protocol_version:
            raise AssaultProtocolError(
                f"field=frame.version offset=4: got {version}, expected {self.protocol_version}"
            )
        if response == RESPONSE_ERROR:
            raise AssaultProtocolError(bytes(view[8:]).decode(errors="replace"))
        if response != RESPONSE_VECTOR_RESULT:
            raise AssaultProtocolError(f"field=frame.response offset=6: unknown response {response}")
        if size < VECTOR_RESULT_HEADER_SIZE:
            raise AssaultProtocolError(
                f"field=result.count offset=72: truncated, got {size} bytes, need {VECTOR_RESULT_HEADER_SIZE}"
            )
        count = struct.unpack_from("<I", view, 72)[0]
        if count < 1:
            raise AssaultProtocolError(f"field=result.count offset=72: invalid count {count}")
        layout = _result_layout(self.protocol_version, count, vector=True)
        size_error = _layout_size_error(layout, size)
        if size_error is not None:
            raise size_error
        if view[8:40] != self.schema_hash or view[40:72] != self.reward_hash:
            if view[8:40] != self.schema_hash:
                raise AssaultProtocolError("field=result.schema_hash offset=8: schema hash mismatch")
            raise AssaultProtocolError("field=result.reward_hash offset=40: reward hash mismatch")
        if count != expected:
            raise AssaultProtocolError(f"field=result.count offset=72: got {count}, expected {expected}")
        offset = VECTOR_RESULT_HEADER_SIZE

        def take(dtype, shape, field):
            nonlocal offset
            data_type = np.dtype(dtype)
            elements = int(np.prod(shape))
            size_in_bytes = elements * data_type.itemsize
            if offset + size_in_bytes > size:
                raise AssaultProtocolError(
                    f"field={field} offset={offset}: truncated, need {offset + size_in_bytes}, frame size {size}"
                )
            result = np.frombuffer(
                buffer, dtype=data_type, count=elements, offset=offset,
            ).reshape(shape)
            offset += size_in_bytes
            return result

        actors = count * HERO_COUNT
        steps = take("<u4", (count,), "result.steps")
        elapsed = take("<f4", (count,), "result.elapsed")
        done = take("u1", (count,), "result.done")
        winners = take("<i4", (count,), "result.winner")
        invalid = take("u1", (count, HERO_COUNT), "result.invalid")
        rewards = take("<f4", (count, HERO_COUNT), "result.reward")
        batched = {"hero": take("<f4", (actors, HERO_FEATURES), "result.hero")}
        if _protocol_has_abilities(self.protocol_version):
            batched["abilities"] = take(
                "<f4", (actors, ABILITY_COUNT, ABILITY_FEATURES), "result.abilities"
            )
        batched.update({
            "entities": take("<f4", (actors, MAX_ENTITIES, ENTITY_FEATURES), "result.entities"),
            "global_state": take("<f4", (actors, GLOBAL_FEATURES), "result.global"),
            "entity_mask": take("u1", (actors, MAX_ENTITIES), "result.entity_mask"),
            "kind_mask": take("u1", (actors, ACTION_KINDS), "result.kind_mask"),
            "target_mask": take("u1", (actors, MAX_ENTITIES), "result.target_mask"),
            "skill_target_mask": take(
                "u1", (actors, ABILITY_COUNT, MAX_ENTITIES), "result.skill_target_mask"
            ),
        })
        if self.protocol_version == AI41_TEACHER_PROTOCOL_VERSION:
            batched["teacher_actions"] = take(ACTION_DTYPE, (actors,), "result.teacher_actions")
            batched["teacher_valid"] = take("u1", (actors,), "result.teacher_valid")
        elif _layout_is_ai42(self.protocol_version):
            batched["teacher_intent"] = take(ACTION_DTYPE, (actors,), "result.teacher_intent")
            batched["teacher_status"] = take("u1", (actors,), "result.teacher_status")
            batched["executed_actions"] = take(ACTION_DTYPE, (actors,), "result.executed_actions")
            batched["executed_valid"] = take("u1", (actors,), "result.executed_valid")
            batched["rejection_reason"] = take("u1", (actors,), "result.rejection_reason")
            if self.protocol_version == AI42_DAGGER_PROTOCOL_VERSION:
                batched["active_order"] = take("u1", (actors,), "result.active_order")
        elif self.protocol_version == AI42_EVALUATION_PROTOCOL_VERSION:
            batched["active_order"] = take("u1", (actors,), "result.active_order")
        if offset != size:
            raise AssaultProtocolError(f"field=result.trailing offset={offset}: got {size-offset} bytes")
        results: list[StepResult] = []
        for index in range(count):
            start, end = index * HERO_COUNT, (index + 1) * HERO_COUNT
            results.append(StepResult(
                int(steps[index]), float(elapsed[index]), bool(done[index]),
                int(winners[index]), invalid[index], rewards[index],
                batched["hero"][start:end], batched["entities"][start:end],
                batched["global_state"][start:end], batched["entity_mask"][start:end],
                batched["kind_mask"][start:end], batched["target_mask"][start:end],
                batched["skill_target_mask"][start:end],
                batched["abilities"][start:end] if "abilities" in batched else None,
                batched["teacher_actions"][start:end] if "teacher_actions" in batched else None,
                batched["teacher_valid"][start:end] if "teacher_valid" in batched else None,
                batched["teacher_intent"][start:end] if "teacher_intent" in batched else None,
                batched["teacher_status"][start:end] if "teacher_status" in batched else None,
                batched["executed_actions"][start:end] if "executed_actions" in batched else None,
                batched["executed_valid"][start:end] if "executed_valid" in batched else None,
                batched["rejection_reason"][start:end] if "rejection_reason" in batched else None,
                bytes(view[8:40]), bytes(view[40:72]),
                batched["active_order"][start:end] if "active_order" in batched else None,
            ))
        return VectorStepResults(results, batched)

    def reset(
        self,
        seeds: Iterable[int],
        max_steps: int,
        rosters: Iterable[Iterable[int] | None],
        controller_sets: Iterable[Iterable[int] | None],
    ) -> list[StepResult]:
        seed_values = list(seeds)
        roster_values = list(rosters)
        controller_values = list(controller_sets)
        if not (len(seed_values) == len(roster_values) == len(controller_values) == self.workers):
            raise ValueError("one reset tuple is required per vector worker")
        payload = bytearray(4 + self.workers * RESET_PAYLOAD_SIZE)
        struct.pack_into("<I", payload, 0, self.workers)
        offset = 4
        for seed, roster, controllers in zip(seed_values, roster_values, controller_values):
            struct.pack_into(
                "<qI10i10B", payload, offset,
                *_reset_values(seed, max_steps, roster, controllers),
            )
            offset += RESET_PAYLOAD_SIZE
        self._write(COMMAND_VECTOR_RESET, payload)
        return self._read_vector_results(self.workers)

    def reset_indices(
        self,
        indices: Iterable[int],
        seeds: Iterable[int],
        max_steps: int,
        rosters: Iterable[Iterable[int] | None],
        controller_sets: Iterable[Iterable[int] | None],
    ) -> dict[int, StepResult]:
        pairs = list(zip(indices, seeds, rosters, controller_sets))
        if not pairs:
            return {}
        if len({int(index) for index, *_ in pairs}) != len(pairs):
            raise ValueError("reset indices must be unique")
        item_size = 4 + RESET_PAYLOAD_SIZE
        payload = bytearray(4 + len(pairs) * item_size)
        struct.pack_into("<I", payload, 0, len(pairs))
        offset = 4
        for index, seed, roster, controllers in pairs:
            if not 0 <= int(index) < self.workers:
                raise IndexError(f"vector worker index {index} out of range")
            struct.pack_into("<I", payload, offset, int(index))
            struct.pack_into(
                "<qI10i10B", payload, offset + 4,
                *_reset_values(seed, max_steps, roster, controllers),
            )
            offset += item_size
        self._write(COMMAND_VECTOR_RESET_INDICES, payload)
        results = self._read_vector_results(len(pairs), partial=True)
        return {int(pair[0]): result for pair, result in zip(pairs, results)}

    def step(self, actions: Iterable[Iterable[HeroAction]] | np.ndarray) -> list[StepResult]:
        dagger = self.protocol_version == AI42_DAGGER_PROTOCOL_VERSION
        controlled = self.protocol_version == AI42_EVALUATION_PROTOCOL_VERSION or dagger
        columns = 6 if dagger else (5 if controlled else 4)
        if isinstance(actions, np.ndarray):
            values = np.asarray(actions)
            if values.shape != (self.workers, HERO_COUNT, columns):
                raise ValueError(
                    f"action array must have shape ({self.workers}, {HERO_COUNT}, {columns})"
                )
        else:
            if controlled:
                raise TypeError(
                    f"AI-42 controlled actions must be a (workers, 10, {columns}) NumPy array"
                )
            workers = [list(worker_actions) for worker_actions in actions]
            if len(workers) != self.workers or any(len(worker) != HERO_COUNT for worker in workers):
                raise ValueError("one ten-hero action batch is required per vector worker")
            values = np.asarray([
                [[action.kind, action.target, action.direction, action.distance] for action in worker]
                for worker in workers
            ])
        packed = np.empty(
            (self.workers, HERO_COUNT),
            dtype=DAGGER_ACTION_DTYPE if dagger else (CONTROLLED_ACTION_DTYPE if controlled else ACTION_DTYPE),
        )
        offset = 0
        if controlled:
            packed["control"] = values[:, :, 0]
            offset = 1
        packed["kind"] = values[:, :, offset]
        packed["target"] = values[:, :, offset + 1]
        packed["direction"] = values[:, :, offset + 2]
        packed["distance"] = values[:, :, offset + 3]
        if dagger:
            packed["intervention"] = values[:, :, offset + 4]
        payload = bytearray(4 + packed.nbytes)
        struct.pack_into("<I", payload, 0, self.workers)
        memoryview(payload)[4:] = packed.view(np.uint8).reshape(-1)
        self._write(COMMAND_VECTOR_STEP, payload)
        return self._read_vector_results(self.workers)

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_pipes()
            return
        try:
            self._write(COMMAND_CLOSE, b"")
            self.process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


class AssaultVectorEnv:
    """Synchronous vector backed by one batched Go rollout process."""

    def __init__(self, executable: str | os.PathLike[str], workers: int,
                 protocol_version: int = PROTOCOL_VERSION):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self._process = AssaultVectorProcess(executable, workers, protocol_version)

    def reset(
        self, seeds: Iterable[int], max_steps: int = 4_500,
        controllers: Iterable[int] | None = None,
        controller_sets: Iterable[Iterable[int]] | None = None,
        rosters: Iterable[Iterable[int]] | None = None,
    ) -> list[StepResult]:
        values = list(seeds)
        if len(values) != self.workers:
            raise ValueError("one seed is required per worker")
        if controllers is not None and controller_sets is not None:
            raise ValueError("pass controllers or controller_sets, not both")
        default_controllers = None if controllers is None else list(controllers)
        controller_values = ([default_controllers] * self.workers if controller_sets is None
                             else [list(worker_controllers) for worker_controllers in controller_sets])
        if len(controller_values) != self.workers:
            raise ValueError("one controller set is required per worker")
        roster_values = ([None] * self.workers if rosters is None
                         else [list(roster) for roster in rosters])
        if len(roster_values) != self.workers:
            raise ValueError("one roster is required per worker")
        return self._process.reset(values, max_steps, roster_values, controller_values)

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
        return self._process.reset_indices(
            [index for index, _ in pairs], [seed for _, seed in pairs], max_steps,
            roster_values, controller_values,
        )

    def step(self, actions: Iterable[Iterable[HeroAction]]) -> list[StepResult]:
        if isinstance(actions, np.ndarray):
            values = np.asarray(actions)
        else:
            values = [list(worker_actions) for worker_actions in actions]
        if len(values) != self.workers:
            raise ValueError("one action batch is required per worker")
        return self._process.step(values)

    def close(self) -> None:
        self._process.close()

    def __enter__(self) -> "AssaultVectorEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
