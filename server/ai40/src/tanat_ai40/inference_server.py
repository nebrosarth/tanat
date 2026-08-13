from __future__ import annotations

import argparse
import struct
import sys

import numpy as np
import onnxruntime as ort

from .env import ACTION_KINDS, ENTITY_FEATURES, GLOBAL_FEATURES, HERO_FEATURES, MAX_ENTITIES

MAGIC = b"AI40"
VERSION = 1
INFER = 1
RESET = 2
OK = 100
ERROR = 255
OBS_FLOATS = HERO_FEATURES + MAX_ENTITIES * ENTITY_FEATURES + GLOBAL_FEATURES
OBS_BYTES = OBS_FLOATS * 4 + MAX_ENTITIES + ACTION_KINDS + MAX_ENTITIES + 4 * MAX_ENTITIES


def read_exact(size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def read_frame() -> tuple[int, bytes]:
    size = struct.unpack("<I", read_exact(4))[0]
    body = read_exact(size)
    if size < 8 or body[:4] != MAGIC:
        raise RuntimeError("invalid AI-40 frame")
    version, command = struct.unpack_from("<HH", body, 4)
    if version != VERSION:
        raise RuntimeError(f"protocol version {version}, expected {VERSION}")
    return command, body[8:]


def write_frame(response: int, payload: bytes = b"") -> None:
    body = MAGIC + struct.pack("<HH", VERSION, response) + payload
    sys.stdout.buffer.write(struct.pack("<I", len(body)) + body)
    sys.stdout.buffer.flush()


def parse_observation(payload: memoryview, offset: int):
    floats = np.frombuffer(payload, "<f4", OBS_FLOATS, offset).copy()
    offset += OBS_FLOATS * 4
    hero = floats[:HERO_FEATURES]
    entity_end = HERO_FEATURES + MAX_ENTITIES * ENTITY_FEATURES
    entities = floats[HERO_FEATURES:entity_end].reshape(MAX_ENTITIES, ENTITY_FEATURES)
    global_state = floats[entity_end:]
    entity_mask = np.frombuffer(payload, np.uint8, MAX_ENTITIES, offset).copy(); offset += MAX_ENTITIES
    kind_mask = np.frombuffer(payload, np.uint8, ACTION_KINDS, offset).copy(); offset += ACTION_KINDS
    target_mask = np.frombuffer(payload, np.uint8, MAX_ENTITIES, offset).copy(); offset += MAX_ENTITIES
    skill_mask = np.frombuffer(payload, np.uint8, 4 * MAX_ENTITIES, offset).reshape(4, MAX_ENTITIES).copy()
    offset += 4 * MAX_ENTITIES
    return (hero, entities, global_state, entity_mask, kind_mask, target_mask, skill_mask), offset


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.isfinite(logits).all():
        raise RuntimeError("model produced NaN/Inf logits")
    if not np.all(mask.any(axis=1)):
        raise RuntimeError("empty action mask")
    return np.where(mask, logits, -np.inf).argmax(axis=1)


class Server:
    def __init__(self, model: str, recurrent_size: int):
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.recurrent_size = recurrent_size
        self.state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._validate_and_warm_up()

    def _validate_and_warm_up(self) -> None:
        expected_inputs = {
            "hero": (HERO_FEATURES,), "entities": (MAX_ENTITIES, ENTITY_FEATURES),
            "global": (GLOBAL_FEATURES,), "entity_mask": (MAX_ENTITIES,),
            "h": (self.recurrent_size,), "c": (self.recurrent_size,),
        }
        inputs = {item.name: item for item in self.session.get_inputs()}
        if list(inputs) != list(expected_inputs):
            raise RuntimeError(f"model inputs {list(inputs)}, expected {list(expected_inputs)}")
        for name, tail in expected_inputs.items():
            shape = tuple(inputs[name].shape)
            if len(shape) != len(tail) + 1 or tuple(shape[1:]) != tail:
                raise RuntimeError(f"model input {name} shape {shape}, expected [batch, {tail}]")
        expected_outputs = ["kind", "target", "direction", "distance", "value", "next_h", "next_c"]
        if [item.name for item in self.session.get_outputs()] != expected_outputs:
            raise RuntimeError("model output names mismatch")
        zeros = {
            "hero": np.zeros((1, HERO_FEATURES), np.float32),
            "entities": np.zeros((1, MAX_ENTITIES, ENTITY_FEATURES), np.float32),
            "global": np.zeros((1, GLOBAL_FEATURES), np.float32),
            "entity_mask": np.ones((1, MAX_ENTITIES), np.float32),
            "h": np.zeros((1, self.recurrent_size), np.float32),
            "c": np.zeros((1, self.recurrent_size), np.float32),
        }
        outputs = self.session.run(None, zeros)
        expected_tails = ((ACTION_KINDS,), (MAX_ENTITIES,), (16,), (3,), (),
                          (self.recurrent_size,), (self.recurrent_size,))
        for name, value, tail in zip(expected_outputs, outputs, expected_tails):
            if value.shape != (1, *tail) or not np.isfinite(value).all():
                raise RuntimeError(f"model output {name} is invalid: shape={value.shape}")

    def infer(self, payload: bytes) -> bytes:
        count = payload[0]
        expected = 1 + count * (4 + OBS_BYTES)
        if count < 1 or count > 10 or len(payload) != expected:
            raise RuntimeError("invalid inference batch payload")
        ids, observations, offset = [], [], 1
        view = memoryview(payload)
        for _ in range(count):
            ids.append(struct.unpack_from("<i", view, offset)[0]); offset += 4
            observation, offset = parse_observation(view, offset)
            observations.append(observation)
        hero, entities, global_state, entity_mask, kind_mask, target_mask, skill_mask = (
            np.stack([obs[i] for obs in observations]) for i in range(7))
        h = np.stack([self.state.get(id_, (np.zeros(self.recurrent_size, np.float32),) * 2)[0] for id_ in ids])
        c = np.stack([self.state.get(id_, (np.zeros(self.recurrent_size, np.float32),) * 2)[1] for id_ in ids])
        outputs = self.session.run(None, {"hero": hero, "entities": entities, "global": global_state,
                                          "entity_mask": entity_mask.astype(np.float32), "h": h, "c": c})
        kind_logits, target_logits, direction_logits, distance_logits, value, next_h, next_c = outputs
        for tensor in (direction_logits, distance_logits, value, next_h, next_c):
            if not np.isfinite(tensor).all():
                raise RuntimeError("model produced NaN/Inf output")
        kinds = masked_argmax(kind_logits, kind_mask.astype(bool))
        effective_targets = target_mask.astype(bool).copy()
        for i, kind in enumerate(kinds):
            if 3 <= kind <= 6:
                effective_targets[i] = skill_mask[i, kind - 3].astype(bool)
            if not effective_targets[i].any():
                effective_targets[i, 0] = True
        targets = masked_argmax(target_logits, effective_targets)
        directions = direction_logits.argmax(axis=1)
        distances = distance_logits.argmax(axis=1)
        for i, id_ in enumerate(ids):
            self.state[id_] = (next_h[i].copy(), next_c[i].copy())
        return b"".join(struct.pack("<B H B B", int(kinds[i]), int(targets[i]),
                                    int(directions[i]), int(distances[i])) for i in range(count))

    def reset(self, payload: bytes) -> None:
        count = payload[0]
        if len(payload) != 1 + count * 4:
            raise RuntimeError("invalid reset payload")
        for i in range(count):
            self.state.pop(struct.unpack_from("<i", payload, 1 + i * 4)[0], None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--recurrent-size", type=int, required=True)
    args = parser.parse_args()
    server = Server(args.model, args.recurrent_size)
    write_frame(OK, b"ready")
    while True:
        try:
            command, payload = read_frame()
            if command == INFER:
                write_frame(OK, server.infer(payload))
            elif command == RESET:
                server.reset(payload)
                write_frame(OK)
            else:
                raise RuntimeError(f"unknown command {command}")
        except EOFError:
            return
        except Exception as exc:
            write_frame(ERROR, str(exc).encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
