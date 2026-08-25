"""Frozen AI-42 behavior-cloning class-balance profiles.

Profiles are deliberately data artifacts rather than learner configuration.
They are computed from the ordered training match stream once, contain no
validation examples, and are immutable after construction.  The supervision
adapter is imported from :mod:`learner_ai42`, so profile counts cannot drift
from the masks used by the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from types import MappingProxyType

import torch

from .learner_ai42 import (
    AI42Batch,
    AI42LearnerError,
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    prepare_ai42_supervision,
    class_balance_weights,
    iter_ai42_dataset_batches,
)


PROFILE_FORMAT = "AI42-bc-class-profile-v2"
PROFILE_VERSION = PROFILE_FORMAT
SUPERVISION_VERSION = "AI42-supervision-v1"
PROTOCOL_VERSION = 13
CLASS_BALANCE_POWER = 0.5
PROFILE_HEADS = HEAD_NAMES


class AI42ProfileError(AI42LearnerError):
    """Raised for malformed, tampered, or incompatible class profiles."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AI42ProfileError(f"profile is not canonical JSON: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise AI42ProfileError(f"{name} must be a lower-case SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise AI42ProfileError(f"{name} must be a lower-case SHA-256 hexadecimal digest") from exc
    return value


def ordered_train_id_hash(match_ids: Sequence[str]) -> str:
    """Hash the exact ordered train-ID list used to build a profile."""

    ids = list(match_ids)
    return hashlib.sha256(_canonical_json(ids)).hexdigest()


def _unsigned_payload(
    *,
    dataset_manifest_hash: str,
    train_match_ids: Sequence[str],
    dataset_schema_version: str,
    shard_schema_version: str,
    counts: Mapping[str, Sequence[int]],
    weights: Mapping[str, Sequence[float]],
    class_balance_power: float,
) -> dict[str, Any]:
    return {
        "format": PROFILE_FORMAT,
        "profile_version": PROFILE_VERSION,
        "supervision_version": SUPERVISION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_schema_version": dataset_schema_version,
        "shard_schema_version": shard_schema_version,
        "dataset_manifest_hash": dataset_manifest_hash,
        "train_match_ids": list(train_match_ids),
        "train_match_ids_hash": ordered_train_id_hash(train_match_ids),
        "class_balance_power": float(class_balance_power),
        "counts": {head: list(counts[head]) for head in PROFILE_HEADS},
        "weights": {head: list(weights[head]) for head in PROFILE_HEADS},
    }


def _profile_hash(unsigned: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(unsigned))).hexdigest()


def _validate_counts_weights(counts: Any, weights: Any, power: float) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[float, ...]]]:
    if not isinstance(counts, Mapping) or not isinstance(weights, Mapping):
        raise AI42ProfileError("profile counts and weights must be mappings")
    if set(counts) != set(PROFILE_HEADS) or set(weights) != set(PROFILE_HEADS):
        raise AI42ProfileError("profile counts/weights must contain exactly the five AI-42 heads")
    normalized_counts: dict[str, tuple[int, ...]] = {}
    normalized_weights: dict[str, tuple[float, ...]] = {}
    for head in PROFILE_HEADS:
        expected = HEAD_CLASS_COUNTS[head]
        raw_counts = counts[head]
        raw_weights = weights[head]
        if not isinstance(raw_counts, (list, tuple)) or len(raw_counts) != expected:
            raise AI42ProfileError(f"profile counts[{head!r}] has the wrong shape")
        if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != expected:
            raise AI42ProfileError(f"profile weights[{head!r}] has the wrong shape")
        checked_counts: list[int] = []
        checked_weights: list[float] = []
        for index, value in enumerate(raw_counts):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AI42ProfileError(f"profile counts[{head!r}][{index}] is invalid")
            checked_counts.append(value)
        present = [count for count in checked_counts if count > 0]
        for index, value in enumerate(raw_weights):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AI42ProfileError(f"profile weights[{head!r}][{index}] is invalid")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise AI42ProfileError(f"profile weights[{head!r}][{index}] is non-finite or negative")
            if checked_counts[index] == 0 and number != 0.0:
                raise AI42ProfileError(f"profile weight for absent {head} class {index} must be zero")
            if checked_counts[index] > 0 and number <= 0.0:
                raise AI42ProfileError(f"profile weight for supported {head} class {index} must be positive")
            checked_weights.append(number)
        if present:
            mean = sum(checked_weights) / len(present)
            if not math.isfinite(mean) or not math.isclose(mean, 1.0, rel_tol=2e-6, abs_tol=2e-6):
                raise AI42ProfileError(f"profile weights[{head!r}] are not mean-one over supported classes")
        elif any(checked_weights):
            raise AI42ProfileError(f"profile weights[{head!r}] must be all zero when the head is absent")
        expected_weights = class_balance_weights(checked_counts, power).detach().cpu().tolist()
        for index, (actual, expected) in enumerate(zip(checked_weights, expected_weights)):
            if not math.isclose(actual, float(expected), rel_tol=1e-6, abs_tol=1e-7):
                raise AI42ProfileError(f"profile weights[{head!r}][{index}] do not match the frozen class counts")
        normalized_counts[head] = tuple(checked_counts)
        normalized_weights[head] = tuple(checked_weights)
    return normalized_counts, normalized_weights


def _validate_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "format", "profile_version", "supervision_version", "protocol_version",
        "dataset_schema_version", "shard_schema_version", "dataset_manifest_hash",
        "train_match_ids", "train_match_ids_hash", "class_balance_power",
        "counts", "weights", "profile_hash",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise AI42ProfileError("profile field set is incomplete or contains unknown fields")
    if value["format"] != PROFILE_FORMAT or value["profile_version"] != PROFILE_VERSION:
        raise AI42ProfileError("profile format/version is incompatible")
    if value["supervision_version"] != SUPERVISION_VERSION or value["protocol_version"] != PROTOCOL_VERSION:
        raise AI42ProfileError("profile supervision/protocol version is incompatible")
    for field in ("dataset_schema_version", "shard_schema_version"):
        if not isinstance(value[field], str) or not value[field]:
            raise AI42ProfileError(f"profile.{field} must be a non-empty string")
    dataset_hash = _validate_hash(value["dataset_manifest_hash"], "profile.dataset_manifest_hash")
    ids = value["train_match_ids"]
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        raise AI42ProfileError("profile.train_match_ids must be a non-empty string list")
    if len(set(ids)) != len(ids):
        raise AI42ProfileError("profile.train_match_ids must be unique")
    id_hash = _validate_hash(value["train_match_ids_hash"], "profile.train_match_ids_hash")
    if id_hash != ordered_train_id_hash(ids):
        raise AI42ProfileError("profile train-match ID hash does not match its ordered IDs")
    power = value["class_balance_power"]
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(float(power)):
        raise AI42ProfileError("profile.class_balance_power must be finite")
    if float(power) != CLASS_BALANCE_POWER:
        raise AI42ProfileError("AI-42 BC-v2 requires class_balance_power=0.5")
    counts, weights = _validate_counts_weights(value["counts"], value["weights"], float(power))
    unsigned = _unsigned_payload(
        dataset_manifest_hash=dataset_hash,
        train_match_ids=ids,
        dataset_schema_version=value["dataset_schema_version"],
        shard_schema_version=value["shard_schema_version"],
        counts=counts,
        weights=weights,
        class_balance_power=float(power),
    )
    if value["profile_hash"] != _profile_hash(unsigned):
        raise AI42ProfileError("profile_hash does not match profile contents")
    return {
        **unsigned,
        "profile_hash": value["profile_hash"],
    }


@dataclass(frozen=True, slots=True)
class AI42ClassBalanceProfile:
    """Immutable, canonical class counts and weights for one train split."""

    dataset_manifest_hash: str
    train_match_ids: tuple[str, ...]
    dataset_schema_version: str
    shard_schema_version: str
    counts: Mapping[str, tuple[int, ...]]
    weights: Mapping[str, tuple[float, ...]]
    class_balance_power: float = CLASS_BALANCE_POWER
    format: str = PROFILE_FORMAT
    profile_version: str = PROFILE_VERSION
    supervision_version: str = SUPERVISION_VERSION
    protocol_version: int = PROTOCOL_VERSION
    train_match_ids_hash: str = ""
    profile_hash: str = ""

    def __post_init__(self) -> None:
        try:
            if isinstance(self.train_match_ids, (str, bytes)):
                raise AI42ProfileError("train_match_ids must be a sequence, not a string")
            ids = tuple(self.train_match_ids)
            counts = {head: tuple(self.counts[head]) for head in PROFILE_HEADS}
            weights = {head: tuple(self.weights[head]) for head in PROFILE_HEADS}
            power = float(self.class_balance_power)
            ids_hash = self.train_match_ids_hash or ordered_train_id_hash(ids)
            unsigned = _unsigned_payload(
                dataset_manifest_hash=self.dataset_manifest_hash,
                train_match_ids=ids,
                dataset_schema_version=self.dataset_schema_version,
                shard_schema_version=self.shard_schema_version,
                counts=counts,
                weights=weights,
                class_balance_power=power,
            )
            payload = {
                "format": self.format,
                "profile_version": self.profile_version,
                "supervision_version": self.supervision_version,
                "protocol_version": self.protocol_version,
                "dataset_schema_version": self.dataset_schema_version,
                "shard_schema_version": self.shard_schema_version,
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "train_match_ids": list(ids),
                "train_match_ids_hash": ids_hash,
                "class_balance_power": power,
                "counts": {head: list(counts[head]) for head in PROFILE_HEADS},
                "weights": {head: list(weights[head]) for head in PROFILE_HEADS},
                "profile_hash": self.profile_hash or _profile_hash(unsigned),
            }
        except AI42ProfileError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AI42ProfileError(f"profile fields are malformed: {exc}") from exc
        normalized = _validate_payload(payload)
        object.__setattr__(self, "dataset_manifest_hash", normalized["dataset_manifest_hash"])
        object.__setattr__(self, "train_match_ids", tuple(normalized["train_match_ids"]))
        object.__setattr__(self, "dataset_schema_version", normalized["dataset_schema_version"])
        object.__setattr__(self, "shard_schema_version", normalized["shard_schema_version"])
        object.__setattr__(self, "counts", MappingProxyType({head: tuple(normalized["counts"][head]) for head in PROFILE_HEADS}))
        object.__setattr__(self, "weights", MappingProxyType({head: tuple(normalized["weights"][head]) for head in PROFILE_HEADS}))
        object.__setattr__(self, "class_balance_power", float(normalized["class_balance_power"]))
        object.__setattr__(self, "train_match_ids_hash", normalized["train_match_ids_hash"])
        object.__setattr__(self, "profile_hash", normalized["profile_hash"])

    @classmethod
    def from_batches(
        cls,
        batches: Iterable[AI42Batch],
        *,
        dataset_manifest_hash: str | None = None,
        train_match_ids: Sequence[str] | None = None,
        dataset_hash: str | None = None,
        train_ids: Sequence[str] | None = None,
        dataset_schema_version: str = "AI42-dataset-v2",
        shard_schema_version: str = "AI42-go-shard-v2",
    ) -> "AI42ClassBalanceProfile":
        supplied_hashes = [value for value in (dataset_manifest_hash, dataset_hash) if value is not None]
        if len(supplied_hashes) != 1:
            raise AI42ProfileError("provide exactly one dataset manifest hash")
        dataset_hash = _validate_hash(supplied_hashes[0], "dataset_manifest_hash")
        supplied_ids = [value for value in (train_match_ids, train_ids) if value is not None]
        if len(supplied_ids) != 1:
            raise AI42ProfileError("provide exactly one ordered train-ID sequence")
        ids = tuple(supplied_ids[0])
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids):
            raise AI42ProfileError("train_match_ids must be a unique non-empty string sequence")
        counts = {head: [0] * HEAD_CLASS_COUNTS[head] for head in PROFILE_HEADS}
        seen = 0
        for raw_batch in batches:
            batch = raw_batch if isinstance(raw_batch, AI42Batch) else AI42Batch.from_mapping(raw_batch)
            prepared = prepare_ai42_supervision(batch)
            for head in PROFILE_HEADS:
                labels = prepared.labels[head]
                active = prepared.active[head]
                if bool(active.any()):
                    values = torch.bincount(labels[active], minlength=HEAD_CLASS_COUNTS[head]).detach().cpu().tolist()
                    for index, value in enumerate(values[:HEAD_CLASS_COUNTS[head]]):
                        counts[head][index] += int(value)
            seen += int(prepared.active["control"].sum().detach().cpu().item())
        if seen < 1:
            raise AI42ProfileError("training stream contains no supported supervision")
        weights = {
            head: tuple(float(value) for value in class_balance_weights(counts[head], CLASS_BALANCE_POWER).tolist())
            for head in PROFILE_HEADS
        }
        return cls(
            dataset_manifest_hash=dataset_hash,
            train_match_ids=ids,
            dataset_schema_version=str(dataset_schema_version),
            shard_schema_version=str(shard_schema_version),
            counts={head: tuple(values) for head, values in counts.items()},
            weights=weights,
            class_balance_power=CLASS_BALANCE_POWER,
            train_match_ids_hash=ordered_train_id_hash(ids),
            profile_hash=_profile_hash(_unsigned_payload(
                dataset_manifest_hash=dataset_hash,
                train_match_ids=ids,
                dataset_schema_version=str(dataset_schema_version),
                shard_schema_version=str(shard_schema_version),
                counts=counts,
                weights=weights,
                class_balance_power=CLASS_BALANCE_POWER,
            )),
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        train_match_ids: Sequence[str] | None = None,
        train_ids: Sequence[str] | None = None,
        sequence_length: int = 64,
        batch_size: int = 8,
    ) -> "AI42ClassBalanceProfile":
        if train_match_ids is not None and train_ids is not None:
            raise AI42ProfileError("provide only one ordered train-ID sequence")
        selected_ids = train_match_ids if train_match_ids is not None else train_ids
        if selected_ids is not None:
            ids = tuple(selected_ids)
        elif hasattr(dataset, "match_ids") and callable(dataset.match_ids):
            ids = tuple(dataset.match_ids("train"))
        elif hasattr(dataset, "split_match_ids") and callable(dataset.split_match_ids):
            ids = tuple(dataset.split_match_ids()["train"])
        else:
            raise AI42ProfileError("dataset does not expose ordered train match IDs")
        if not ids:
            raise AI42ProfileError("dataset contains no training matches")
        class _OrderedView:
            manifest = getattr(dataset, "manifest", {})
            def iter_matches(self, split: str):
                if split != "train":
                    raise AI42ProfileError("profile counting view only supports train")
                if hasattr(dataset, "arrays_for_match"):
                    for match_id in ids:
                        yield match_id, dataset.arrays_for_match(match_id)
                else:
                    available = {str(key): value for key, value in dataset.iter_matches("train")}
                    for match_id in ids:
                        if match_id not in available:
                            raise AI42ProfileError(f"training match {match_id!r} is missing")
                        yield match_id, available[match_id]
        manifest = getattr(dataset, "manifest", {})
        return cls.from_batches(
            iter_ai42_dataset_batches(_OrderedView(), split="train", sequence_length=sequence_length, batch_size=batch_size),
            dataset_manifest_hash=str(getattr(dataset, "manifest_hash", manifest.get("manifest_hash", ""))),
            train_match_ids=ids,
            dataset_schema_version=str(manifest.get("dataset_schema_version", "AI42-dataset-v2")),
            shard_schema_version=str(manifest.get("shard_schema_version", "AI42-go-shard-v2")),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AI42ClassBalanceProfile":
        normalized = _validate_payload(value)
        return cls(
            dataset_manifest_hash=normalized["dataset_manifest_hash"],
            train_match_ids=tuple(normalized["train_match_ids"]),
            dataset_schema_version=normalized["dataset_schema_version"],
            shard_schema_version=normalized["shard_schema_version"],
            counts={head: tuple(normalized["counts"][head]) for head in PROFILE_HEADS},
            weights={head: tuple(normalized["weights"][head]) for head in PROFILE_HEADS},
            class_balance_power=normalized["class_balance_power"],
            format=normalized["format"], profile_version=normalized["profile_version"],
            supervision_version=normalized["supervision_version"], protocol_version=normalized["protocol_version"],
            train_match_ids_hash=normalized["train_match_ids_hash"], profile_hash=normalized["profile_hash"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> "AI42ClassBalanceProfile":
        try:
            value = json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AI42ProfileError(f"profile JSON is invalid: {exc}") from exc
        if not isinstance(value, Mapping):
            raise AI42ProfileError("profile JSON root must be an object")
        profile = cls.from_mapping(value)
        raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if raw != profile.to_json().encode("utf-8"):
            raise AI42ProfileError("profile JSON is not canonical")
        return profile

    def to_dict(self) -> dict[str, Any]:
        return _validate_payload({
            "format": self.format,
            "profile_version": self.profile_version,
            "supervision_version": self.supervision_version,
            "protocol_version": self.protocol_version,
            "dataset_schema_version": self.dataset_schema_version,
            "shard_schema_version": self.shard_schema_version,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "train_match_ids": list(self.train_match_ids),
            "train_match_ids_hash": self.train_match_ids_hash,
            "class_balance_power": self.class_balance_power,
            "counts": {head: list(self.counts[head]) for head in PROFILE_HEADS},
            "weights": {head: list(self.weights[head]) for head in PROFILE_HEADS},
            "profile_hash": self.profile_hash,
        })

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()).decode("utf-8")

    def class_weights(self) -> dict[str, tuple[float, ...]]:
        return {head: tuple(self.weights[head]) for head in PROFILE_HEADS}

    @property
    def dataset_hash(self) -> str:
        return self.dataset_manifest_hash

    @property
    def train_id_hash(self) -> str:
        return self.train_match_ids_hash

    @property
    def hash(self) -> str:
        return self.profile_hash

    def to_mapping(self) -> dict[str, Any]:
        return self.to_dict()


def save_ai42_profile(path: str | os.PathLike[str], profile: AI42ClassBalanceProfile) -> Path:
    destination = Path(path)
    payload = _canonical_json(profile.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def load_ai42_profile(path: str | os.PathLike[str]) -> AI42ClassBalanceProfile:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise AI42ProfileError(f"cannot read profile: {exc}") from exc
    return AI42ClassBalanceProfile.from_json(payload)


ClassBalanceProfile = AI42ClassBalanceProfile
AI42BCProfile = AI42ClassBalanceProfile
AI42ClassProfile = AI42ClassBalanceProfile
build_ai42_class_profile = AI42ClassBalanceProfile.from_dataset
build_ai42_profile = AI42ClassBalanceProfile.from_dataset
load_class_balance_profile = load_ai42_profile
save_class_balance_profile = save_ai42_profile


__all__ = [
    "AI42ClassBalanceProfile", "AI42ProfileError", "CLASS_BALANCE_POWER", "PROFILE_FORMAT",
    "PROFILE_HEADS", "PROFILE_VERSION", "SUPERVISION_VERSION", "ClassBalanceProfile", "AI42BCProfile", "AI42ClassProfile",
    "build_ai42_class_profile", "build_ai42_profile", "load_ai42_profile", "load_class_balance_profile",
    "ordered_train_id_hash", "save_ai42_profile", "save_class_balance_profile",
]
