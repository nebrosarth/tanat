"""Deterministic held-out metrics for the AI-42 behavior-cloning actor."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .learner_ai42 import (
    AI42Batch,
    AI42LearnerError,
    ATTACK_KIND,
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    NAVIGATION_OFFSETS,
    SKILL_KINDS,
    prepare_ai42_supervision,
)
from .model_ai42_actor import CONTROL_ISSUE


METRICS_FORMAT = "AI42-bc-metrics-v2"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AI42LearnerError(f"{name} is non-finite")
    return result


def _grid_distance(left: Tensor, right: Tensor) -> Tensor:
    left_row, left_col = torch.div(left, 9, rounding_mode="floor"), left.remainder(9)
    right_row, right_col = torch.div(right, 9, rounding_mode="floor"), right.remainder(9)
    return (left_row - right_row).abs() + (left_col - right_col).abs()


def _empty_head(classes: int) -> dict[str, Any]:
    return {
        "count": 0,
        "correct": 0,
        "micro_accuracy": 0.0,
        "micro_loss": 0.0,
        "unweighted_loss": 0.0,
        "weighted_numerator": 0.0,
        "weighted_denominator": 0.0,
        "unweighted_numerator": 0.0,
        "confusion_matrix": [[0 for _ in range(classes)] for _ in range(classes)],
        "per_class": {
            str(index): {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            for index in range(classes)
        },
        "supported_macro_f1": 0.0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
    }


class AI42MetricAccumulator:
    """Streaming accumulator; all results are ordinary deterministic JSON."""

    def __init__(self, *, class_weights: Mapping[str, Sequence[float]] | None = None) -> None:
        self.class_weights = class_weights or {}
        self.heads = {head: _empty_head(HEAD_CLASS_COUNTS[head]) for head in HEAD_NAMES}
        self._weighted_numerator_parts = {head: [] for head in HEAD_NAMES}
        self._weighted_denominator_parts = {head: [] for head in HEAD_NAMES}
        self._unweighted_numerator_parts = {head: [] for head in HEAD_NAMES}
        self.action_count = 0
        self.action_correct = 0
        self.offset_count = 0
        self.offset_top1 = 0
        self.offset_top5 = 0
        self.offset_distance = 0.0
        self.batches = 0

    def _weights(self, head: str, device: torch.device) -> Tensor:
        values = self.class_weights.get(head)
        if values is None:
            return torch.ones(HEAD_CLASS_COUNTS[head], dtype=torch.float32, device=device)
        result = torch.as_tensor(values, dtype=torch.float32, device=device)
        if tuple(result.shape) != (HEAD_CLASS_COUNTS[head],) or not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
            raise AI42LearnerError(f"metrics class weights for {head} are invalid")
        return result

    def _update_head(self, head: str, logits: Tensor, labels: Tensor, active: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        classes = HEAD_CLASS_COUNTS[head]
        if tuple(logits.shape) != (labels.shape[0], classes) or tuple(mask.shape) != tuple(labels.shape) + (classes,):
            raise AI42LearnerError(f"metrics {head} tensors have inconsistent shapes")
        if bool(active.any()):
            selected_labels = labels[active]
            selected_mask = mask[active]
            if bool((selected_labels < 0).any()) or bool((selected_labels >= classes).any()):
                raise AI42LearnerError(f"metrics {head} label is outside the model vocabulary")
            if not bool(selected_mask.gather(1, selected_labels[:, None]).squeeze(1).all()):
                raise AI42LearnerError(f"metrics {head} teacher label is excluded by its action mask")
            selected_logits = logits[active].masked_fill(~selected_mask, -torch.inf)
            prediction = selected_logits.argmax(dim=-1)
            losses = torch.nn.functional.cross_entropy(selected_logits, selected_labels, reduction="none")
            weights = self._weights(head, logits.device)[selected_labels]
            weighted_numerator = (losses.to(dtype=torch.float64) * weights.to(dtype=torch.float64)).sum()
            weighted_denominator = weights.to(dtype=torch.float64).sum()
            correct = prediction == selected_labels
            matrix = torch.zeros((classes, classes), dtype=torch.int64, device="cpu")
            for true, predicted in zip(selected_labels.detach().cpu().tolist(), prediction.detach().cpu().tolist()):
                matrix[int(true), int(predicted)] += 1
            destination = self.heads[head]
            destination["count"] += int(selected_labels.numel())
            destination["correct"] += int(correct.sum().detach().cpu().item())
            weighted_numerator_value = _finite(weighted_numerator.detach().cpu().item(), f"metrics {head} weighted numerator")
            weighted_denominator_value = _finite(weighted_denominator.detach().cpu().item(), f"metrics {head} weighted denominator")
            unweighted_numerator_value = _finite(losses.to(dtype=torch.float64).sum().detach().cpu().item(), f"metrics {head} unweighted numerator")
            self._weighted_numerator_parts[head].append(weighted_numerator_value)
            self._weighted_denominator_parts[head].append(weighted_denominator_value)
            self._unweighted_numerator_parts[head].append(unweighted_numerator_value)
            old = torch.as_tensor(destination["confusion_matrix"], dtype=torch.int64)
            destination["confusion_matrix"] = (old + matrix).tolist()
            return prediction, selected_labels
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    def update(self, batch: AI42Batch, outputs: Mapping[str, Tensor]) -> None:
        if not isinstance(batch, AI42Batch):
            batch = AI42Batch.from_mapping(batch)  # type: ignore[arg-type]
        prepared = prepare_ai42_supervision(batch)
        for name in HEAD_NAMES:
            if name not in outputs:
                raise AI42LearnerError(f"metrics output is missing {name}")
            if not isinstance(outputs[name], Tensor) or not bool(torch.isfinite(outputs[name]).all()):
                raise AI42LearnerError(f"metrics output {name} is non-finite or invalid")
        flat_labels = prepared.labels
        flat_active = prepared.active
        flat_masks = prepared.masks
        rows = torch.arange(flat_labels["kind"].shape[0], device=flat_labels["kind"].device)
        output_control = outputs["control"].reshape(-1, HEAD_CLASS_COUNTS["control"])
        output_kind = outputs["kind"].reshape(-1, HEAD_CLASS_COUNTS["kind"])
        output_target = outputs["target"].reshape(-1, HEAD_CLASS_COUNTS["kind"], HEAD_CLASS_COUNTS["target"])[rows, flat_labels["kind"]]
        output_offset = outputs["offset"].reshape(-1, HEAD_CLASS_COUNTS["kind"], NAVIGATION_OFFSETS)[rows, flat_labels["kind"]]
        output_anchor = outputs["anchor"].reshape(-1, HEAD_CLASS_COUNTS["kind"], HEAD_CLASS_COUNTS["anchor"])[rows, flat_labels["kind"]]
        for name, logits in (
            ("control", output_control), ("kind", output_kind), ("target", output_target),
            ("offset", output_offset), ("anchor", output_anchor),
        ):
            self._update_head(name, logits, flat_labels[name], flat_active[name], flat_masks[name])

        control_logits = output_control.masked_fill(~flat_masks["control"], -torch.inf)
        kind_logits = output_kind.masked_fill(~flat_masks["kind"], -torch.inf)
        control_prediction = control_logits.argmax(dim=-1)
        kind_prediction = kind_logits.argmax(dim=-1)
        action_rows = flat_active["kind"]
        if bool(action_rows.any()):
            target_required = flat_active["target"]
            if bool((target_required & ~flat_masks["target"].any(dim=-1)).any()):
                raise AI42LearnerError("metrics target has an empty applicable action mask")
            target_prediction = output_target.masked_fill(~flat_masks["target"], -torch.inf).argmax(dim=-1)
            offset_prediction = output_offset.argmax(dim=-1)
            anchor_prediction = output_anchor.argmax(dim=-1)
            correct = (control_prediction == CONTROL_ISSUE) & (kind_prediction == flat_labels["kind"])
            offset_required = flat_active["offset"]
            anchor_required = flat_active["anchor"]
            correct &= (~target_required | (target_prediction == flat_labels["target"]))
            correct &= (~offset_required | (offset_prediction == flat_labels["offset"]))
            correct &= (~anchor_required | (anchor_prediction == flat_labels["anchor"]))
            correct &= action_rows
            self.action_count += int(action_rows.sum().detach().cpu().item())
            self.action_correct += int(correct[action_rows].sum().detach().cpu().item())

        offset_rows = flat_active["offset"]
        if bool(offset_rows.any()):
            offset_logits_selected = output_offset[offset_rows]
            offset_labels = flat_labels["offset"][offset_rows]
            topk = offset_logits_selected.topk(k=min(5, NAVIGATION_OFFSETS), dim=-1).indices
            top1 = topk[:, 0]
            self.offset_count += int(offset_rows.sum().detach().cpu().item())
            self.offset_top1 += int((top1 == offset_labels).sum().detach().cpu().item())
            self.offset_top5 += int((topk == offset_labels[:, None]).any(dim=-1).sum().detach().cpu().item())
            self.offset_distance += _finite(
                _grid_distance(top1, offset_labels).float().sum().detach().cpu().item(),
                "metrics offset Manhattan distance",
            )
        self.batches += 1

    def _finalize_head(self, head: str) -> dict[str, Any]:
        result = self.heads[head]
        count = int(result["count"])
        result["weighted_numerator"] = math.fsum(self._weighted_numerator_parts[head])
        result["weighted_denominator"] = math.fsum(self._weighted_denominator_parts[head])
        result["unweighted_numerator"] = math.fsum(self._unweighted_numerator_parts[head])
        result["micro_accuracy"] = result["correct"] / count if count else 0.0
        weighted_denominator = float(result["weighted_denominator"])
        result["micro_loss"] = result["weighted_numerator"] / weighted_denominator if weighted_denominator else 0.0
        result["unweighted_loss"] = result["unweighted_numerator"] / count if count else 0.0
        result["accuracy"] = result["micro_accuracy"]
        result["loss"] = result["micro_loss"]
        matrix = result["confusion_matrix"]
        supported_f1: list[float] = []
        supported_recall: list[float] = []
        for index in range(HEAD_CLASS_COUNTS[head]):
            support = sum(matrix[index])
            predicted = sum(matrix[row][index] for row in range(HEAD_CLASS_COUNTS[head]))
            true_positive = matrix[index][index]
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            result["per_class"][str(index)] = {
                "support": int(support), "precision": precision, "recall": recall, "f1": f1,
            }
            if support:
                supported_f1.append(f1)
                supported_recall.append(recall)
        result["supported_macro_f1"] = math.fsum(supported_f1) / len(supported_f1) if supported_f1 else 0.0
        result["macro_f1"] = result["supported_macro_f1"]
        result["balanced_accuracy"] = math.fsum(supported_recall) / len(supported_recall) if supported_recall else 0.0
        return result

    def to_dict(self) -> dict[str, Any]:
        heads = {head: self._finalize_head(head) for head in HEAD_NAMES}
        total_count = sum(int(value["count"]) for value in heads.values())
        total_correct = sum(int(value["correct"]) for value in heads.values())
        total_weighted_numerator = math.fsum(float(value["weighted_numerator"]) for value in heads.values())
        total_weighted_denominator = math.fsum(float(value["weighted_denominator"]) for value in heads.values())
        total_loss = total_weighted_numerator / total_weighted_denominator if total_weighted_denominator else 0.0
        return {
            "format": METRICS_FORMAT,
            "batches": self.batches,
            "heads": heads,
            "micro": {
                "count": total_count,
                "correct": total_correct,
                "accuracy": total_correct / total_count if total_count else 0.0,
                "loss": total_loss,
                "weighted_numerator": total_weighted_numerator,
                "weighted_denominator": total_weighted_denominator,
            },
            "micro_accuracy": total_correct / total_count if total_count else 0.0,
            "micro_loss": total_loss,
            "action": {
                "count": self.action_count,
                "correct": self.action_correct,
                "accuracy": self.action_correct / self.action_count if self.action_count else 0.0,
                "end_to_end_correct": self.action_correct,
                "end_to_end_accuracy": self.action_correct / self.action_count if self.action_count else 0.0,
            },
            "offset": {
                "count": self.offset_count,
                "top1_correct": self.offset_top1,
                "top5_correct": self.offset_top5,
                "top1_accuracy": self.offset_top1 / self.offset_count if self.offset_count else 0.0,
                "top5_accuracy": self.offset_top5 / self.offset_count if self.offset_count else 0.0,
                "mean_manhattan_grid_distance": self.offset_distance / self.offset_count if self.offset_count else 0.0,
                "mean_manhattan_distance": self.offset_distance / self.offset_count if self.offset_count else 0.0,
            },
            "macro_f1": {head: heads[head]["supported_macro_f1"] for head in HEAD_NAMES},
            "balanced_accuracy": {head: heads[head]["balanced_accuracy"] for head in HEAD_NAMES},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_ai42_metrics(
    batch: AI42Batch,
    outputs: Mapping[str, Tensor],
    *,
    class_weights: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Compute deterministic metrics for one batch without changing a model."""

    accumulator = AI42MetricAccumulator(class_weights=class_weights)
    accumulator.update(batch, outputs)
    return accumulator.to_dict()


def evaluate_ai42_metrics(
    learner: Any,
    batches: Iterable[AI42Batch],
    *,
    profile: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a stream with one recurrent forward per batch."""

    weights = profile.class_weights() if profile is not None and hasattr(profile, "class_weights") else getattr(getattr(learner, "config", None), "class_weights", None)
    accumulator = AI42MetricAccumulator(class_weights=weights)
    was_training = learner.actor.training
    learner.actor.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                outputs = learner.forward(batch)
                accumulator.update(batch, outputs)
    finally:
        learner.actor.train(was_training)
    return accumulator.to_dict()


metrics_ai42 = compute_ai42_metrics
compute_metrics_ai42 = compute_ai42_metrics
evaluate_metrics_ai42 = evaluate_ai42_metrics
AI42MetricsAccumulator = AI42MetricAccumulator


__all__ = [
    "AI42MetricAccumulator", "AI42MetricsAccumulator", "METRICS_FORMAT", "compute_ai42_metrics", "compute_metrics_ai42", "evaluate_ai42_metrics", "evaluate_metrics_ai42", "metrics_ai42",
]
