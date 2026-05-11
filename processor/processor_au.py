"""Metrics for multi-label Action Unit occurrence detection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the training repo depends on torch.
    torch = None


AU_NAMES = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU9",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU25",
    "AU26",
]
DISFA8_INDICES = [0, 1, 2, 4, 5, 6, 10, 11]
DISFA8_AU_NAMES = [AU_NAMES[index] for index in DISFA8_INDICES]


def _to_numpy(values: Any) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _validate_batch(name: str, values: Any, expected_width: int) -> np.ndarray:
    array = _to_numpy(values)
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(
            f"{name} must have shape [B, {expected_width}], got {array.shape}"
        )
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_probabilities(probs: np.ndarray) -> None:
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probs must be sigmoid probabilities in the [0, 1] range")


def _validate_binary_targets(targets: np.ndarray) -> None:
    is_binary = np.isclose(targets, 0.0) | np.isclose(targets, 1.0)
    if not np.all(is_binary):
        raise ValueError(
            "targets must be binary 0/1 occurrence labels; binarize DISFA "
            "intensity labels with threshold >= 2 before evaluation"
        )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _safe_scalar_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _binary_roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = targets == 1
    negatives = targets == 0
    num_pos = int(positives.sum())
    num_neg = int(negatives.sum())

    if num_pos == 0 or num_neg == 0:
        return float("nan")

    ranks = _rankdata_average(scores)
    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def _nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _named_float_dict(
    values: np.ndarray,
    names: List[str],
    indices: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    if indices is None:
        indices = range(len(names))
    return {names[index]: float(values[index]) for index in indices}


def _named_int_dict(
    values: np.ndarray,
    names: List[str],
    indices: Optional[Iterable[int]] = None,
) -> Dict[str, int]:
    if indices is None:
        indices = range(len(names))
    return {names[index]: int(values[index]) for index in indices}


def _metadata_to_list(values: Any, batch_size: int, name: str) -> List[Any]:
    if torch is not None and isinstance(values, torch.Tensor):
        items = values.detach().cpu().tolist()
    elif isinstance(values, np.ndarray):
        items = values.tolist()
    elif isinstance(values, (list, tuple)):
        items = list(values)
    else:
        items = [values]

    if len(items) != batch_size:
        raise ValueError(f"{name} must contain {batch_size} items, got {len(items)}")
    return items


class AUEvaluator:
    """Evaluator for binary multi-label AU occurrence detection."""

    def __init__(self, threshold: float = 0.5, au_names: Optional[List[str]] = None):
        self.threshold = float(threshold)
        self.au_names = list(au_names or AU_NAMES)
        if len(self.au_names) != len(AU_NAMES):
            raise ValueError(f"AUEvaluator expects {len(AU_NAMES)} AU names")
        self.num_aus = len(self.au_names)
        self.reset()

    def reset(self) -> None:
        self._probs: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self.image_ids: List[Any] = []
        self.subject_ids: List[Any] = []

    def update(
        self,
        probs: Any,
        targets: Any,
        image_ids: Optional[Any] = None,
        subject_ids: Optional[Any] = None,
    ) -> None:
        probs_array = _validate_batch("probs", probs, self.num_aus)
        targets_array = _validate_batch("targets", targets, self.num_aus)
        _validate_probabilities(probs_array)
        _validate_binary_targets(targets_array)

        if probs_array.shape[0] != targets_array.shape[0]:
            raise ValueError(
                "probs and targets must have the same batch size, "
                f"got {probs_array.shape[0]} and {targets_array.shape[0]}"
            )

        self._probs.append(probs_array)
        self._targets.append(targets_array)

        batch_size = probs_array.shape[0]
        if image_ids is not None:
            self.image_ids.extend(_metadata_to_list(image_ids, batch_size, "image_ids"))
        if subject_ids is not None:
            self.subject_ids.extend(
                _metadata_to_list(subject_ids, batch_size, "subject_ids")
            )

    def compute(self) -> Dict[str, Any]:
        if not self._probs:
            raise RuntimeError("AUEvaluator.compute() called before update().")

        probs = np.concatenate(self._probs, axis=0)
        targets = (np.concatenate(self._targets, axis=0) > 0.5).astype(np.int64)
        preds = (probs > self.threshold).astype(np.int64)

        true_pos = (
            np.logical_and(preds == 1, targets == 1).sum(axis=0).astype(np.float64)
        )
        false_pos = (
            np.logical_and(preds == 1, targets == 0).sum(axis=0).astype(np.float64)
        )
        false_neg = (
            np.logical_and(preds == 0, targets == 1).sum(axis=0).astype(np.float64)
        )

        precision = _safe_divide(true_pos, true_pos + false_pos)
        recall = _safe_divide(true_pos, true_pos + false_neg)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        support = targets.sum(axis=0)

        per_au_auc = np.array(
            [
                _binary_roc_auc(targets[:, index], probs[:, index])
                for index in range(self.num_aus)
            ],
            dtype=np.float64,
        )

        total_tp = float(true_pos.sum())
        total_fp = float(false_pos.sum())
        total_fn = float(false_neg.sum())
        precision_micro = _safe_scalar_divide(total_tp, total_tp + total_fp)
        recall_micro = _safe_scalar_divide(total_tp, total_tp + total_fn)
        f1_micro = _safe_scalar_divide(
            2.0 * precision_micro * recall_micro,
            precision_micro + recall_micro,
        )

        f1_macro = float(f1.mean())
        roc_auc_macro = _nanmean(per_au_auc)
        element_accuracy = float((preds == targets).mean())

        metrics: Dict[str, Any] = {
            "avg_f1": f1_macro,
            "avg_auc": roc_auc_macro,
            "accuracy": element_accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision_macro": float(precision.mean()),
            "precision_micro": precision_micro,
            "recall_macro": float(recall.mean()),
            "recall_micro": recall_micro,
            "roc_auc_macro": roc_auc_macro,
            "exact_match_accuracy": float(np.all(preds == targets, axis=1).mean()),
            "hamming_loss": float((preds != targets).mean()),
            "per_au_f1": _named_float_dict(f1, self.au_names),
            "per_au_precision": _named_float_dict(precision, self.au_names),
            "per_au_recall": _named_float_dict(recall, self.au_names),
            "per_au_auc": _named_float_dict(per_au_auc, self.au_names),
            "per_au_support": _named_int_dict(support, self.au_names),
            "disfa8_f1_macro": float(f1[DISFA8_INDICES].mean()),
            "disfa8_auc_macro": _nanmean(per_au_auc[DISFA8_INDICES]),
            "disfa8_per_au_f1": _named_float_dict(
                f1, self.au_names, DISFA8_INDICES
            ),
            "num_samples": int(targets.shape[0]),
            "num_aus": self.num_aus,
            "threshold": self.threshold,
            "au_names": list(self.au_names),
            "disfa8_au_names": list(DISFA8_AU_NAMES),
        }

        for index in range(self.num_aus):
            metrics[f"per_class_f1_AU_{index}"] = float(f1[index])
            metrics[f"per_class_precision_AU_{index}"] = float(precision[index])
            metrics[f"per_class_recall_AU_{index}"] = float(recall[index])
            metrics[f"per_class_auc_AU_{index}"] = float(per_au_auc[index])
            metrics[f"per_class_support_AU_{index}"] = int(support[index])

        return metrics
