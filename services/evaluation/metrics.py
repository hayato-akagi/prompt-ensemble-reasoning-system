"""
Evaluation metrics: Accuracy, F1 (binary), ECE (Expected Calibration Error).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support_pos: int   # ground-truth positive count
    support_neg: int   # ground-truth negative count
    n: int             # total samples
    ece: float


def compute_metrics(
    predictions: list[str],
    labels: list[str],
    confidences: list[float],
    positive: str = "yes",
    n_bins: int = 10,
) -> BinaryMetrics:
    """
    Compute Accuracy, Precision, Recall, F1, and ECE for binary predictions.

    Parameters
    ----------
    predictions : list[str]   — predicted labels ("yes"/"no")
    labels      : list[str]   — ground truth labels
    confidences : list[float] — model confidence per prediction (0.0–1.0)
    positive    : str         — which label counts as "positive"
    n_bins      : int         — number of bins for ECE
    """
    assert len(predictions) == len(labels) == len(confidences), "Length mismatch"
    n = len(predictions)
    if n == 0:
        return BinaryMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)

    tp = sum(p == positive and l == positive for p, l in zip(predictions, labels))
    fp = sum(p == positive and l != positive for p, l in zip(predictions, labels))
    fn = sum(p != positive and l == positive for p, l in zip(predictions, labels))
    correct = sum(p == l for p, l in zip(predictions, labels))

    acc = round(correct / n, 4)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    support_pos = sum(l == positive for l in labels)
    support_neg = n - support_pos

    ece_val = _ece(confidences, predictions, labels, n_bins)

    return BinaryMetrics(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        support_pos=support_pos,
        support_neg=support_neg,
        n=n,
        ece=ece_val,
    )


def _ece(
    confidences: list[float],
    predictions: list[str],
    labels: list[str],
    n_bins: int,
) -> float:
    """Expected Calibration Error — average |accuracy - confidence| weighted by bin size."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, pred, label in zip(confidences, predictions, labels):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, pred == label))

    ece_val = 0.0
    n = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(ok for _, ok in b) / len(b)
        ece_val += (len(b) / n) * abs(avg_conf - avg_acc)

    return round(ece_val, 4)
