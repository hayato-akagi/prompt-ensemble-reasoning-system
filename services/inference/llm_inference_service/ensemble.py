"""
Ensemble aggregation logic.

Takes N inference results (each with answer/confidence/reason)
and produces a single aggregated result.

Aggregation strategy:
  - Weighted vote: each result votes yes/no weighted by its confidence.
  - Final answer: the side with higher total weight wins.
  - Final confidence: winning weight / total weight.
  - Final reason: reason from the result with the highest individual confidence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceResult:
    answer: str          # "yes" or "no"
    confidence: float    # 0.0 - 1.0
    reason: str


@dataclass
class EnsembleResult:
    answer: str
    confidence: float
    reason: str
    raw_results: list[InferenceResult]
    yes_ratio: float     # fraction of results that answered "yes"


def aggregate_majority(results: list[InferenceResult]) -> EnsembleResult:
    """Aggregate by simple majority vote (B4 baseline: no confidence weighting)."""
    if not results:
        raise ValueError("No results to aggregate")

    yes_count = sum(1 for r in results if r.answer.strip().lower() == "yes")
    no_count = len(results) - yes_count
    final_answer = "yes" if yes_count >= no_count else "no"
    yes_ratio = round(yes_count / len(results), 4)

    # Report average confidence (informational only — not used for decision)
    avg_conf = round(sum(r.confidence for r in results) / len(results), 4)

    winning_results = [r for r in results if r.answer.strip().lower() == final_answer]
    best = max(winning_results, key=lambda r: r.confidence)

    return EnsembleResult(
        answer=final_answer,
        confidence=avg_conf,
        reason=best.reason,
        raw_results=results,
        yes_ratio=yes_ratio,
    )


def aggregate(results: list[InferenceResult]) -> EnsembleResult:
    """Aggregate N InferenceResults into one EnsembleResult."""
    if not results:
        raise ValueError("No results to aggregate")

    yes_weight = 0.0
    no_weight = 0.0

    for r in results:
        normalized = r.answer.strip().lower()
        if normalized == "yes":
            yes_weight += r.confidence
        else:
            no_weight += r.confidence

    total_weight = yes_weight + no_weight
    if total_weight == 0:
        total_weight = 1.0  # avoid division by zero

    final_answer = "yes" if yes_weight >= no_weight else "no"
    final_confidence = round(
        (yes_weight if final_answer == "yes" else no_weight) / total_weight, 4
    )
    yes_ratio = round(
        sum(1 for r in results if r.answer.strip().lower() == "yes") / len(results), 4
    )

    # Pick the reason from the highest-confidence result on the winning side
    winning_results = [
        r for r in results if r.answer.strip().lower() == final_answer
    ]
    best = max(winning_results, key=lambda r: r.confidence)

    return EnsembleResult(
        answer=final_answer,
        confidence=final_confidence,
        reason=best.reason,
        raw_results=results,
        yes_ratio=yes_ratio,
    )
