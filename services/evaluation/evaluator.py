"""
Evaluator: runs B1–B4 baselines against eval_labels.json and computes metrics.

Baseline definitions
--------------------
B1: N=1,  knowledge_sampling="all",    aggregation="weighted"  — 単回推論
B2: N=5,  knowledge_sampling="all",    aggregation="weighted"  — 温度多様性のみ（知識固定）
B3: N=5,  knowledge_sampling="random", aggregation="weighted"  — 本手法 RKSSE
B4: N=5,  knowledge_sampling="random", aggregation="majority"  — 単純多数決

eval_labels.json format
-----------------------
[
  {
    "log_id": "log_001",
    "log_text": "...",
    "difficulty": "easy",
    "note": "...",
    "labels": {
      "これは電気系の障害ですか？": "yes",
      "これはソフトウェア系の障害ですか？": "no"
    }
  }
]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .metrics import BinaryMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    log_id: str
    log_text: str
    difficulty: str
    note: str
    labels: dict[str, str]   # question -> "yes"/"no"


@dataclass
class PredictionRecord:
    log_id: str
    question: str
    predicted: str
    confidence: float
    yes_ratio: float
    ground_truth: str
    difficulty: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaselineResult:
    baseline: str
    label: str
    n_ensemble: int
    knowledge_sampling: str
    aggregation: str
    metrics: BinaryMetrics
    records: list[PredictionRecord] = field(default_factory=list, repr=False)

    def to_summary_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "label": self.label,
            "n_ensemble": self.n_ensemble,
            "knowledge_sampling": self.knowledge_sampling,
            "aggregation": self.aggregation,
            "accuracy": self.metrics.accuracy,
            "f1": self.metrics.f1,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "ece": self.metrics.ece,
            "n_samples": self.metrics.n,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str, str], None]  # done, total, log_id, question


class Evaluator:
    """Run evaluation baselines against a labeled dataset."""

    BASELINES: dict[str, dict] = {
        "B1": dict(
            n_ensemble=1,
            knowledge_sampling="all",
            aggregation="weighted",
            label="B1: 単回推論 (N=1)",
        ),
        "B2": dict(
            n_ensemble=5,
            knowledge_sampling="all",
            aggregation="weighted",
            label="B2: 温度多様性のみ (N=5, 知識固定)",
        ),
        "B3": dict(
            n_ensemble=5,
            knowledge_sampling="random",
            aggregation="weighted",
            label="B3: 本手法 RKSSE (N=5, ランダム知識)",
        ),
        "B4": dict(
            n_ensemble=5,
            knowledge_sampling="random",
            aggregation="majority",
            label="B4: 単純多数決 (N=5, ランダム知識)",
        ),
    }

    def __init__(self, eval_labels_path: Path, knowledge_manager) -> None:
        self._labels_path = Path(eval_labels_path)
        self._km = knowledge_manager

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_items(self) -> list[EvalItem]:
        with open(self._labels_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            EvalItem(
                log_id=d["log_id"],
                log_text=d["log_text"],
                difficulty=d.get("difficulty", "unknown"),
                note=d.get("note", ""),
                labels=d["labels"],
            )
            for d in raw
        ]

    # ------------------------------------------------------------------
    # Run baselines
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        baseline_key: str,
        items: list[EvalItem],
        n_ensemble: int | None = None,
        callback: ProgressCallback | None = None,
    ) -> BaselineResult:
        """
        Run a single baseline against all items.

        Parameters
        ----------
        baseline_key : "B1" | "B2" | "B3" | "B4"
        items        : evaluation items from load_items()
        n_ensemble   : override the default N for this baseline
        callback     : optional progress callback(done, total, log_id, question)
        """
        from services.inference.llm_inference_service.inference_service import InferenceService

        cfg = self.BASELINES[baseline_key]
        n = n_ensemble if n_ensemble is not None else cfg["n_ensemble"]
        ks = cfg["knowledge_sampling"]
        agg = cfg["aggregation"]

        svc = InferenceService(
            n_ensemble=n,
            knowledge_sampling=ks,
            aggregation=agg,
        )
        all_texts = self._km.texts()

        records: list[PredictionRecord] = []
        total = sum(len(item.labels) for item in items)
        done = 0

        for item in items:
            for question, ground_truth in item.labels.items():
                if callback:
                    callback(done, total, item.log_id, question)
                result = svc.run(
                    knowledge_texts=all_texts,
                    log=item.log_text,
                    question=question,
                )
                records.append(PredictionRecord(
                    log_id=item.log_id,
                    question=question,
                    predicted=result.answer,
                    confidence=result.confidence,
                    yes_ratio=result.yes_ratio,
                    ground_truth=ground_truth,
                    difficulty=item.difficulty,
                ))
                done += 1

        metrics = compute_metrics(
            predictions=[r.predicted for r in records],
            labels=[r.ground_truth for r in records],
            confidences=[r.confidence for r in records],
        )

        return BaselineResult(
            baseline=baseline_key,
            label=cfg["label"],
            n_ensemble=n,
            knowledge_sampling=ks,
            aggregation=agg,
            metrics=metrics,
            records=records,
        )

    # ------------------------------------------------------------------
    # N-accuracy curve
    # ------------------------------------------------------------------

    def run_n_curve(
        self,
        items: list[EvalItem],
        n_values: list[int],
        knowledge_sampling: str = "random",
        aggregation: str = "weighted",
        callback: Callable[[int, str], None] | None = None,
    ) -> list[dict]:
        """
        Run B3-style inference for multiple N values.

        Returns list of {"n": int, "accuracy": float, "f1": float, "ece": float}.
        """
        from services.inference.llm_inference_service.inference_service import InferenceService

        all_texts = self._km.texts()
        results = []

        for n in n_values:
            svc = InferenceService(
                n_ensemble=n,
                knowledge_sampling=knowledge_sampling,
                aggregation=aggregation,
            )
            records: list[PredictionRecord] = []
            for item in items:
                for question, gt in item.labels.items():
                    if callback:
                        callback(n, question)
                    result = svc.run(
                        knowledge_texts=all_texts,
                        log=item.log_text,
                        question=question,
                    )
                    records.append(PredictionRecord(
                        log_id=item.log_id,
                        question=question,
                        predicted=result.answer,
                        confidence=result.confidence,
                        yes_ratio=result.yes_ratio,
                        ground_truth=gt,
                        difficulty=item.difficulty,
                    ))

            m = compute_metrics(
                predictions=[r.predicted for r in records],
                labels=[r.ground_truth for r in records],
                confidences=[r.confidence for r in records],
            )
            results.append({"n": n, "accuracy": m.accuracy, "f1": m.f1, "ece": m.ece})

        return results
