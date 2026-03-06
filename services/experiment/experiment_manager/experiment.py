"""
Experiment data models.

ExperimentConfig  : パラメータ設定（実験定義）
RunResult         : 1回の推論結果（単一 Yes/No 質問）
LabelPrediction   : 1ラベルの Yes/No 推論結果
ClassRunResult    : 1ログに対するマルチラベル分類結果
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExperimentConfig:
    experiment_id: str
    description: str = ""
    created_at: str = ""
    # model
    model_id: str = ""
    # prompt
    template_name: str = "default"
    # inference
    n_ensemble: int = 5
    max_knowledge_units: int | None = None
    # generation (override per experiment)
    temperature: float | None = None
    max_tokens: int | None = None
    # classification
    labels: list[str] = field(default_factory=list)
    question_template: str = "Is this log related to a {label} failure?"
    # knowledge set (empty list = use all registered knowledge)
    knowledge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunResult:
    run_id: str
    timestamp: str
    log_input: str
    question: str
    answer: str
    confidence: float
    yes_ratio: float
    reason: str
    n_runs: int                              # 実際に集計できた推論回数
    raw_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LabelPrediction:
    """1ラベルに対する Yes/No 推論の結果。"""
    label: str
    question: str
    answer: str          # "yes" or "no"
    confidence: float
    yes_ratio: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelPrediction:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClassRunResult:
    """1ログに対するマルチラベル分類の結果（Case B: 複数正解あり）。"""
    run_id: str
    timestamp: str
    log_input: str
    predicted_labels: list[str]       # answer=="yes" のラベル（confidence 降順）
    top_label: str                    # 最も高 confidence の "yes" ラベル（なければ最高 confidence）
    ground_truth: list[str]           # 正解ラベル（複数可）
    exact_match: bool                 # predicted_labels set == ground_truth set
    jaccard: float                    # |intersection| / |union|
    label_predictions: list[LabelPrediction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClassRunResult:
        label_preds = [
            LabelPrediction.from_dict(lp) if isinstance(lp, dict) else lp
            for lp in d.get("label_predictions", [])
        ]
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "label_predictions"}
        return cls(**fields, label_predictions=label_preds)


@dataclass
class ComparisonRow:
    """1つのログ×質問ペアに対する複数実験の比較行。"""
    log_input: str
    question: str
    results: dict[str, RunResult | None]     # experiment_id → RunResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_input": self.log_input,
            "question": self.question,
            "results": {
                exp_id: (r.to_dict() if r else None)
                for exp_id, r in self.results.items()
            },
        }
