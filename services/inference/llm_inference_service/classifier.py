"""
ClassificationService: multi-class classification via per-label Yes/No ensemble inference.

For each label in `labels`, the service generates a Yes/No question using `question_template`
(must contain `{label}` placeholder), runs ensemble inference, and returns which labels
were predicted as "yes" along with their confidence scores.

Usage:
    from services.inference.llm_inference_service.classifier import ClassificationService

    svc = ClassificationService(
        labels=["electrical", "software", "mechanical"],
        question_template="このログは{label}系の障害ですか？",
        n_ensemble=5,
        knowledge_sampling="random",
        aggregation="weighted",
    )
    result = svc.classify(
        knowledge_texts=["rule A", "rule B"],
        log="Motor overcurrent error occurred.",
    )
    print(result.top_label)       # e.g. "electrical"
    print(result.predicted_labels) # e.g. ["electrical"]
    print(result.label_details)    # per-label breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inference_service import InferenceService


@dataclass
class LabelResult:
    label: str
    question: str
    answer: str          # "yes" or "no"
    confidence: float
    yes_ratio: float
    reason: str


@dataclass
class ClassificationResult:
    predicted_labels: list[str]          # labels where answer == "yes", sorted by confidence desc
    top_label: str                        # highest-confidence "yes" label; fallback: highest overall
    label_details: list[LabelResult]      # all per-label results
    labels: list[str] = field(default_factory=list)   # original label order


class ClassificationService:
    """
    Multi-class log classification via per-label Yes/No ensemble inference.

    Parameters
    ----------
    labels : list[str]
        Category labels to classify against (e.g. ["electrical", "software", "mechanical"]).
    question_template : str
        Template with ``{label}`` placeholder, e.g. "このログは{label}系の障害ですか？".
    n_ensemble : int | None
        Number of ensemble runs. Passed to InferenceService (None → reads config).
    template_name : str | None
        Prompt template name. Passed to InferenceService.
    max_knowledge_units : int | None
        Max knowledge texts to sample per run.
    knowledge_sampling : str
        "random" (RKSSE) or "all" (fixed, B2 style).
    aggregation : str
        "weighted" (confidence-weighted vote) or "majority" (simple count).
    """

    DEFAULT_TEMPLATE = "Is this log related to a {label} failure?"

    def __init__(
        self,
        labels: list[str],
        question_template: str | None = None,
        n_ensemble: int | None = None,
        template_name: str | None = None,
        max_knowledge_units: int | None = None,
        knowledge_sampling: str = "random",
        aggregation: str = "weighted",
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        self._labels = labels
        self._question_template = question_template or self.DEFAULT_TEMPLATE
        self._svc = InferenceService(
            n_ensemble=n_ensemble,
            template_name=template_name,
            max_knowledge_units=max_knowledge_units,
            knowledge_sampling=knowledge_sampling,
            aggregation=aggregation,
        )

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def question_template(self) -> str:
        return self._question_template

    def classify(
        self,
        knowledge_texts: list[str],
        log: str,
    ) -> ClassificationResult:
        """
        Run per-label Yes/No inference and return a ClassificationResult.

        Parameters
        ----------
        knowledge_texts : list[str]
            Knowledge base texts passed to InferenceService.
        log : str
            Log text to classify.
        """
        label_results: list[LabelResult] = []

        for label in self._labels:
            question = self._question_template.format(label=label)
            ensemble = self._svc.run(
                knowledge_texts=knowledge_texts,
                log=log,
                question=question,
            )
            label_results.append(LabelResult(
                label=label,
                question=question,
                answer=ensemble.answer,
                confidence=ensemble.confidence,
                yes_ratio=ensemble.yes_ratio,
                reason=ensemble.reason,
            ))

        # predicted labels: where answer == "yes", sorted by confidence desc
        yes_results = [r for r in label_results if r.answer == "yes"]
        yes_results_sorted = sorted(yes_results, key=lambda r: r.confidence, reverse=True)
        predicted_labels = [r.label for r in yes_results_sorted]

        # top_label: highest confidence "yes", or if none, highest confidence overall
        if yes_results_sorted:
            top_label = yes_results_sorted[0].label
        else:
            top_label = max(label_results, key=lambda r: r.confidence).label

        return ClassificationResult(
            predicted_labels=predicted_labels,
            top_label=top_label,
            label_details=label_results,
            labels=list(self._labels),
        )
