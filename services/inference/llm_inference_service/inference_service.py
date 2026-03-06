"""
InferenceService: orchestrates ensemble inference.

Usage:
    service = InferenceService()
    result = service.run(
        knowledge_texts=["rule A", "rule B"],
        log="Motor overcurrent error occurred repeatedly",
        question="Is this an electrical system problem?",
    )
    print(result.answer, result.confidence, result.reason)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .llm_client import LLMClient
from .prompt_builder import build_prompt
from .prompt_template_manager import PromptTemplateManager
from .ensemble import InferenceResult, EnsembleResult, aggregate, aggregate_majority


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).parents[3] / "config" / "inference.json"
    with open(config_path) as f:
        return json.load(f)


class InferenceService:
    """
    High-level service for ensemble inference.

    Parameters
    ----------
    n_ensemble : int | None
        Number of inference runs. If None, reads from config.
    template_name : str | None
        Template name to load from data/prompts/. If None, reads active_template from config.
    max_knowledge_units : int | None
        Max knowledge texts to sample per run. If None, uses all.
    knowledge_sampling : str
        "random" (default) — randomly sample a subset each run (RKSSE).
        "all"    — use all knowledge texts every run (no subsampling, B2 baseline).
    aggregation : str
        "weighted" (default) — confidence-weighted vote.
        "majority"           — simple majority vote, no weighting (B4 baseline).
    """

    def __init__(
        self,
        n_ensemble: int | None = None,
        template_name: str | None = None,
        max_knowledge_units: int | None = None,
        knowledge_sampling: str = "random",
        aggregation: str = "weighted",
    ) -> None:
        cfg = _load_config()
        self._n = n_ensemble if n_ensemble is not None else cfg["ensemble"]["n"]
        self._max_knowledge_units = max_knowledge_units
        self._knowledge_sampling = knowledge_sampling
        self._aggregation = aggregation

        self._template_manager = PromptTemplateManager()
        active_template = template_name or cfg.get("active_template", "default")
        self._template = self._template_manager.load(active_template)
        self.template_name = active_template

        self._client = LLMClient()

    def run(
        self,
        knowledge_texts: list[str],
        log: str,
        question: str,
    ) -> EnsembleResult:
        """
        Run N inference calls with randomly sampled knowledge subsets.
        Returns aggregated EnsembleResult.
        """
        raw_results: list[InferenceResult] = []

        for i in range(self._n):
            if self._knowledge_sampling == "all":
                sampled = knowledge_texts.copy()
                random.shuffle(sampled)
            else:
                sampled = _sample_knowledge(knowledge_texts, self._max_knowledge_units)
            knowledge_block = _format_knowledge_block(sampled)
            prompt = build_prompt(knowledge_block, log, question, self._template)

            try:
                output = self._client.generate_json(prompt)
                result = _parse_output(output)
            except (ValueError, KeyError) as e:
                print(f"[run {i+1}/{self._n}] Parse error: {e} — skipping")
                continue

            print(
                f"[run {i+1}/{self._n}] answer={result.answer}  "
                f"confidence={result.confidence:.2f}"
            )
            raw_results.append(result)

        if not raw_results:
            raise RuntimeError("All inference runs failed. Check the model output.")

        if self._aggregation == "majority":
            return aggregate_majority(raw_results)
        return aggregate(raw_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_knowledge(
    knowledge_texts: list[str],
    max_units: int | None,
) -> list[str]:
    """Randomly sample up to max_units knowledge texts."""
    if not knowledge_texts:
        return []
    if max_units is None or max_units >= len(knowledge_texts):
        sampled = knowledge_texts.copy()
    else:
        sampled = random.sample(knowledge_texts, max_units)
    random.shuffle(sampled)
    return sampled


def _format_knowledge_block(texts: list[str]) -> str:
    if not texts:
        return "(no knowledge provided)"
    return "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))


def _parse_output(output: dict[str, Any]) -> InferenceResult:
    answer = str(output.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        raise ValueError(f"Invalid answer value: {answer!r}")

    confidence = float(output.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reason = str(output.get("reason", ""))

    return InferenceResult(answer=answer, confidence=confidence, reason=reason)
