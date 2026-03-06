"""
LLM 統合テスト — 実際の GGUF モデルが必要。

実行方法:
    pytest -m integration -v

通常の pytest 実行（pytest.ini の addopts）では除外される。
docker compose run --rm inference pytest -m integration でも実行可能。
"""

import json
import pytest
from pathlib import Path

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def service():
    from services.inference.llm_inference_service.inference_service import InferenceService
    from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

    km = KnowledgeManager()
    knowledge_texts = km.texts()
    assert knowledge_texts, "data/knowledge/ にナレッジファイルがありません"

    svc = InferenceService(n_ensemble=3)
    return svc, knowledge_texts


SAMPLE_LOG = "インバータにE01(過電流)エラーが連続3回発生。その後ブレーカーがトリップした。"
QUESTIONS = [
    "このログは電気系のトラブルですか？",
    "このログはソフト系のトラブルですか？",
    "このログはメカ系のトラブルですか？",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_inference_returns_valid_result(service, question):
    svc, knowledge_texts = service

    result = svc.run(
        knowledge_texts=knowledge_texts,
        log=SAMPLE_LOG,
        question=question,
    )

    assert result.answer in ("yes", "no")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reason, str) and len(result.reason) > 0
    assert 0.0 <= result.yes_ratio <= 1.0
    assert len(result.raw_results) > 0


def test_electrical_question_answers_yes(service):
    """電気系の質問に対して yes が多数を占めることを期待する。"""
    svc, knowledge_texts = service

    result = svc.run(
        knowledge_texts=knowledge_texts,
        log=SAMPLE_LOG,
        question="このログは電気系のトラブルですか？",
    )

    # 電気系エラーなので yes_ratio が 0.5 を超えることを期待
    assert result.yes_ratio >= 0.5, (
        f"Expected yes_ratio >= 0.5 for electrical question, got {result.yes_ratio}"
    )


def test_results_saved_to_log(service, tmp_path):
    svc, knowledge_texts = service

    result = svc.run(
        knowledge_texts=knowledge_texts,
        log=SAMPLE_LOG,
        question="このログは電気系のトラブルですか？",
    )

    out = tmp_path / "result.json"
    out.write_text(
        json.dumps({
            "answer": result.answer,
            "confidence": result.confidence,
            "yes_ratio": result.yes_ratio,
            "reason": result.reason,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert out.exists()
