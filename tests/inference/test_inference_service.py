"""
InferenceService のユニットテスト。
LLMClient と PromptTemplateManager をモック化して LLM なしで動作確認する。
"""

from unittest.mock import MagicMock, patch
import pytest

from services.inference.llm_inference_service.inference_service import InferenceService

_MOCK_TEMPLATE = "K={knowledge} L={log} Q={question}"


def _make_service(n_ensemble: int = 3) -> InferenceService:
    """LLMClient と PromptTemplateManager をモック化した InferenceService を返す。"""
    with patch(
        "services.inference.llm_inference_service.inference_service.LLMClient"
    ) as MockClient, patch(
        "services.inference.llm_inference_service.inference_service.PromptTemplateManager"
    ) as MockTM:
        MockTM.return_value.load.return_value = _MOCK_TEMPLATE
        instance = MockClient.return_value
        instance.generate_json.return_value = {
            "answer": "yes",
            "confidence": 0.8,
            "reason": "mocked reason",
        }
        service = InferenceService(n_ensemble=n_ensemble)
        service._client = instance
        return service


@pytest.fixture
def service():
    return _make_service(n_ensemble=3)


def test_run_returns_ensemble_result(service):
    result = service.run(
        knowledge_texts=["rule A", "rule B"],
        log="some log",
        question="Is this a problem?",
    )
    assert result.answer in ("yes", "no")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reason, str)


def test_run_calls_llm_n_times(service):
    service.run(
        knowledge_texts=["rule A"],
        log="log",
        question="question?",
    )
    assert service._client.generate_json.call_count == 3


def test_run_with_no_knowledge(service):
    result = service.run(
        knowledge_texts=[],
        log="some log",
        question="question?",
    )
    assert result.answer in ("yes", "no")


def _patch_both(**llm_kwargs):
    """LLMClient と PromptTemplateManager を同時にモック化するコンテキストマネージャ。"""
    return (
        patch("services.inference.llm_inference_service.inference_service.LLMClient", **llm_kwargs),
        patch("services.inference.llm_inference_service.inference_service.PromptTemplateManager"),
    )


def test_run_aggregates_mixed_answers():
    with patch(
        "services.inference.llm_inference_service.inference_service.LLMClient"
    ) as MockClient, patch(
        "services.inference.llm_inference_service.inference_service.PromptTemplateManager"
    ) as MockTM:
        MockTM.return_value.load.return_value = _MOCK_TEMPLATE
        instance = MockClient.return_value
        instance.generate_json.side_effect = [
            {"answer": "yes", "confidence": 0.9, "reason": "r1"},
            {"answer": "yes", "confidence": 0.8, "reason": "r2"},
            {"answer": "no",  "confidence": 0.7, "reason": "r3"},
        ]
        service = InferenceService(n_ensemble=3)
        service._client = instance

    result = service.run(["rule"], "log", "q?")
    assert result.answer == "yes"
    assert result.yes_ratio == pytest.approx(2 / 3, abs=1e-3)


def test_run_skips_invalid_json_and_still_aggregates():
    with patch(
        "services.inference.llm_inference_service.inference_service.LLMClient"
    ) as MockClient, patch(
        "services.inference.llm_inference_service.inference_service.PromptTemplateManager"
    ) as MockTM:
        MockTM.return_value.load.return_value = _MOCK_TEMPLATE
        instance = MockClient.return_value
        instance.generate_json.side_effect = [
            ValueError("parse error"),
            {"answer": "no", "confidence": 0.7, "reason": "valid"},
            {"answer": "no", "confidence": 0.8, "reason": "valid2"},
        ]
        service = InferenceService(n_ensemble=3)
        service._client = instance

    result = service.run(["rule"], "log", "q?")
    assert result.answer == "no"
    assert len(result.raw_results) == 2


def test_run_all_failed_raises():
    with patch(
        "services.inference.llm_inference_service.inference_service.LLMClient"
    ) as MockClient, patch(
        "services.inference.llm_inference_service.inference_service.PromptTemplateManager"
    ) as MockTM:
        MockTM.return_value.load.return_value = _MOCK_TEMPLATE
        instance = MockClient.return_value
        instance.generate_json.side_effect = ValueError("always fails")
        service = InferenceService(n_ensemble=3)
        service._client = instance

    with pytest.raises(RuntimeError):
        service.run(["rule"], "log", "q?")


def test_max_knowledge_units_limits_sample():
    with patch(
        "services.inference.llm_inference_service.inference_service.LLMClient"
    ) as MockClient, patch(
        "services.inference.llm_inference_service.inference_service.PromptTemplateManager"
    ) as MockTM:
        MockTM.return_value.load.return_value = _MOCK_TEMPLATE
        instance = MockClient.return_value
        instance.generate_json.return_value = {
            "answer": "yes", "confidence": 0.8, "reason": "r"
        }
        service = InferenceService(n_ensemble=1, max_knowledge_units=2)
        service._client = instance

    captured_prompts = []
    original_generate = instance.generate_json.side_effect

    def capture(prompt):
        captured_prompts.append(prompt)
        return {"answer": "yes", "confidence": 0.8, "reason": "r"}

    instance.generate_json.side_effect = capture

    service.run(
        knowledge_texts=["k1", "k2", "k3", "k4", "k5"],
        log="log",
        question="q?",
    )
    # プロンプト内に含まれるナレッジ番号が最大2つであることを確認
    assert captured_prompts[0].count("[1]") == 1
    assert "[3]" not in captured_prompts[0]
