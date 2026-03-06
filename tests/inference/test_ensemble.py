import pytest
from services.inference.llm_inference_service.ensemble import (
    InferenceResult,
    aggregate,
)


def _r(answer: str, confidence: float, reason: str = "") -> InferenceResult:
    return InferenceResult(answer=answer, confidence=confidence, reason=reason)


def test_majority_yes_wins():
    results = [_r("yes", 0.8), _r("yes", 0.9), _r("no", 0.6)]
    result = aggregate(results)
    assert result.answer == "yes"


def test_majority_no_wins():
    results = [_r("no", 0.9), _r("no", 0.7), _r("yes", 0.5)]
    result = aggregate(results)
    assert result.answer == "no"


def test_confidence_is_between_0_and_1():
    results = [_r("yes", 0.8), _r("no", 0.6)]
    result = aggregate(results)
    assert 0.0 <= result.confidence <= 1.0


def test_weighted_vote_tie_goes_to_yes():
    # yes: 0.5, no: 0.5 → yes wins (>= condition)
    results = [_r("yes", 0.5), _r("no", 0.5)]
    result = aggregate(results)
    assert result.answer == "yes"


def test_yes_ratio_calculation():
    results = [_r("yes", 0.8), _r("yes", 0.7), _r("no", 0.6), _r("no", 0.5)]
    result = aggregate(results)
    assert result.yes_ratio == pytest.approx(0.5)


def test_yes_ratio_all_yes():
    results = [_r("yes", 0.9), _r("yes", 0.8)]
    result = aggregate(results)
    assert result.yes_ratio == pytest.approx(1.0)


def test_yes_ratio_all_no():
    results = [_r("no", 0.9), _r("no", 0.8)]
    result = aggregate(results)
    assert result.yes_ratio == pytest.approx(0.0)


def test_reason_from_highest_confidence_on_winning_side():
    results = [
        _r("yes", 0.9, reason="strong yes"),
        _r("yes", 0.6, reason="weak yes"),
        _r("no", 0.7, reason="no reason"),
    ]
    result = aggregate(results)
    assert result.reason == "strong yes"


def test_raw_results_preserved():
    results = [_r("yes", 0.8), _r("no", 0.6)]
    result = aggregate(results)
    assert len(result.raw_results) == 2


def test_empty_results_raises():
    with pytest.raises(ValueError):
        aggregate([])


def test_single_result():
    result = aggregate([_r("yes", 0.75, reason="only one")])
    assert result.answer == "yes"
    assert result.confidence == pytest.approx(1.0)
    assert result.reason == "only one"


def test_answer_case_insensitive():
    results = [_r("YES", 0.8), _r("No", 0.6)]
    result = aggregate(results)
    assert result.answer in ("yes", "no")
