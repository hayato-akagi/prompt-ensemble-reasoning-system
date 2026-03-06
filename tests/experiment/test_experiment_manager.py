"""
ExperimentManager のユニットテスト。
InferenceService は使わず、EnsembleResult をモックオブジェクトで代替する。
"""

from types import SimpleNamespace
import pytest

from services.experiment.experiment_manager.experiment_manager import ExperimentManager
from services.experiment.experiment_manager.experiment import (
    ExperimentConfig,
    RunResult,
    ComparisonRow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def em(tmp_path):
    return ExperimentManager(experiments_dir=tmp_path)


def _mock_ensemble(answer="yes", confidence=0.8, yes_ratio=0.8, reason="test reason"):
    """EnsembleResult の代わりに使うモックオブジェクト。"""
    raw = [
        SimpleNamespace(answer=answer, confidence=confidence, reason=reason)
    ]
    return SimpleNamespace(
        answer=answer,
        confidence=confidence,
        yes_ratio=yes_ratio,
        reason=reason,
        raw_results=raw,
    )


# ---------------------------------------------------------------------------
# create / load_config
# ---------------------------------------------------------------------------

def test_create_returns_config(em):
    config = em.create("exp_001", description="test", model_id="model-a")
    assert config.experiment_id == "exp_001"
    assert config.model_id == "model-a"
    assert config.created_at != ""


def test_create_writes_config_json(em, tmp_path):
    em.create("exp_001")
    assert (tmp_path / "exp_001" / "config.json").exists()


def test_create_duplicate_raises(em):
    em.create("exp_001")
    with pytest.raises(FileExistsError):
        em.create("exp_001")


def test_create_overwrite(em):
    em.create("exp_001", description="first")
    em.create("exp_001", description="updated", overwrite=True)
    config = em.load_config("exp_001")
    assert config.description == "updated"


def test_load_config_roundtrip(em):
    em.create(
        "exp_001",
        description="desc",
        model_id="m",
        template_name="strict",
        n_ensemble=3,
        max_knowledge_units=2,
        temperature=0.5,
        max_tokens=256,
    )
    config = em.load_config("exp_001")
    assert config.template_name == "strict"
    assert config.n_ensemble == 3
    assert config.max_knowledge_units == 2
    assert config.temperature == 0.5
    assert config.max_tokens == 256


def test_load_config_nonexistent_raises(em):
    with pytest.raises(FileNotFoundError):
        em.load_config("nonexistent")


# ---------------------------------------------------------------------------
# list_ids / delete
# ---------------------------------------------------------------------------

def test_list_ids_empty(em):
    assert em.list_ids() == []


def test_list_ids_returns_created(em):
    em.create("exp_001")
    em.create("exp_002")
    ids = em.list_ids()
    assert set(ids) == {"exp_001", "exp_002"}


def test_delete_removes_experiment(em):
    em.create("exp_001")
    em.delete("exp_001")
    assert "exp_001" not in em.list_ids()


def test_delete_nonexistent_raises(em):
    with pytest.raises(FileNotFoundError):
        em.delete("nonexistent")


def test_len(em):
    em.create("exp_001")
    em.create("exp_002")
    assert len(em) == 2


# ---------------------------------------------------------------------------
# save_result / load_results
# ---------------------------------------------------------------------------

def test_save_result_returns_run_result(em):
    em.create("exp_001")
    result = em.save_result("exp_001", "log text", "is this X?", _mock_ensemble())
    assert isinstance(result, RunResult)
    assert result.answer == "yes"
    assert result.run_id != ""
    assert result.timestamp != ""


def test_save_result_persists(em):
    em.create("exp_001")
    em.save_result("exp_001", "log", "q?", _mock_ensemble("no", 0.6, 0.3))
    results = em.load_results("exp_001")
    assert len(results) == 1
    assert results[0].answer == "no"
    assert results[0].confidence == pytest.approx(0.6)


def test_save_multiple_results_appends(em):
    em.create("exp_001")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes"))
    em.save_result("exp_001", "log2", "q?", _mock_ensemble("no"))
    results = em.load_results("exp_001")
    assert len(results) == 2


def test_load_results_empty_when_no_runs(em):
    em.create("exp_001")
    assert em.load_results("exp_001") == []


def test_save_result_includes_raw_results(em):
    em.create("exp_001")
    result = em.save_result("exp_001", "log", "q?", _mock_ensemble())
    assert len(result.raw_results) == 1
    assert result.raw_results[0]["answer"] == "yes"


def test_save_result_n_runs(em):
    em.create("exp_001")
    mock = _mock_ensemble()
    # raw_results に3件追加
    mock.raw_results = [mock.raw_results[0]] * 3
    result = em.save_result("exp_001", "log", "q?", mock)
    assert result.n_runs == 3


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_same_log_across_experiments(em):
    em.create("exp_001")
    em.create("exp_002")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes", 0.9))
    em.save_result("exp_002", "log1", "q?", _mock_ensemble("no", 0.7))

    rows = em.compare(["exp_001", "exp_002"], log_input="log1", question="q?")

    assert len(rows) == 1
    assert rows[0].results["exp_001"].answer == "yes"
    assert rows[0].results["exp_002"].answer == "no"


def test_compare_missing_result_is_none(em):
    em.create("exp_001")
    em.create("exp_002")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes"))
    # exp_002 には結果なし

    rows = em.compare(["exp_001", "exp_002"], log_input="log1", question="q?")

    assert rows[0].results["exp_001"] is not None
    assert rows[0].results["exp_002"] is None


def test_compare_multiple_logs(em):
    em.create("exp_001")
    em.create("exp_002")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes"))
    em.save_result("exp_001", "log2", "q?", _mock_ensemble("no"))
    em.save_result("exp_002", "log1", "q?", _mock_ensemble("no"))
    em.save_result("exp_002", "log2", "q?", _mock_ensemble("yes"))

    rows = em.compare(["exp_001", "exp_002"])

    assert len(rows) == 2


def test_compare_latest_result_when_duplicates(em):
    em.create("exp_001")
    em.create("exp_002")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes", 0.7))
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("no", 0.9))  # 最新

    rows = em.compare(["exp_001", "exp_002"], log_input="log1", question="q?")

    assert rows[0].results["exp_001"].answer == "no"  # 最新を返す


def test_compare_to_dict(em):
    em.create("exp_001")
    em.save_result("exp_001", "log1", "q?", _mock_ensemble("yes"))
    rows = em.compare(["exp_001"])
    d = rows[0].to_dict()
    assert "log_input" in d
    assert "question" in d
    assert "results" in d
