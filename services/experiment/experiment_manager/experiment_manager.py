"""
ExperimentManager: 実験の設定管理・結果保存・比較。

設計方針:
  - ExperimentManager は推論を直接実行しない（InferenceService に依存しない）
  - 推論結果は EnsembleResult を受け取って save_result() で保存する
  - これにより unit test でモックなしに動作確認できる

ディレクトリ構造:
    data/experiments/
        exp_001/
            config.json    ← ExperimentConfig
            results.json   ← list[RunResult]
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import ExperimentConfig, RunResult, ComparisonRow, LabelPrediction, ClassRunResult


class ExperimentManager:
    """
    実験の CRUD・結果保存・比較を管理する。

    Parameters
    ----------
    experiments_dir : Path | str | None
        実験データの保存先。デフォルトは <project_root>/data/experiments/。
    """

    def __init__(self, experiments_dir: Path | str | None = None) -> None:
        if experiments_dir is None:
            experiments_dir = Path(__file__).parents[3] / "data" / "experiments"
        self._dir = Path(experiments_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 実験の CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        experiment_id: str,
        description: str = "",
        model_id: str = "",
        template_name: str = "default",
        n_ensemble: int = 5,
        max_knowledge_units: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        labels: list[str] | None = None,
        question_template: str = "Is this log related to a {label} failure?",
        knowledge_ids: list[str] | None = None,
        overwrite: bool = False,
    ) -> ExperimentConfig:
        """実験を新規作成して config.json を保存する。"""
        exp_dir = self._dir / experiment_id
        config_path = exp_dir / "config.json"

        if config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Experiment '{experiment_id}' already exists. Use overwrite=True."
            )

        exp_dir.mkdir(parents=True, exist_ok=True)

        config = ExperimentConfig(
            experiment_id=experiment_id,
            description=description,
            created_at=_now(),
            model_id=model_id,
            template_name=template_name,
            n_ensemble=n_ensemble,
            max_knowledge_units=max_knowledge_units,
            temperature=temperature,
            max_tokens=max_tokens,
            labels=labels or [],
            question_template=question_template,
            knowledge_ids=knowledge_ids or [],
        )
        _write_json(config_path, config.to_dict())
        return config

    def load_config(self, experiment_id: str) -> ExperimentConfig:
        """実験設定を読み込む。"""
        config_path = self._experiment_dir(experiment_id) / "config.json"
        return ExperimentConfig.from_dict(_read_json(config_path))

    def list_ids(self) -> list[str]:
        """実験 ID の一覧を返す（作成日時順）。"""
        dirs = [d for d in self._dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
        dirs.sort(key=lambda d: (self._dir / d / "config.json").stat().st_mtime)
        return [d.name for d in dirs]

    def delete(self, experiment_id: str) -> None:
        """実験ディレクトリを丸ごと削除する。"""
        import shutil
        exp_dir = self._experiment_dir(experiment_id)
        shutil.rmtree(exp_dir)

    # ------------------------------------------------------------------
    # 結果の保存・取得
    # ------------------------------------------------------------------

    def save_result(
        self,
        experiment_id: str,
        log_input: str,
        question: str,
        ensemble_result: Any,   # EnsembleResult (型ヒントを避け循環 import を防ぐ)
    ) -> RunResult:
        """
        InferenceService.run() の返り値（EnsembleResult）を受け取り、
        results.json に追記して RunResult を返す。
        """
        run_result = RunResult(
            run_id=str(uuid.uuid4()),
            timestamp=_now(),
            log_input=log_input,
            question=question,
            answer=ensemble_result.answer,
            confidence=ensemble_result.confidence,
            yes_ratio=ensemble_result.yes_ratio,
            reason=ensemble_result.reason,
            n_runs=len(ensemble_result.raw_results),
            raw_results=[
                {"answer": r.answer, "confidence": r.confidence, "reason": r.reason}
                for r in ensemble_result.raw_results
            ],
        )

        results_path = self._experiment_dir(experiment_id) / "results.json"
        existing = _read_json(results_path) if results_path.exists() else []
        existing.append(run_result.to_dict())
        _write_json(results_path, existing)

        return run_result

    def load_results(self, experiment_id: str) -> list[RunResult]:
        """実験の全推論結果を返す。"""
        results_path = self._experiment_dir(experiment_id) / "results.json"
        if not results_path.exists():
            return []
        return [RunResult.from_dict(d) for d in _read_json(results_path)]

    # ------------------------------------------------------------------
    # マルチラベル分類結果の保存・取得
    # ------------------------------------------------------------------

    def save_class_result(
        self,
        experiment_id: str,
        log_input: str,
        classification_result: Any,   # ClassificationResult from classifier.py
        ground_truth: list[str],
        log_id: str = "",
    ) -> ClassRunResult:
        """
        ClassificationService.classify() の返り値を受け取り、
        class_results.json に追記して ClassRunResult を返す。

        Parameters
        ----------
        classification_result : ClassificationResult
            ClassificationService.classify() の返り値。
        ground_truth : list[str]
            正解ラベル（複数可）。
        """
        pred_set = set(classification_result.predicted_labels)
        gt_set = set(ground_truth)
        exact_match = pred_set == gt_set
        union = pred_set | gt_set
        jaccard = round(len(pred_set & gt_set) / len(union), 4) if union else 1.0

        label_preds = [
            LabelPrediction(
                label=lr.label,
                question=lr.question,
                answer=lr.answer,
                confidence=lr.confidence,
                yes_ratio=lr.yes_ratio,
                reason=lr.reason,
            )
            for lr in classification_result.label_details
        ]

        run_result = ClassRunResult(
            run_id=str(uuid.uuid4()),
            timestamp=_now(),
            log_input=log_input,
            predicted_labels=classification_result.predicted_labels,
            top_label=classification_result.top_label,
            ground_truth=ground_truth,
            exact_match=exact_match,
            jaccard=jaccard,
            log_id=log_id,
            label_predictions=label_preds,
        )

        results_path = self._experiment_dir(experiment_id) / "class_results.json"
        existing = _read_json(results_path) if results_path.exists() else []
        existing.append(run_result.to_dict())
        _write_json(results_path, existing)

        return run_result

    def load_class_results(self, experiment_id: str) -> list[ClassRunResult]:
        """実験の全分類結果を返す。"""
        results_path = self._experiment_dir(experiment_id) / "class_results.json"
        if not results_path.exists():
            return []
        return [ClassRunResult.from_dict(d) for d in _read_json(results_path)]

    # ------------------------------------------------------------------
    # 複数実験の比較
    # ------------------------------------------------------------------

    def compare(
        self,
        experiment_ids: list[str],
        log_input: str | None = None,
        question: str | None = None,
    ) -> list[ComparisonRow]:
        """
        複数実験の結果を比較する。

        log_input / question でフィルタリングし、
        一致する RunResult を experiment_id ごとに並べて返す。
        フィルタなしの場合はすべての結果を対象にする（最新1件ずつ）。
        """
        # 各実験の結果を読み込み
        all_results: dict[str, list[RunResult]] = {
            exp_id: self.load_results(exp_id) for exp_id in experiment_ids
        }

        # フィルタリング
        if log_input is not None or question is not None:
            all_results = {
                exp_id: [
                    r for r in results
                    if (log_input is None or r.log_input == log_input)
                    and (question is None or r.question == question)
                ]
                for exp_id, results in all_results.items()
            }

        # (log_input, question) のペアを全実験から収集
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for results in all_results.values():
            for r in results:
                key = (r.log_input, r.question)
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)

        rows: list[ComparisonRow] = []
        for log_in, q in pairs:
            row_results: dict[str, RunResult | None] = {}
            for exp_id in experiment_ids:
                matched = [
                    r for r in all_results[exp_id]
                    if r.log_input == log_in and r.question == q
                ]
                # 同じ log/question が複数あれば最新を返す
                row_results[exp_id] = matched[-1] if matched else None
            rows.append(ComparisonRow(log_input=log_in, question=q, results=row_results))

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _experiment_dir(self, experiment_id: str) -> Path:
        exp_dir = self._dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found.")
        return exp_dir

    def __len__(self) -> int:
        return len(self.list_ids())

    def __repr__(self) -> str:
        return f"ExperimentManager(dir={self._dir}, count={len(self)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
