# experiment service

推論実験の設定管理・結果保存・複数実験の比較を行うサービスです。

**注意**: ExperimentManager は推論を直接実行しません。
InferenceService で得た `EnsembleResult` を受け取って保存します。

## ファイル構成

```
services/experiment/
├── experiment_manager/
│   ├── experiment_manager.py  # CRUD・結果保存・比較
│   └── experiment.py          # データクラス定義
└── Dockerfile
```

## データ形式

```
data/experiments/
    exp_001/
        config.json    # ExperimentConfig
        results.json   # list[RunResult]
```

### config.json

```json
{
  "experiment_id": "exp_001",
  "description": "3B モデル vs 7B モデル比較",
  "created_at": "2025-01-01T00:00:00+00:00",
  "model_id": "qwen2.5-3b-instruct-q4_k_m",
  "template_name": "default",
  "n_ensemble": 5,
  "max_knowledge_units": null,
  "temperature": 0.7,
  "max_tokens": 512
}
```

## API

### ExperimentManager

```python
from services.experiment.experiment_manager.experiment_manager import ExperimentManager

em = ExperimentManager()                        # デフォルト: data/experiments/
em = ExperimentManager(experiments_dir=path)    # テスト用

# 実験の CRUD
cfg = em.create(
    experiment_id="exp_001",
    description="テスト実験",
    template_name="default",
    n_ensemble=5,
    temperature=0.7,
    max_tokens=512,
)
cfg    = em.load_config("exp_001")
ids    = em.list_ids()
em.delete("exp_001")

# 推論結果の保存（EnsembleResult を渡す）
run_result = em.save_result(
    experiment_id="exp_001",
    log_input="エラーログテキスト",
    question="これは電気系の問題ですか？",
    ensemble_result=result,   # InferenceService.run() の返り値
)

# 結果の取得
results = em.load_results("exp_001")  # list[RunResult]

# 複数実験の比較
rows = em.compare(["exp_001", "exp_002"])
rows = em.compare(["exp_001", "exp_002"], log_input="...", question="...")
# rows: list[ComparisonRow]
# row.results["exp_001"]  → RunResult | None
```

### データクラス

```python
# ExperimentConfig
cfg.experiment_id, cfg.description, cfg.created_at
cfg.model_id, cfg.template_name, cfg.n_ensemble
cfg.max_knowledge_units, cfg.temperature, cfg.max_tokens

# RunResult
r.run_id, r.timestamp
r.log_input, r.question
r.answer, r.confidence, r.yes_ratio, r.reason
r.n_runs, r.raw_results  # list[dict]
```

## テスト

```bash
# ユニットテスト（LLM 不要）
docker compose run --rm experiment
```

テストは `tests/experiment/test_experiment_manager.py` にあります。
