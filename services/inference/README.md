# inference service

GGUF モデルを使ったアンサンブル推論エンジンです。

## ファイル構成

```
services/inference/
├── llm_inference_service/
│   ├── inference_service.py       # アンサンブル推論の統括
│   ├── llm_client.py              # llama-cpp-python ラッパー
│   ├── ensemble.py                # 集約ロジック（weighted confidence vote）
│   ├── prompt_builder.py          # プロンプト組み立て
│   └── prompt_template_manager.py # テンプレートファイル管理
└── Dockerfile
```

## 設定ファイル

### config/inference.json

```json
{
  "active_model": "qwen2.5-3b-instruct-q4_k_m",
  "active_template": "default",
  "model":      { "n_ctx": 4096, "n_gpu_layers": 0, "verbose": false },
  "generation": { "temperature": 0.7, "max_tokens": 512, "top_p": 0.95 },
  "ensemble":   { "n": 5 }
}
```

`active_model` は `config/models.json` の `id` と対応します。

### config/models.json

```json
{
  "models": [
    {
      "id": "qwen2.5-3b-instruct-q4_k_m",
      "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
      "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
      "description": "Qwen 2.5 3B Instruct (Q4_K_M, ~2.0GB)",
      "context_length": 4096
    }
  ]
}
```

## API

### InferenceService

```python
from services.inference.llm_inference_service.inference_service import InferenceService

svc = InferenceService()
# オプションで個別オーバーライド可能
svc = InferenceService(
    n_ensemble=5,
    template_name="default",
    max_knowledge_units=3,
)

result = svc.run(
    knowledge_texts=["ルールA", "ルールB"],
    log="モーター過電流エラーが繰り返し発生",
    question="これは電気系統の問題ですか？",
)

print(result.answer)      # "yes" or "no"
print(result.confidence)  # 0.0 ~ 1.0
print(result.yes_ratio)   # Yes 回答の割合
print(result.reason)      # 根拠テキスト
print(result.raw_results) # list[InferenceResult]
```

### アンサンブル集約ロジック

各推論の `yes / no` + `confidence` を重み付きで集計します。

```
yes_weight = sum(confidence for "yes" answers)
no_weight  = sum(confidence for "no" answers)
final_answer = "yes" if yes_weight >= no_weight else "no"
confidence   = max(yes_weight, no_weight) / total_weight
```

### PromptTemplateManager

```python
from services.inference.llm_inference_service.prompt_template_manager import PromptTemplateManager

tm = PromptTemplateManager()          # デフォルト: data/prompts/
names   = tm.list_names()             # list[str]
content = tm.load("default")          # str
tm.save("my_template", content)       # 上書きあり
tm.delete("my_template")
```

テンプレートには `{knowledge}` `{log}` `{question}` を含めること。

## テスト

```bash
# ユニットテスト（LLM は unittest.mock でモック）
docker compose run --rm inference

# 統合テスト（実際の GGUF モデルが必要）
docker compose run --rm inference pytest -m integration -v
```

テストは `tests/inference/` にあります。

- `test_prompt_builder.py` — プロンプト組み立て
- `test_ensemble.py` — 集約ロジック
- `test_inference_service.py` — InferenceService（LLM モック）
- `test_prompt_template_manager.py` — テンプレート管理
- `tests/integration/test_llm_integration.py` — 実際の LLM 呼び出し（`-m integration`）
