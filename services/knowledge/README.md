# knowledge service

`data/knowledge/` に保存されたナレッジユニットを管理するサービスです。

## ファイル構成

```
services/knowledge/
├── knowledge_manager/
│   ├── knowledge_manager.py   # CRUD・サンプリング・サマリーキャッシュ
│   └── knowledge_unit.py      # KnowledgeUnit データクラス + ファイルローダー
└── Dockerfile
```

## データ形式

1 ナレッジ = 3 ファイル（`knowledge_id` がキー）

```
data/knowledge/
    rule_01.md           # 本文 (Markdown)
    rule_01.json         # メタデータ (title / source)
    rule_01.summary.txt  # サマリーキャッシュ（任意）
```

### メタデータ形式 (.json)

```json
{
  "knowledge_id": "rule_01",
  "title": "ルール01",
  "source": "manual",
  "summary": null,
  "embedding": null
}
```

## API

### KnowledgeManager

```python
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

km = KnowledgeManager()                    # デフォルト: data/knowledge/
km = KnowledgeManager(knowledge_dir=path)  # テスト用にパス指定可能

# 読み取り
ids   = km.list_ids()           # list[str]
unit  = km.load("rule_01")      # KnowledgeUnit
units = km.load_all()           # list[KnowledgeUnit]
units = km.sample(n=3)          # ランダムサンプル
texts = km.texts()              # list[str]（サマリーがあればサマリーを優先）

# 書き込み
unit = km.add("rule_01", text, title="タイトル", source="出所")
unit = km.add("rule_01", text, overwrite=True)  # 上書き
km.delete("rule_01")

# サマリーキャッシュ
km.save_summary("rule_01", summary_text)
km.delete_summary("rule_01")
```

### KnowledgeUnit

```python
unit.knowledge_id   # str
unit.text           # str (Markdown 本文)
unit.title          # str
unit.source         # str
unit.summary        # str | None
unit.effective_text # summary があれば summary、なければ text
str(unit)           # effective_text と同じ
```

## テスト

```bash
# ユニットテスト（LLM 不要）
docker compose run --rm knowledge
```

テストは `tests/knowledge/` にあります。

- `test_knowledge_unit.py` — KnowledgeUnit のロード・メタデータ
- `test_knowledge_manager.py` — CRUD・サンプリング・サマリー
