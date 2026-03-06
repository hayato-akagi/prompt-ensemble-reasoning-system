# ingestion service

様々な形式のドキュメントを Markdown に変換して KnowledgeManager に登録するサービスです。

## ファイル構成

```
services/ingestion/
├── document_to_markdown/
│   ├── converters.py          # 形式別変換関数
│   └── ingestion_service.py   # IngestionService（変換 + 登録の統括）
├── requirements.txt
└── Dockerfile
```

## 対応フォーマット

| 拡張子 | 変換方法 | 依存ライブラリ |
| ------ | -------- | -------------- |
| .txt   | そのまま読み込み | — |
| .md    | そのまま読み込み | — |
| .csv   | Markdown テーブルに変換 | 標準 csv |
| .xlsx / .xls | 各シートを Markdown テーブルに変換 | pandas / openpyxl |
| .pdf   | テキスト抽出 | pdfminer.six |
| .docx  | 段落・テーブル抽出 | python-docx |

## API

### IngestionService

```python
from services.ingestion.document_to_markdown.ingestion_service import IngestionService
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

# KnowledgeManager を注入（テスト用）
km = KnowledgeManager(knowledge_dir=tmp_path)
svc = IngestionService(knowledge_manager=km)

# デフォルト（data/knowledge/ を使用）
svc = IngestionService()

# 単一ファイルの取り込み
unit = svc.ingest(Path("report.pdf"))
unit = svc.ingest(
    Path("doc.txt"),
    knowledge_id="custom_id",  # 省略時はファイル名のステム
    title="レポート",
    source="手動アップロード",
    overwrite=False,           # True で上書き
)

# ディレクトリ一括取り込み（対応外・エラーはスキップ）
units = svc.ingest_directory(Path("docs/"))
units = svc.ingest_directory(Path("docs/"), overwrite=True)

# 対応拡張子の確認
svc.supported_extensions  # [".txt", ".md", ".csv", ...]
```

### 変換関数（単体利用）

```python
from services.ingestion.document_to_markdown.converters import convert_to_markdown

markdown = convert_to_markdown(Path("file.csv"))
```

## エラーハンドリング

| 状況 | 例外 |
| ---- | ---- |
| ファイルが存在しない | `FileNotFoundError` |
| 対応していない拡張子 | `ValueError` |
| 同名の knowledge_id が存在する | `FileExistsError`（overwrite=False 時）|
| ディレクトリが存在しない | `NotADirectoryError` |

`ingest_directory()` はファイルごとのエラーをスキップして成功分だけを返します。

## テスト

```bash
# ユニットテスト（LLM 不要）
docker compose run --rm ingestion
```

テストは `tests/ingestion/` にあります。

- `test_converters.py` — 各形式の変換（PDF / DOCX は mock）
- `test_ingestion_service.py` — ingest / ingest_directory の動作
