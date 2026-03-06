# scripts — モデルダウンローダー

`config/models.json` に登録された GGUF モデルを HuggingFace Hub からダウンロードします。

## 使い方

### モデル一覧を確認

```bash
docker compose run --rm downloader --list
```

### モデルをダウンロード（モデル ID 指定）

```bash
# ダウンロードのみ
docker compose run --rm downloader --model-id qwen2.5-3b-instruct-q4_k_m

# ダウンロード + config/inference.json の active_model を更新
docker compose run --rm downloader --model-id qwen2.5-3b-instruct-q4_k_m --set-active
```

### モデルをダウンロード（リポジトリ直接指定）

`config/models.json` に登録されていないモデルを直接取得する場合:

```bash
docker compose run --rm downloader \
  --repo-id Qwen/Qwen2.5-7B-Instruct-GGUF \
  --filename qwen2.5-7b-instruct-q4_k_m.gguf
```

## ダウンロード先

```
data/models/<filename>.gguf
```

ホスト側の `data/models/` にダウンロードされ、UI コンテナや inference コンテナと共有されます。

## 登録モデル一覧（config/models.json）

| ID | サイズ | 説明 |
| -- | ------ | ---- |
| `qwen2.5-3b-instruct-q4_k_m` | ~2.0GB | 軽量・高速 |
| `qwen2.5-7b-instruct-q4_k_m` | ~4.4GB | バランス型 |
| `qwen2.5-14b-instruct-q4_k_m` | ~8.4GB | 高精度 |

新しいモデルを追加する場合は `config/models.json` の `models` 配列にエントリを追加してください。
