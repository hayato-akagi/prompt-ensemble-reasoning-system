# Dataset Format Guide

このシステムで使用するデータセットの形式を説明します。

---

## 概要

本システムは **ログテキスト → 多クラスラベル分類** を行います。
推論は各ラベルに対して個別に Yes/No 質問を行い、全ラベルの結果を統合して分類を決定します。

---

## 1. 分類データセット（バッチ実行用）

Experiments ページの「バッチ実行」タブでアップロードするファイルです。

### ファイル形式

- ファイル名: 任意（例: `dataset.json`）
- 保存先: ローカルで管理し、UI からアップロード
- 文字コード: UTF-8
- 形式: JSON 配列

### スキーマ

```json
[
  {
    "log_id":      "<string>  — ログの一意 ID（例: log_001）",
    "log_text":    "<string>  — 分類対象のログテキスト",
    "ground_truth": ["<label>", ...]  — 正解ラベルのリスト（1つ以上）
  }
]
```

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `log_id` | string | 必須 | ログを識別する一意の文字列 |
| `log_text` | string | 必須 | 分類対象のログ本文 |
| `ground_truth` | list[string] | 必須 | 正解ラベルのリスト。複数可（複合障害など） |

### 例

```json
[
  {
    "log_id": "log_001",
    "log_text": "インバータ E01 エラーが発生。起動直後から過電流トリップを繰り返している。",
    "ground_truth": ["electrical"]
  },
  {
    "log_id": "log_002",
    "log_text": "PLC とサーボ間の通信タイムアウト（E10）が頻発。プログラムがフリーズ。",
    "ground_truth": ["software"]
  },
  {
    "log_id": "log_003",
    "log_text": "モーター過電流（E01）と軸受け摩耗が同時に確認された。",
    "ground_truth": ["electrical", "mechanical"]
  }
]
```

### 注意点

- `ground_truth` のラベル名は、実験設定の「分類ラベル」と完全一致させること
- 大文字・小文字を区別する（例: `"electrical"` と `"Electrical"` は別扱い）
- 正解ラベルが複数ある場合（複合障害）は配列に並べる
- ラベルが存在しないケース（正常ログなど）も扱う場合は `[]` を使用

---

## 2. ナレッジベース

推論時に参照する知識ソースです。
Knowledge ページから登録するか、`data/knowledge/` に直接配置します。

### ファイル形式

- **Markdown（`.md`）**: 推奨。見出し・箇条書きで構造化して記述
- **テキスト（`.txt`）**: プレーンテキスト
- **JSON（`.json`）**: `{"title": "...", "content": "..."}` 形式

### 配置場所

```
data/knowledge/
  ├── electrical_trouble_signs.md   # 電気系トラブルの特徴
  ├── software_trouble_signs.md     # ソフト系トラブルの特徴
  ├── mechanical_trouble_signs.md   # メカ系トラブルの特徴
  └── error_code_E01.md             # エラーコード解説など
```

### 記述のポイント

- 分類したいラベルに対応した知識を1ファイル1トピックで記述する
- エラーコードと症状の対応、典型的な現象を具体的に書く
- 短く明確な文体が推論精度に有利（1ファイル 200〜500 字程度が目安）

### 例（`electrical_trouble_signs.md`）

```markdown
電気系トラブルの特徴的なサイン

- 過電流エラー（E01）
- 絶縁抵抗の低下
- ブレーカーのトリップ
- モーターの焼損・異常発熱
- アーク放電・焦げ臭い

代表的なエラーコード: E01（過電流）、E03（地絡）、E05（短絡）
```

---

## 3. ラベル設計のガイドライン

### 基本方針

- **排他的でなくて良い**: 1つのログが複数ラベルに該当しても問題ない
- **3〜7 ラベル程度が推奨**: ラベルが多いほど推論コスト（= ラベル数 × N 回）が増加する
- **英語ラベル推奨**: 小規模モデルでは英語ラベルの方が JSON 出力が安定する

### 質問テンプレートの書き方

`{label}` プレースホルダーを含む英語の Yes/No 質問を設定します。

```
Is this log related to a {label} failure?
```

- `{label}` が各ラベル名に置換されて推論される
- 日本語より英語の方が小規模モデル（7B 以下）で安定
- 疑問文の形式（Does / Is / Has）が推奨

### ラベル設計の例

| ドメイン | ラベル例 | 質問テンプレート |
|---|---|---|
| 設備トラブル分類 | `electrical`, `software`, `mechanical` | `Is this log related to a {label} failure?` |
| IT インシデント分類 | `network`, `server`, `application`, `security` | `Is this incident caused by a {label} issue?` |
| 製品不良分類 | `design`, `material`, `process`, `inspection` | `Is this defect attributed to a {label} problem?` |
| 顧客問い合わせ分類 | `billing`, `technical`, `shipping`, `general` | `Does this inquiry belong to the {label} category?` |

---

## 4. 評価指標

バッチ実行後に自動計算される指標です。

| 指標 | 説明 |
|---|---|
| **完全一致率 (Exact Match)** | `predicted_labels の集合 == ground_truth の集合` の割合 |
| **平均 Jaccard** | `|予測 ∩ 正解| / |予測 ∪ 正解|` の平均。部分一致も評価できる |

### Jaccard の例

| 正解 | 予測 | Jaccard |
|---|---|---|
| `[electrical]` | `[electrical]` | 1.0（完全一致） |
| `[electrical, mechanical]` | `[electrical]` | 0.5（部分一致） |
| `[electrical]` | `[software]` | 0.0（完全不一致） |
| `[electrical, mechanical]` | `[electrical, mechanical]` | 1.0（完全一致） |

---

## 5. ファイル配置まとめ

```
data/
  knowledge/
    *.md / *.txt / *.json    — ナレッジファイル（Knowledge ページで登録）
    sample_dataset.json      — サンプル分類データセット
  eval_results/
    <run_id>/
      summary.json           — B1〜B4 ベースライン比較結果
      B1_predictions.json    — B1 個別予測
      n_curve.json           — N-accuracy curve（オプション）
  experiments/
    <experiment_id>/
      config.json            — 実験設定（ラベル・テンプレート・N など）
      class_results.json     — バッチ分類結果
```
