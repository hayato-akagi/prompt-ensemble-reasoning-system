# ui service

Streamlit ベースの管理 UI です。全機能を GUI から操作できます。

## 起動

```bash
docker compose up ui
```

ブラウザで http://localhost:8501 を開いてください。

## ファイル構成

```
services/ui/
├── app.py            # ホームページ
├── pages/
│   ├── 1_Knowledge.py    # ナレッジ管理
│   ├── 2_Inference.py    # 推論実行
│   ├── 3_Experiments.py  # 実験管理
│   ├── 4_Templates.py    # テンプレート編集
│   └── 5_Settings.py     # 設定
├── requirements.txt
└── Dockerfile
```

## ページ詳細

### Knowledge

- ナレッジの一覧表示（title・source・サマリー有無を表示）
- ファイルアップロード（.txt / .md / .csv / .xlsx / .pdf / .docx）
- 手動入力（Markdown テキスト直接入力）
- 編集・サマリーキャッシュの追加/更新/削除
- 削除（確認ダイアログあり）

### Inference

- ログテキストと質問を入力してアンサンブル推論を実行
- 最大ナレッジ数・アンサンブル回数を調整可能
- 結果表示: answer / confidence / yes_ratio / reason
- 個別推論結果 (raw) の確認
- モデル未ダウンロード時はエラーメッセージとダウンロードコマンドを表示

### Experiments

| タブ | 説明 |
| ---- | ---- |
| 作成 | 実験 ID・テンプレート・アンサンブル回数・temperature などを設定して作成 |
| 推論実行 | 実験設定で推論を実行し結果を保存 |
| 結果一覧 | 保存済み結果を新しい順に表示 |
| 比較 | 2 つ以上の実験を横断比較（同じログ×質問ペアで並べて表示）|

### Templates

- テンプレート一覧から選択して編集・保存
- 新規テンプレートの作成
- 削除（`default` テンプレートは削除不可）
- プレースホルダー: `{knowledge}` `{log}` `{question}`

### Settings

- `active_model` の切り替え（`config/models.json` に登録されたモデルから選択）
- モデルのダウンロード状態を表示
- `active_template` の切り替え
- 推論パラメータ（n_ensemble / temperature / max_tokens / top_p / n_ctx / n_gpu_layers）
- 「設定を保存」で `config/inference.json` を更新

## ボリュームマウント

UI コンテナは以下のディレクトリをホストと共有します。

| コンテナ内パス | 用途 |
| -------------- | ---- |
| `/app/data/knowledge` | ナレッジファイル |
| `/app/data/models` | GGUF モデルファイル |
| `/app/data/prompts` | プロンプトテンプレート |
| `/app/data/experiments` | 実験設定・結果 |
| `/app/config` | inference.json / models.json |
