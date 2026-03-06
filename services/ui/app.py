"""
Prompt Ensemble Reasoning System — Streamlit UI (home page).
"""

import streamlit as st

st.set_page_config(
    page_title="Prompt Ensemble Reasoning System",
    page_icon="robot",
    layout="wide",
)

st.title("Prompt Ensemble Reasoning System")
st.caption("ローカル GGUF モデルを使ったアンサンブル推論プラットフォーム")

st.markdown("""
## ページ一覧

| ページ | 説明 |
|--------|------|
| **Knowledge** | ナレッジの閲覧・追加・編集・削除 |
| **Inference** | ログ入力 → アンサンブル推論の実行 |
| **Experiments** | 実験の作成・実行・比較 |
| **Templates** | プロンプトテンプレートの編集 |
| **Settings** | モデル・推論パラメータの設定 |

サイドバーからページを選択してください。
""")
