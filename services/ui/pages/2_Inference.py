"""
Inference page — multi-class log classification via per-label Yes/No ensemble inference.

Users configure:
  - Labels (e.g. electrical, software, mechanical)
  - Question template with {label} placeholder
  - Ensemble method (preset baseline or custom)
  - Log text to classify

The system runs one Yes/No ensemble inference per label and displays a classification result.
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path

from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

st.set_page_config(page_title="Inference", layout="wide")
st.title("Inference")
st.caption("ログをラベルで多クラス分類（各ラベルに Yes/No アンサンブル推論）")

_ROOT = Path(__file__).resolve().parents[3]


def _load_inference_cfg() -> dict:
    with open(_ROOT / "config" / "inference.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar: knowledge status
# ---------------------------------------------------------------------------
km = KnowledgeManager()
all_ids = km.list_ids()
st.sidebar.metric("ナレッジ件数", len(all_ids))
if not all_ids:
    st.sidebar.warning("ナレッジが登録されていません。Knowledge ページで登録してください。")

cfg = _load_inference_cfg()
knowledge_token_limit: int = cfg.get("knowledge_token_limit", 512)

# ---------------------------------------------------------------------------
# Section 1: Classification settings
# ---------------------------------------------------------------------------
st.subheader("分類設定")

col_left, col_right = st.columns([1, 1])

with col_left:
    labels_raw = st.text_area(
        "分類ラベル（1行1ラベル）",
        value="electrical\nsoftware\nmechanical",
        height=120,
        help="ログを分類するカテゴリラベルを1行に1つ入力してください。",
    )
    labels = [l.strip() for l in labels_raw.strip().splitlines() if l.strip()]

    question_template = st.text_input(
        "質問テンプレート（{label} プレースホルダー必須）",
        value="Is this log related to a {label} failure?",
        help="各ラベルに対して生成される Yes/No 質問のテンプレートです。{label} が各ラベルに置換されます。",
    )

with col_right:
    st.markdown("**アンサンブル設定**")
    preset = st.radio(
        "プリセット",
        options=[
            "B1: 単回推論",
            "B2: 温度多様性 (N=5, 知識固定)",
            "B3: RKSSE (N=5, ランダム知識)",
            "B4: 多数決 (N=5, ランダム知識)",
            "カスタム",
        ],
        index=2,
    )

    if preset == "B1: 単回推論":
        n_ensemble = 1
        knowledge_sampling = "all"
        aggregation = "weighted"
    elif preset == "B2: 温度多様性 (N=5, 知識固定)":
        n_ensemble = 5
        knowledge_sampling = "all"
        aggregation = "weighted"
    elif preset == "B3: RKSSE (N=5, ランダム知識)":
        n_ensemble = 5
        knowledge_sampling = "random"
        aggregation = "weighted"
    elif preset == "B4: 多数決 (N=5, ランダム知識)":
        n_ensemble = 5
        knowledge_sampling = "random"
        aggregation = "majority"
    else:
        n_ensemble = st.slider("N（アンサンブル回数）", 1, 20, 5)
        knowledge_sampling = st.selectbox("知識サンプリング", ["random", "all"])
        aggregation = st.selectbox("集約方法", ["weighted", "majority"])

    st.caption(f"N={n_ensemble} / サンプリング={knowledge_sampling} / 集約={aggregation}")

# Validate
if "{label}" not in question_template:
    st.error("質問テンプレートに `{label}` が含まれていません。")
    st.stop()

if not labels:
    st.warning("分類ラベルを1つ以上入力してください。")
    st.stop()

# Preview questions
with st.expander("生成される質問のプレビュー"):
    for lbl in labels:
        st.markdown(f"- `{lbl}` → {question_template.format(label=lbl)}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Knowledge set selection
# ---------------------------------------------------------------------------
st.subheader("ナレッジセット")

if all_ids:
    selected_knowledge_ids = st.multiselect(
        "使用するナレッジを選択（未選択 = 全件使用）",
        options=all_ids,
        default=[],
        help="推論に使用するナレッジを絞り込めます。未選択の場合は登録済み全件を使用します。",
        key="infer_knowledge_ids",
    )
    if selected_knowledge_ids:
        st.caption(f"{len(selected_knowledge_ids)} 件を選択中（全 {len(all_ids)} 件中）")
        active_units = [km.load(kid) for kid in selected_knowledge_ids]
    else:
        st.caption(f"全 {len(all_ids)} 件を使用")
        active_units = km.load_all()
else:
    selected_knowledge_ids = []
    active_units = []

if knowledge_token_limit > 0:
    st.caption(f"ナレッジ自動要約: 有効（閾値 {knowledge_token_limit} トークン）")
else:
    st.caption("ナレッジ自動要約: 無効（Settings で設定可）")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Log input
# ---------------------------------------------------------------------------
st.subheader("ログ入力")

log_text = st.text_area(
    "分類対象のログ",
    height=180,
    placeholder=(
        "例: モーター過電流エラーが複数回発生。電源投入直後から断続的に発生しており、"
        "電流値は定格の1.5倍を記録。"
    ),
    key="infer_log",
)

summarize_log_opt = st.checkbox(
    f"ログが {knowledge_token_limit} トークンを超える場合は要約する",
    key="infer_summarize_log",
    help="エラーコード・異常値など重要情報を保持したまま要約します。",
    disabled=(knowledge_token_limit == 0),
)

run_btn = st.button("推論実行", type="primary", disabled=not log_text.strip())

# ---------------------------------------------------------------------------
# Section 4: Result
# ---------------------------------------------------------------------------
if run_btn:
    if not active_units:
        st.error("ナレッジが登録されていません。Knowledge ページで登録してください。")
        st.stop()

    # --- ナレッジ自動要約 ---
    active_texts = km.texts(active_units)
    if knowledge_token_limit > 0:
        needs_check = [u for u in active_units if len(u.effective_text) > knowledge_token_limit * 2]
        if needs_check:
            from services.inference.llm_inference_service.summarizer import SummarizationService
            summarizer = SummarizationService()
            updated = False
            with st.spinner(f"ナレッジのトークン数を確認・要約中（対象 {len(needs_check)} 件）..."):
                for unit in needs_check:
                    effective = unit.effective_text
                    if summarizer.needs_summarization(effective, knowledge_token_limit):
                        summary = summarizer.summarize_knowledge(effective)
                        km.save_summary(unit.knowledge_id, summary)
                        updated = True
            if updated:
                active_units = [km.load(u.knowledge_id) for u in active_units]
                active_texts = km.texts(active_units)
                st.info("長いナレッジを要約しました。")
            del summarizer

    # --- ログ要約 ---
    log_to_use = log_text.strip()
    if summarize_log_opt and knowledge_token_limit > 0:
        from services.inference.llm_inference_service.summarizer import SummarizationService
        summarizer = SummarizationService()
        with st.spinner("ログのトークン数を確認・要約中..."):
            log_to_use, was_summarized = summarizer.maybe_summarize_log(log_text.strip(), knowledge_token_limit)
        if was_summarized:
            st.info("ログを要約しました。")
            with st.expander("要約後のログ"):
                st.text(log_to_use)
        del summarizer

    with st.spinner(f"推論中（{len(labels)} ラベル × N={n_ensemble}）..."):
        from services.inference.llm_inference_service.classifier import ClassificationService
        svc = ClassificationService(
            labels=labels,
            question_template=question_template,
            n_ensemble=n_ensemble,
            knowledge_sampling=knowledge_sampling,
            aggregation=aggregation,
        )
        try:
            result = svc.classify(knowledge_texts=active_texts, log=log_to_use)
        except RuntimeError as e:
            st.error(f"推論エラー: {e}")
            st.stop()

    st.session_state["infer_result"] = result

if "infer_result" in st.session_state:
    result = st.session_state["infer_result"]

    st.subheader("分類結果")

    col1, col2 = st.columns(2)
    with col1:
        predicted_str = "、".join(result.predicted_labels) if result.predicted_labels else "（なし）"
        st.metric("予測ラベル (Yes)", predicted_str)
    with col2:
        st.metric("最有力ラベル (Top)", result.top_label)

    st.markdown("**ラベル別詳細**")
    rows = []
    for lr in result.label_details:
        rows.append({
            "ラベル": lr.label,
            "質問": lr.question,
            "判定": "✓ Yes" if lr.answer == "yes" else "✗ No",
            "Confidence": round(lr.confidence, 3),
            "Yes 比率": round(lr.yes_ratio, 3),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    chart_df = pd.DataFrame({
        "Confidence": {lr.label: lr.confidence for lr in result.label_details},
        "Yes 比率": {lr.label: lr.yes_ratio for lr in result.label_details},
    })
    st.bar_chart(chart_df)

    st.markdown("**推論理由**")
    for lr in result.label_details:
        icon = "✅" if lr.answer == "yes" else "❌"
        with st.expander(f"{icon} {lr.label}（{lr.answer}, conf={lr.confidence:.3f}）"):
            st.write(lr.reason or "（理由なし）")
