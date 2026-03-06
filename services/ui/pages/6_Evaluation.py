"""
Evaluation page — visualize baseline comparison results from data/eval_results/.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation")
st.caption("B1〜B4 ベースライン比較・評価指標の可視化")

_ROOT = Path(__file__).resolve().parents[3]
_LABELS_PATH = _ROOT / "data" / "knowledge" / "eval_labels.json"
_RESULTS_DIR = _ROOT / "data" / "eval_results"

# ---------------------------------------------------------------------------
# Section 1: Eval data status
# ---------------------------------------------------------------------------
st.subheader("評価データ")

if _LABELS_PATH.exists():
    with open(_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    difficulties = [l.get("difficulty", "unknown") for l in labels]
    diff_counts = pd.Series(difficulties).value_counts()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("サンプル数", len(labels))
    with col2:
        st.metric("easy", diff_counts.get("easy", 0))
    with col3:
        st.metric("medium", diff_counts.get("medium", 0))
    with col4:
        st.metric("hard", diff_counts.get("hard", 0))

    with st.expander("サンプル一覧"):
        rows = []
        for item in labels:
            for q, ans in item.get("labels", {}).items():
                rows.append({"log_id": item["log_id"], "difficulty": item.get("difficulty", "—"),
                             "question": q[:60], "answer": ans, "note": item.get("note", "")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.warning(
        "`data/knowledge/eval_labels.json` が見つかりません。\n\n"
        "README.md の「正解ラベルの形式」を参照してファイルを作成してください。"
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 2: How to run
# ---------------------------------------------------------------------------
st.subheader("評価の実行方法")
st.info(
    "評価は LLM を使った実推論のため、CLI スクリプトで実行してください。\n\n"
    "```bash\n"
    "# B1〜B4 すべてを実行\n"
    "docker compose run --rm inference python scripts/run_evaluation.py\n\n"
    "# N-accuracy curve も計測\n"
    "docker compose run --rm inference python scripts/run_evaluation.py --n-curve\n\n"
    "# 特定のベースラインのみ\n"
    "docker compose run --rm inference python scripts/run_evaluation.py --baselines B3 B4\n"
    "```"
)

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Results
# ---------------------------------------------------------------------------
st.subheader("評価結果")

if not _RESULTS_DIR.exists() or not any(_RESULTS_DIR.iterdir()):
    st.info("評価結果がありません。上記コマンドで評価を実行してください。")
    st.stop()

run_dirs = sorted(
    [d for d in _RESULTS_DIR.iterdir() if d.is_dir()],
    reverse=True,
)
run_ids = [d.name for d in run_dirs]
selected_run = st.selectbox("実行を選択", run_ids, key="sel_run")
run_dir = _RESULTS_DIR / selected_run
summary_path = run_dir / "summary.json"

if not summary_path.exists():
    st.error("summary.json が見つかりません。")
    st.stop()

with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)

st.caption(f"サンプル数: {summary['n_items']}  |  実行 ID: {summary['run_id']}")

# Metrics table
baselines_data = summary["baselines"]
df_metrics = pd.DataFrame([
    {
        "ベースライン": b["label"],
        "N": b["n_ensemble"],
        "知識サンプリング": b["knowledge_sampling"],
        "集約": b["aggregation"],
        "Accuracy": b["accuracy"],
        "F1 (yes)": b["f1"],
        "Precision": b["precision"],
        "Recall": b["recall"],
        "ECE": b["ece"],
    }
    for b in baselines_data
])
st.dataframe(df_metrics, use_container_width=True)

# Bar charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy 比較")
    acc_df = pd.DataFrame({
        b["baseline"]: [b["accuracy"]] for b in baselines_data
    })
    st.bar_chart(acc_df)

with col2:
    st.subheader("ECE 比較（低いほど良い）")
    ece_df = pd.DataFrame({
        b["baseline"]: [b["ece"]] for b in baselines_data
    })
    st.bar_chart(ece_df)

# F1 chart
st.subheader("F1 / Precision / Recall 比較")
prf_df = pd.DataFrame([
    {"ベースライン": b["baseline"], "F1": b["f1"],
     "Precision": b["precision"], "Recall": b["recall"]}
    for b in baselines_data
]).set_index("ベースライン")
st.bar_chart(prf_df)

# ---------------------------------------------------------------------------
# Section 4: N-accuracy curve
# ---------------------------------------------------------------------------
n_curve_path = run_dir / "n_curve.json"
if n_curve_path.exists():
    st.divider()
    st.subheader("N-accuracy curve（B3 スタイル: ランダム知識 + 重み付き集約）")
    with open(n_curve_path, encoding="utf-8") as f:
        n_data = json.load(f)
    n_df = pd.DataFrame(n_data).set_index("n")[["accuracy", "f1", "ece"]]
    st.line_chart(n_df)
    st.dataframe(n_df.reset_index(), use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Per-prediction details
# ---------------------------------------------------------------------------
st.divider()
st.subheader("個別予測の詳細")

available_baselines = [
    b["baseline"] for b in baselines_data
    if (run_dir / f"{b['baseline']}_predictions.json").exists()
]

if available_baselines:
    sel_b = st.selectbox("ベースライン", available_baselines, key="sel_detail")
    with open(run_dir / f"{sel_b}_predictions.json", encoding="utf-8") as f:
        preds = json.load(f)

    filter_diff = st.multiselect(
        "難易度フィルタ",
        options=sorted({p["difficulty"] for p in preds}),
        default=sorted({p["difficulty"] for p in preds}),
    )
    preds_filtered = [p for p in preds if p["difficulty"] in filter_diff]

    pred_df = pd.DataFrame([
        {
            "log_id": p["log_id"],
            "difficulty": p["difficulty"],
            "question": p["question"][:45],
            "predicted": p["predicted"],
            "ground_truth": p["ground_truth"],
            "correct": "✓" if p["predicted"] == p["ground_truth"] else "✗",
            "confidence": round(p["confidence"], 3),
            "yes_ratio": round(p["yes_ratio"], 3),
        }
        for p in preds_filtered
    ])

    n_correct = (pred_df["correct"] == "✓").sum()
    st.caption(f"{n_correct} / {len(pred_df)} 件正解")
    st.dataframe(pred_df, use_container_width=True)

    # Confidence distribution by correctness
    st.subheader("Confidence 分布（正解 vs 不正解）")
    correct_conf = [p["confidence"] for p in preds_filtered if p["predicted"] == p["ground_truth"]]
    wrong_conf = [p["confidence"] for p in preds_filtered if p["predicted"] != p["ground_truth"]]

    col1, col2 = st.columns(2)
    with col1:
        if correct_conf:
            st.metric("正解の平均 confidence", f"{sum(correct_conf)/len(correct_conf):.3f}")
    with col2:
        if wrong_conf:
            st.metric("不正解の平均 confidence", f"{sum(wrong_conf)/len(wrong_conf):.3f}")
