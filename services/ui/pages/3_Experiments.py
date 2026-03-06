"""
Experiments page — batch multi-class classification experiments with accuracy metrics.

Tabs:
  - 作成: Create experiment with labels + question template + ensemble settings
  - バッチ実行: Upload a dataset (JSON with log_text + ground_truth) and run batch classification
  - 結果一覧: View per-log classification results with Jaccard/exact-match metrics
  - 比較: Compare accuracy metrics across experiments
"""

import json
import pandas as pd
import streamlit as st

from services.experiment.experiment_manager.experiment_manager import ExperimentManager
from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

st.set_page_config(page_title="Experiments", layout="wide")
st.title("Experiments")
st.caption("ラベルベース多クラス分類実験 — データセット一括評価")

em = ExperimentManager()
km = KnowledgeManager()

tab_create, tab_run, tab_results, tab_compare = st.tabs(
    ["作成", "バッチ実行", "結果一覧", "比較"]
)


# ===========================================================================
# CREATE
# ===========================================================================
with tab_create:
    st.subheader("実験を作成")
    st.caption("ラベル・質問テンプレート・アンサンブル設定を定義して実験を登録します。")

    exp_id = st.text_input("実験 ID *", placeholder="例: exp_electrical_v1", key="new_exp_id")
    description = st.text_area("説明", height=60, key="new_exp_desc")

    st.markdown("**分類設定**")
    col_labels, col_template = st.columns([1, 2])
    with col_labels:
        labels_raw = st.text_area(
            "分類ラベル（1行1ラベル）",
            value="electrical\nsoftware\nmechanical",
            height=110,
            key="new_exp_labels",
        )
        labels = [l.strip() for l in labels_raw.strip().splitlines() if l.strip()]
    with col_template:
        question_template = st.text_input(
            "質問テンプレート（{label} 必須）",
            value="Is this log related to a {label} failure?",
            key="new_exp_qtpl",
        )
        if labels and "{label}" in question_template:
            st.caption("生成される質問例:")
            for lbl in labels[:3]:
                st.caption(f"　• {question_template.format(label=lbl)}")

    st.markdown("**アンサンブル設定**")
    col1, col2, col3 = st.columns(3)
    with col1:
        preset = st.selectbox(
            "プリセット",
            ["B1: 単回推論", "B2: 温度多様性 (N=5, 知識固定)", "B3: RKSSE (N=5, ランダム知識)", "B4: 多数決 (N=5, ランダム知識)", "カスタム"],
            index=2,
            key="new_exp_preset",
        )
    with col2:
        if preset == "B1: 単回推論":
            n_ensemble = 1; knowledge_sampling = "all"; aggregation = "weighted"
        elif preset == "B2: 温度多様性 (N=5, 知識固定)":
            n_ensemble = 5; knowledge_sampling = "all"; aggregation = "weighted"
        elif preset == "B3: RKSSE (N=5, ランダム知識)":
            n_ensemble = 5; knowledge_sampling = "random"; aggregation = "weighted"
        elif preset == "B4: 多数決 (N=5, ランダム知識)":
            n_ensemble = 5; knowledge_sampling = "random"; aggregation = "majority"
        else:
            n_ensemble = st.number_input("N（アンサンブル回数）", 1, 20, 5, key="new_exp_n")
            knowledge_sampling = st.selectbox("知識サンプリング", ["random", "all"], key="new_exp_ks")
            aggregation = st.selectbox("集約方法", ["weighted", "majority"], key="new_exp_agg")
        if preset != "カスタム":
            st.caption(f"N={n_ensemble} / {knowledge_sampling} / {aggregation}")
    with col3:
        template_names = []
        try:
            from services.inference.llm_inference_service.prompt_template_manager import PromptTemplateManager
            template_names = PromptTemplateManager().list_names()
        except Exception:
            pass
        template_name = st.selectbox(
            "プロンプトテンプレート", template_names or ["default"], key="new_exp_tpl"
        )
        max_ku = st.number_input("最大ナレッジ数 (0=全件)", min_value=0, value=0, key="new_exp_maxku")

    st.markdown("**ナレッジセット**")
    all_knowledge_ids = km.list_ids()
    if all_knowledge_ids:
        selected_knowledge_ids = st.multiselect(
            "使用するナレッジを選択（未選択 = 全件使用）",
            options=all_knowledge_ids,
            default=[],
            help="この実験で使用するナレッジを絞り込めます。未選択の場合は実行時点の全件を使用します。",
            key="new_exp_knowledge_ids",
        )
        if selected_knowledge_ids:
            st.caption(f"{len(selected_knowledge_ids)} 件を選択（全 {len(all_knowledge_ids)} 件中）")
        else:
            st.caption(f"全 {len(all_knowledge_ids)} 件を使用（実行時点）")
    else:
        selected_knowledge_ids = []
        st.caption("ナレッジが登録されていません。Knowledge ページで登録してください。")

    overwrite = st.checkbox("既存の実験を上書き", key="new_exp_overwrite")

    if st.button("作成", type="primary", key="btn_create"):
        if not exp_id.strip():
            st.warning("実験 ID は必須です。")
        elif not labels:
            st.warning("ラベルを1つ以上入力してください。")
        elif "{label}" not in question_template:
            st.warning("質問テンプレートに {label} が必要です。")
        else:
            try:
                cfg = em.create(
                    experiment_id=exp_id.strip(),
                    description=description,
                    template_name=template_name,
                    n_ensemble=int(n_ensemble),
                    max_knowledge_units=int(max_ku) if max_ku > 0 else None,
                    labels=labels,
                    question_template=question_template,
                    knowledge_ids=selected_knowledge_ids,
                    overwrite=overwrite,
                )
                km_note = f"ナレッジ: {', '.join(selected_knowledge_ids)}" if selected_knowledge_ids else "ナレッジ: 全件"
                st.success(f"実験 `{cfg.experiment_id}` を作成しました（ラベル: {', '.join(labels)} / {km_note}）。")
            except FileExistsError:
                st.error("同名の実験が存在します。「上書き」を有効にするか別の ID を使ってください。")
            except Exception as e:
                st.error(f"エラー: {e}")


# ===========================================================================
# BATCH RUN
# ===========================================================================
with tab_run:
    st.subheader("バッチ分類実行")
    st.caption(
        "JSON データセットをアップロードして全ログを一括分類します。\n\n"
        "データセット形式（JSON 配列）:\n"
        "```json\n"
        "[\n"
        "  {\"log_id\": \"log_001\", \"log_text\": \"...\", \"ground_truth\": [\"electrical\"]},\n"
        "  {\"log_id\": \"log_002\", \"log_text\": \"...\", \"ground_truth\": [\"software\", \"mechanical\"]}\n"
        "]\n"
        "```"
    )

    exp_ids = em.list_ids()
    if not exp_ids:
        st.info("実験がありません。「作成」タブで実験を作成してください。")
    else:
        selected_exp = st.selectbox("実験を選択", exp_ids, key="run_exp")

        try:
            cfg = em.load_config(selected_exp)
        except Exception as e:
            st.error(f"設定の読み込みエラー: {e}")
            st.stop()

        with st.expander("実験設定"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**ラベル:** {', '.join(cfg.labels) if cfg.labels else '（未設定）'}")
                st.markdown(f"**テンプレート:** {cfg.question_template}")
            with col_b:
                st.markdown(f"**N:** {cfg.n_ensemble}")
                km_ids_note = ", ".join(cfg.knowledge_ids) if cfg.knowledge_ids else "全件"
                st.markdown(f"**ナレッジセット:** {km_ids_note}")

        if not cfg.labels:
            st.warning("この実験にラベルが設定されていません。「作成」タブで再作成してください。")
        else:
            # Resolve knowledge texts for this experiment
            if cfg.knowledge_ids:
                try:
                    _units = [km.load(kid) for kid in cfg.knowledge_ids]
                    exp_texts = km.texts(_units)
                    st.caption(f"ナレッジ: {len(exp_texts)} 件（実験設定で固定）")
                except Exception as e:
                    st.warning(f"ナレッジの読み込みエラー: {e}。全件を使用します。")
                    exp_texts = km.texts()
            else:
                exp_texts = km.texts()
                st.caption(f"ナレッジ: 全 {len(exp_texts)} 件")

            uploaded = st.file_uploader(
                "データセット JSON をアップロード",
                type=["json"],
                key="run_dataset",
                help="log_id, log_text, ground_truth（list[str]）を含む JSON 配列",
            )

            dataset = None
            if uploaded is not None:
                try:
                    dataset = json.load(uploaded)
                    st.success(f"{len(dataset)} 件のログを読み込みました。")
                    preview_df = pd.DataFrame([
                        {
                            "log_id": d.get("log_id", ""),
                            "log_text（先頭60字）": str(d.get("log_text", ""))[:60],
                            "ground_truth": ", ".join(d.get("ground_truth", [])),
                        }
                        for d in dataset[:5]
                    ])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    if len(dataset) > 5:
                        st.caption(f"（先頭5件を表示。全{len(dataset)}件）")
                except Exception as e:
                    st.error(f"JSON 解析エラー: {e}")
                    dataset = None

            run_disabled = dataset is None or not exp_texts
            if not exp_texts:
                st.warning("ナレッジが登録されていません。")

            if st.button("バッチ実行", type="primary", disabled=run_disabled, key="btn_batch_run"):
                from services.inference.llm_inference_service.classifier import ClassificationService
                svc = ClassificationService(
                    labels=cfg.labels,
                    question_template=cfg.question_template,
                    n_ensemble=cfg.n_ensemble,
                    knowledge_sampling="random",
                    aggregation="weighted",
                )

                progress = st.progress(0)
                status = st.empty()
                results_saved = []

                for i, item in enumerate(dataset):
                    log_text = item.get("log_text", "")
                    ground_truth = item.get("ground_truth", [])
                    log_id = item.get("log_id", f"log_{i+1}")

                    status.caption(f"[{i+1}/{len(dataset)}] {log_id} を推論中...")

                    try:
                        cls_result = svc.classify(knowledge_texts=exp_texts, log=log_text)
                        run_result = em.save_class_result(
                            selected_exp, log_text, cls_result, ground_truth
                        )
                        results_saved.append(run_result)
                    except Exception as e:
                        st.warning(f"{log_id}: エラー — {e}")

                    progress.progress((i + 1) / len(dataset))

                status.empty()
                n_correct = sum(r.exact_match for r in results_saved)
                avg_jaccard = sum(r.jaccard for r in results_saved) / len(results_saved) if results_saved else 0.0
                st.success(
                    f"完了: {len(results_saved)} 件処理 / "
                    f"完全一致 {n_correct}/{len(results_saved)} "
                    f"({n_correct/len(results_saved)*100:.1f}%) / "
                    f"平均 Jaccard {avg_jaccard:.3f}"
                )


# ===========================================================================
# RESULTS
# ===========================================================================
with tab_results:
    st.subheader("結果一覧")

    exp_ids = em.list_ids()
    if not exp_ids:
        st.info("実験がありません。")
    else:
        selected_exp_r = st.selectbox("実験を選択", exp_ids, key="res_exp")

        try:
            cfg_r = em.load_config(selected_exp_r)
            class_results = em.load_class_results(selected_exp_r)
        except Exception as e:
            st.error(f"エラー: {e}")
            class_results = []

        if not class_results:
            st.info("分類結果がまだありません。「バッチ実行」タブで実行してください。")
        else:
            n_total = len(class_results)
            n_exact = sum(r.exact_match for r in class_results)
            avg_jaccard = sum(r.jaccard for r in class_results) / n_total

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ログ数", n_total)
            with col2:
                st.metric("完全一致率", f"{n_exact/n_total*100:.1f}% ({n_exact}/{n_total})")
            with col3:
                st.metric("平均 Jaccard", f"{avg_jaccard:.3f}")

            summary_rows = []
            for r in class_results:
                summary_rows.append({
                    "timestamp": r.timestamp[:19],
                    "log（先頭50字）": r.log_input[:50],
                    "正解": ", ".join(r.ground_truth),
                    "予測": ", ".join(r.predicted_labels) if r.predicted_labels else "（なし）",
                    "Top": r.top_label,
                    "完全一致": "✓" if r.exact_match else "✗",
                    "Jaccard": round(r.jaccard, 3),
                })
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("**ラベル別詳細**")
            sel_idx = st.selectbox(
                "詳細を表示するログ",
                options=list(range(len(class_results))),
                format_func=lambda i: f"[{i+1}] {class_results[i].log_input[:60]}",
                key="res_detail_idx",
            )
            if sel_idx is not None:
                r = class_results[sel_idx]
                st.caption(f"正解: {', '.join(r.ground_truth)}  |  予測: {', '.join(r.predicted_labels) or '（なし）'}  |  Jaccard: {r.jaccard}")
                detail_rows = []
                for lp in r.label_predictions:
                    detail_rows.append({
                        "ラベル": lp.label,
                        "質問": lp.question,
                        "判定": "✓ Yes" if lp.answer == "yes" else "✗ No",
                        "Confidence": round(lp.confidence, 3),
                        "Yes 比率": round(lp.yes_ratio, 3),
                    })
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# COMPARE
# ===========================================================================
with tab_compare:
    st.subheader("実験間比較")
    st.caption("複数の実験の分類精度を比較します。")

    all_ids = em.list_ids()
    if len(all_ids) < 2:
        st.info("比較するには実験が 2 つ以上必要です。")
    else:
        selected_exps = st.multiselect(
            "比較する実験を選択（2つ以上）",
            all_ids,
            default=all_ids[:min(3, len(all_ids))],
            key="cmp_exps",
        )

        if len(selected_exps) >= 2:
            compare_rows = []
            for exp_id in selected_exps:
                try:
                    results = em.load_class_results(exp_id)
                    cfg_c = em.load_config(exp_id)
                except Exception:
                    continue
                if not results:
                    continue
                n = len(results)
                exact = sum(r.exact_match for r in results)
                avg_jac = sum(r.jaccard for r in results) / n
                compare_rows.append({
                    "実験 ID": exp_id,
                    "ラベル": ", ".join(cfg_c.labels) if cfg_c.labels else "—",
                    "N": cfg_c.n_ensemble,
                    "ログ数": n,
                    "完全一致率": round(exact / n, 4),
                    "平均 Jaccard": round(avg_jac, 4),
                })

            if compare_rows:
                cmp_df = pd.DataFrame(compare_rows)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                st.bar_chart(cmp_df.set_index("実験 ID")[["完全一致率", "平均 Jaccard"]])
            else:
                st.info("選択した実験に分類結果がありません。「バッチ実行」タブで実行してください。")
