"""
Knowledge Manager page.
"""

import streamlit as st
from pathlib import Path
import tempfile

from services.knowledge.knowledge_manager.knowledge_manager import KnowledgeManager

st.set_page_config(page_title="Knowledge Manager", layout="wide")
st.title("Knowledge Manager")

km = KnowledgeManager()

# --- session state ---
if "km_mode" not in st.session_state:
    st.session_state.km_mode = "list"
if "km_selected_id" not in st.session_state:
    st.session_state.km_selected_id = None


def _set_mode(mode: str, selected_id: str | None = None) -> None:
    st.session_state.km_mode = mode
    st.session_state.km_selected_id = selected_id


# ===========================================================================
# LIST
# ===========================================================================
if st.session_state.km_mode == "list":
    ids = km.list_ids()

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.subheader(f"ナレッジ一覧 ({len(ids)} 件)")
    with col_btn:
        if st.button("+ 追加", use_container_width=True, type="primary"):
            _set_mode("add")
            st.rerun()

    if not ids:
        st.info("ナレッジがありません。「追加」ボタンで登録してください。")
    else:
        for kid in ids:
            try:
                unit = km.load(kid)
            except Exception as e:
                st.error(f"{kid}: {e}")
                continue

            label = f"**{unit.title or kid}** `{kid}`"
            if unit.summary:
                label += "  ✓ サマリーあり"

            with st.expander(label):
                st.caption(f"source: {unit.source or '—'}")
                st.text_area("内容", value=unit.text, height=120, disabled=True, key=f"view_{kid}")
                c1, c2, _ = st.columns([1, 1, 4])
                with c1:
                    if st.button("編集", key=f"edit_{kid}", use_container_width=True):
                        _set_mode("edit", kid)
                        st.rerun()
                with c2:
                    if st.button("削除", key=f"del_{kid}", use_container_width=True):
                        _set_mode("delete", kid)
                        st.rerun()


# ===========================================================================
# ADD
# ===========================================================================
elif st.session_state.km_mode == "add":
    st.subheader("ナレッジを追加")

    tab_upload, tab_manual = st.tabs(["ファイルアップロード", "手動入力"])

    with tab_upload:
        uploaded = st.file_uploader(
            "ファイルを選択 (.txt, .md, .csv, .xlsx, .pdf, .docx)",
            type=["txt", "md", "csv", "xlsx", "xls", "pdf", "docx"],
        )
        if uploaded:
            kid_u = st.text_input("Knowledge ID（空欄でファイル名を使用）", key="kid_u")
            title_u = st.text_input("タイトル", key="title_u")
            source_u = st.text_input("出所", value=uploaded.name, key="source_u")

            if st.button("登録", key="btn_upload", type="primary"):
                from services.ingestion.document_to_markdown.ingestion_service import IngestionService
                svc = IngestionService(knowledge_manager=km)
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)
                try:
                    unit = svc.ingest(
                        tmp_path,
                        knowledge_id=kid_u or None,
                        title=title_u,
                        source=source_u,
                    )
                    st.success(f"登録しました: `{unit.knowledge_id}`")
                    tmp_path.unlink(missing_ok=True)
                    _set_mode("list")
                    st.rerun()
                except Exception as e:
                    st.error(f"エラー: {e}")
                    tmp_path.unlink(missing_ok=True)

    with tab_manual:
        kid_m = st.text_input("Knowledge ID *", key="kid_m")
        title_m = st.text_input("タイトル", key="title_m")
        source_m = st.text_input("出所", key="source_m")
        text_m = st.text_area("内容 (Markdown) *", height=300, key="text_m")

        if st.button("登録", key="btn_manual", type="primary"):
            if not kid_m.strip() or not text_m.strip():
                st.warning("Knowledge ID と内容は必須です。")
            else:
                try:
                    km.add(kid_m.strip(), text_m, title=title_m, source=source_m)
                    st.success(f"登録しました: `{kid_m}`")
                    _set_mode("list")
                    st.rerun()
                except FileExistsError:
                    st.error(f"ID `{kid_m}` はすでに存在します。")
                except Exception as e:
                    st.error(f"エラー: {e}")

    st.divider()
    if st.button("← キャンセル"):
        _set_mode("list")
        st.rerun()


# ===========================================================================
# EDIT
# ===========================================================================
elif st.session_state.km_mode == "edit":
    kid = st.session_state.km_selected_id
    unit = km.load(kid)
    st.subheader(f"ナレッジを編集: `{kid}`")

    new_title = st.text_input("タイトル", value=unit.title or "")
    new_source = st.text_input("出所", value=unit.source or "")
    new_text = st.text_area("内容 (Markdown)", value=unit.text, height=400)

    st.divider()
    st.subheader("サマリーキャッシュ")
    if unit.summary:
        new_summary = st.text_area("サマリー", value=unit.summary, height=120)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("サマリーを更新"):
                km.save_summary(kid, new_summary)
                st.success("更新しました。")
        with c2:
            if st.button("サマリーを削除", type="secondary"):
                km.delete_summary(kid)
                st.success("削除しました。")
                st.rerun()
    else:
        new_summary_add = st.text_area("サマリー（新規）", height=100)
        if new_summary_add.strip() and st.button("サマリーを保存"):
            km.save_summary(kid, new_summary_add)
            st.success("保存しました。")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("保存", type="primary", use_container_width=True):
            try:
                km.add(kid, new_text, title=new_title, source=new_source, overwrite=True)
                st.success("保存しました。")
                _set_mode("list")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
    with c2:
        if st.button("← キャンセル", use_container_width=True):
            _set_mode("list")
            st.rerun()


# ===========================================================================
# DELETE
# ===========================================================================
elif st.session_state.km_mode == "delete":
    kid = st.session_state.km_selected_id
    st.subheader(f"ナレッジを削除: `{kid}`")
    st.warning(f"`{kid}` を削除します。この操作は元に戻せません。")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("削除する", type="primary", use_container_width=True):
            km.delete(kid)
            st.success(f"`{kid}` を削除しました。")
            _set_mode("list")
            st.rerun()
    with c2:
        if st.button("← キャンセル", use_container_width=True):
            _set_mode("list")
            st.rerun()
