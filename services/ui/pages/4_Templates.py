"""
Templates page — view, edit, and manage prompt templates.
"""

import streamlit as st

from services.inference.llm_inference_service.prompt_template_manager import PromptTemplateManager

st.set_page_config(page_title="Prompt Templates", layout="wide")
st.title("Prompt Templates")
st.caption("プロンプトテンプレートを管理します。{knowledge} {log} {question} プレースホルダーが使えます。")

tm = PromptTemplateManager()

# --- session state ---
if "tpl_mode" not in st.session_state:
    st.session_state.tpl_mode = "list"
if "tpl_selected" not in st.session_state:
    st.session_state.tpl_selected = None


def _set_mode(mode: str, name: str | None = None) -> None:
    st.session_state.tpl_mode = mode
    st.session_state.tpl_selected = name


# ===========================================================================
# LIST / EDIT
# ===========================================================================
if st.session_state.tpl_mode in ("list", "edit"):
    names = tm.list_names()

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("テンプレート一覧")
        if not names:
            st.info("テンプレートがありません。")
        for name in names:
            selected = st.session_state.tpl_selected == name
            label = f"**{name}**" if selected else name
            if st.button(label, key=f"sel_{name}", use_container_width=True):
                _set_mode("edit", name)
                st.rerun()

        st.divider()
        if st.button("+ 新規テンプレート", use_container_width=True, type="primary"):
            _set_mode("new")
            st.rerun()

    with col_right:
        if st.session_state.tpl_selected:
            name = st.session_state.tpl_selected
            st.subheader(f"編集: `{name}`")
            try:
                content = tm.load(name)
            except FileNotFoundError:
                st.error(f"テンプレート `{name}` が見つかりません。")
                content = ""

            new_content = st.text_area(
                "テンプレート内容",
                value=content,
                height=400,
                key=f"tpl_edit_{name}",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存", type="primary", use_container_width=True):
                    try:
                        tm.save(name, new_content, overwrite=True)
                        st.success(f"`{name}` を保存しました。")
                    except Exception as e:
                        st.error(f"エラー: {e}")
            with c2:
                if name != "default":
                    if st.button("削除", type="secondary", use_container_width=True):
                        _set_mode("delete", name)
                        st.rerun()
                else:
                    st.caption("`default` は削除できません。")
        else:
            st.info("左のリストからテンプレートを選択してください。")


# ===========================================================================
# NEW
# ===========================================================================
elif st.session_state.tpl_mode == "new":
    st.subheader("新規テンプレートを作成")

    new_name = st.text_input("テンプレート名 *")
    new_content = st.text_area(
        "テンプレート内容",
        value=(
            "以下のナレッジとログを参照して質問に答えてください。\n\n"
            "## ナレッジ\n{knowledge}\n\n"
            "## ログ\n{log}\n\n"
            "## 質問\n{question}\n\n"
            "回答は JSON 形式で返してください:\n"
            '{{"answer": "yes/no", "confidence": 0.0~1.0, "reason": "根拠"}}'
        ),
        height=400,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("作成", type="primary", use_container_width=True):
            if not new_name.strip():
                st.warning("テンプレート名は必須です。")
            else:
                try:
                    tm.save(new_name.strip(), new_content, overwrite=False)
                    st.success(f"`{new_name}` を作成しました。")
                    _set_mode("edit", new_name.strip())
                    st.rerun()
                except FileExistsError:
                    st.error(f"テンプレート `{new_name}` はすでに存在します。")
                except Exception as e:
                    st.error(f"エラー: {e}")
    with c2:
        if st.button("← キャンセル", use_container_width=True):
            _set_mode("list")
            st.rerun()


# ===========================================================================
# DELETE
# ===========================================================================
elif st.session_state.tpl_mode == "delete":
    name = st.session_state.tpl_selected
    st.subheader(f"テンプレートを削除: `{name}`")
    st.warning("この操作は元に戻せません。")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("削除する", type="primary", use_container_width=True):
            try:
                tm.delete(name)
                st.success(f"`{name}` を削除しました。")
                _set_mode("list")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
    with c2:
        if st.button("← キャンセル", use_container_width=True):
            _set_mode("list")
            st.rerun()
