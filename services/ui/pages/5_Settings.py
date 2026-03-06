"""
Settings page — active model, active template, and inference parameters.
"""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Settings", layout="wide")
st.title("Settings")
st.caption("モデルと推論パラメータを設定します。変更後はコンテナの再起動は不要ですが、推論ページをリロードしてください。")

# Paths (pages/ is 4 levels deep from project root: services/ui/pages/5_Settings.py)
_ROOT = Path(__file__).resolve().parents[3]
_INFERENCE_JSON = _ROOT / "config" / "inference.json"
_MODELS_JSON = _ROOT / "config" / "models.json"


def _load_inference_cfg() -> dict:
    with open(_INFERENCE_JSON, encoding="utf-8") as f:
        return json.load(f)


def _save_inference_cfg(cfg: dict) -> None:
    with open(_INFERENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _load_models() -> list[dict]:
    with open(_MODELS_JSON, encoding="utf-8") as f:
        return json.load(f).get("models", [])


def _model_downloaded(model_meta: dict) -> bool:
    path = _ROOT / "data" / "models" / model_meta["filename"]
    return path.exists()


# --------------------------------------------------------------------------
# Load current config
# --------------------------------------------------------------------------
try:
    cfg = _load_inference_cfg()
    models = _load_models()
except Exception as e:
    st.error(f"設定ファイルの読み込みに失敗しました: {e}")
    st.stop()

# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------
st.subheader("モデル設定")

model_ids = [m["id"] for m in models]
current_model = cfg.get("active_model", "")

selected_model_id = st.selectbox(
    "アクティブモデル",
    model_ids,
    index=model_ids.index(current_model) if current_model in model_ids else 0,
    key="sel_model",
)

# Show model info and download status
selected_meta = next((m for m in models if m["id"] == selected_model_id), None)
if selected_meta:
    downloaded = _model_downloaded(selected_meta)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(
            f"説明: {selected_meta.get('description', '—')}\n\n"
            f"Context: {selected_meta.get('context_length', '—')} tokens"
        )
    with col2:
        if downloaded:
            st.success("ダウンロード済み")
        else:
            st.warning("未ダウンロード")
            st.code(
                f"docker compose run --rm downloader "
                f"--model-id {selected_model_id} --set-active",
                language="bash",
            )

st.divider()

# --------------------------------------------------------------------------
# Template selection
# --------------------------------------------------------------------------
st.subheader("テンプレート設定")

try:
    from services.inference.llm_inference_service.prompt_template_manager import PromptTemplateManager
    template_names = PromptTemplateManager().list_names()
except Exception:
    template_names = ["default"]

current_template = cfg.get("active_template", "default")
selected_template = st.selectbox(
    "アクティブテンプレート",
    template_names,
    index=template_names.index(current_template) if current_template in template_names else 0,
    key="sel_template",
)

st.divider()

# --------------------------------------------------------------------------
# Inference parameters
# --------------------------------------------------------------------------
st.subheader("推論パラメータ")

col1, col2 = st.columns(2)
with col1:
    n_ensemble = st.number_input(
        "アンサンブル回数 (ensemble.n)",
        min_value=1,
        max_value=20,
        value=cfg.get("ensemble", {}).get("n", 5),
        key="cfg_n",
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        float(cfg.get("generation", {}).get("temperature", 0.7)),
        0.05,
        key="cfg_temp",
    )
with col2:
    max_tokens = st.number_input(
        "Max tokens",
        min_value=64,
        max_value=4096,
        value=cfg.get("generation", {}).get("max_tokens", 512),
        key="cfg_max_tokens",
    )
    top_p = st.slider(
        "Top-p",
        0.0,
        1.0,
        float(cfg.get("generation", {}).get("top_p", 0.95)),
        0.01,
        key="cfg_top_p",
    )

st.divider()

# --------------------------------------------------------------------------
# LLM model parameters
# --------------------------------------------------------------------------
st.subheader("モデルロードパラメータ")

col1, col2 = st.columns(2)
with col1:
    n_ctx = st.number_input(
        "コンテキスト長 (n_ctx)",
        min_value=512,
        max_value=32768,
        value=cfg.get("model", {}).get("n_ctx", 4096),
        step=512,
        key="cfg_n_ctx",
    )
with col2:
    n_gpu_layers = st.number_input(
        "GPU レイヤー数 (n_gpu_layers, 0=CPU のみ)",
        min_value=0,
        max_value=999,
        value=cfg.get("model", {}).get("n_gpu_layers", 0),
        key="cfg_gpu",
    )

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
st.divider()
if st.button("設定を保存", type="primary"):
    new_cfg = {
        "active_model": selected_model_id,
        "active_template": selected_template,
        "model": {
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": cfg.get("model", {}).get("verbose", False),
        },
        "generation": {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        },
        "ensemble": {
            "n": int(n_ensemble),
        },
    }
    try:
        _save_inference_cfg(new_cfg)
        st.success("設定を保存しました。推論ページをリロードすると新しい設定が反映されます。")
    except Exception as e:
        st.error(f"保存エラー: {e}")

# --------------------------------------------------------------------------
# Current config (read-only preview)
# --------------------------------------------------------------------------
with st.expander("現在の inference.json を表示"):
    try:
        st.json(_load_inference_cfg())
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
