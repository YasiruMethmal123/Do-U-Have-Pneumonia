# pneumonia_ui.py
# ─────────────────────────────────────────────────────────────────────────────
# UI: Streamlit interface — imports all logic from pneumonia_backend.py
# Run with: streamlit run pneumonia_ui.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st
from PIL import Image

from service.pneumonia_backend import (
    load_yolo_model,
    load_sentence_transformer,
    run_inference,
    generate_report,
    get_image_metadata,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# DEFAULT_MODEL_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "model", "model.pkl"
# )

DEFAULT_MODEL_PATH = "model/best.pt"
ST_MODEL_NAME        = "all-MiniLM-L6-v2"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background: #0a0a0f;
        color: #e8e8f0;
        font-family: 'DM Mono', monospace;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 20% 20%, #0f1a2e 0%, #0a0a0f 50%, #0f0a1e 100%);
        min-height: 100vh;
    }
    [data-testid="stHeader"] { background: transparent; }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #7eb8f7 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #4a4a6a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2.5rem;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a2a4a, transparent);
        margin: 2rem 0;
    }
    .upload-zone {
        border: 1px dashed #2a2a5a;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1.5rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-normal   { background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,78,59,0.15)); border: 1px solid rgba(16,185,129,0.3); }
    .result-pneumonia{ background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(127,29,29,0.15)); border: 1px solid rgba(239,68,68,0.3); }
    .result-label    { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .label-normal    { color: #34d399; }
    .label-pneumonia { color: #f87171; }
    .confidence-bar-bg { height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; margin: 1rem 0; overflow: hidden; }
    .bar-normal   { height:100%; background: linear-gradient(90deg,#10b981,#34d399); border-radius:99px; }
    .bar-pneumonia{ height:100%; background: linear-gradient(90deg,#ef4444,#f87171); border-radius:99px; }
    .tag { display:inline-block; font-size:0.65rem; letter-spacing:0.1em; padding:0.2rem 0.6rem; border-radius:99px; text-transform:uppercase; font-weight:500; }
    .tag-normal    { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
    .tag-pneumonia { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(248,113,113,0.3); }
    .report-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid #1e1e3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        font-size: 0.88rem;
        line-height: 1.8;
        color: #c8c8e0;
    }
    .report-header {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4a4a6a;
        margin-bottom: 1rem;
    }
    .meta-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #3a3a5a;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #1a1a2e;
    }
    .stButton > button {
        background: linear-gradient(135deg,#7eb8f7,#a78bfa) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stImage"] img { border-radius: 12px; border: 1px solid #1e1e3a; }
    </style>
    """, unsafe_allow_html=True)


# ── Component Renderers ────────────────────────────────────────────────────────

def render_header():
    st.markdown('<div class="main-title">PneumoScan AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">⬡ Chest X-Ray Pneumonia Classifier · YOLOv8 + Sentence Transformer</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("###  Configuration")
        model_path = st.text_input(
            "Model path (best.pt)",
            value=DEFAULT_MODEL_PATH,
            help="Absolute or relative path to your trained YOLOv8 best.pt file",
        )
        st.caption("Default: `best.pt` in the same folder as this script.")
        st.markdown("---")
        st.markdown("**YOLO Model:** YOLOv8m-cls")
        st.markdown(f"**Encoder:** `{ST_MODEL_NAME}`")
        st.markdown("**Classes:** NORMAL · PNEUMONIA")
    return model_path


def render_empty_state():
    st.markdown("""
    <div class="upload-zone">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#6a6a8a; margin-bottom:0.5rem">
            Drop your chest X-ray here
        </div>
        <div style="font-size:0.75rem; color:#3a3a5a; letter-spacing:0.1em; text-transform:uppercase">
            JPG · JPEG · PNG supported
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_image_preview(image: Image.Image, meta: dict):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    with col2:
        st.markdown("##### Image Info")
        st.markdown(f"""
        - **File:** `{meta['filename']}`
        - **Size:** `{meta['width']} × {meta['height']} px`
        - **Mode:** `{meta['mode']}`
        - **File size:** `{meta['size_kb']:.1f} KB`
        """)


def render_result_card(label: str, confidence: float):
    card_class  = "result-normal"   if label == "NORMAL" else "result-pneumonia"
    label_class = "label-normal"    if label == "NORMAL" else "label-pneumonia"
    tag_class   = "tag-normal"      if label == "NORMAL" else "tag-pneumonia"
    bar_class   = "bar-normal"      if label == "NORMAL" else "bar-pneumonia"
    icon        = "✅"              if label == "NORMAL" else "⚠️"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
            <span style="font-size:1.8rem">{icon}</span>
            <span class="result-label {label_class}">{label}</span>
            <span class="tag {tag_class}">{confidence:.1%} confidence</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="{bar_class}" style="width:{confidence*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_class_probabilities(class_names: list, all_probs: list):
    st.markdown("**Class Probabilities**")
    for cname, prob in zip(class_names, all_probs):
        st.progress(prob, text=f"{cname}: {prob:.1%}")


def render_report(report_text: str, st_model_name: str):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="report-box">
        <div class="report-header">AI Clinical Summary</div>
        {report_text.replace(chr(10), '<br>')}
        <div class="meta-row">
            <span>Model · YOLOv8m-cls</span>
            <span>Encoder · {st_model_name}</span>
            <span>⚠ Not a medical diagnosis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    inject_css()
    render_header()
    model_path = render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        render_empty_state()
        return

    image = Image.open(uploaded_file).convert("RGB")
    meta  = get_image_metadata(uploaded_file, image)
    render_image_preview(image, meta)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not st.button("🔬 Analyse X-Ray"):
        return

    # ── Load models ────────────────────────────────────────────────────────────
    with st.spinner("Loading YOLO model..."):
        try:
            yolo_model = load_yolo_model(model_path)
        except FileNotFoundError as e:
            st.error(f"{e}\n\nUpdate the model path in the sidebar.")
            return

    with st.spinner("Loading Sentence Transformer..."):
        st_model = load_sentence_transformer(ST_MODEL_NAME)

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Analysing X-Ray..."):
        prediction = run_inference(image, yolo_model)
        time.sleep(0.2)

    label       = prediction["label"]
    confidence  = prediction["confidence"]
    class_names = prediction["class_names"]
    all_probs   = prediction["all_probs"]

    render_result_card(label, confidence)
    render_class_probabilities(class_names, all_probs)

    # ── Report ─────────────────────────────────────────────────────────────────
    with st.spinner("Generating clinical report..."):
        report = generate_report(label, confidence, st_model)
        time.sleep(0.3)

    render_report(report, ST_MODEL_NAME)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(
        "**Disclaimer:** This tool is for research and educational purposes only. "
        "Always consult a licensed radiologist or physician for medical decisions."
    )


if __name__ == "__main__":
    main()