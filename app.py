"""
Streamlit demo app — upload a chest X-ray, get a diagnosis + Grad-CAM heatmap.

Run:
    streamlit run app.py
"""
import os
import torch
import numpy as np
from PIL import Image
import streamlit as st

import config
from src.model import load_model
from src.gradcam import generate_gradcam, side_by_side

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray AI Diagnostic Tool",
    page_icon="🫁",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title  { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
  .sub-title   { color: #5F5E5A; font-size: 1rem; margin-bottom: 2rem; }
  .result-box  { padding: 1.2rem 1.5rem; border-radius: 12px;
                 border: 1px solid; margin-top: 1rem; }
  .normal      { background: #EAF3DE; border-color: #3B6D11; color: #27500A; }
  .pneumonia   { background: #FCEBEB; border-color: #A32D2D; color: #791F1F; }
  .disclaimer  { font-size: 0.78rem; color: #888780; margin-top: 2rem;
                 border-top: 1px solid #D3D1C7; padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached so it only loads once) ───────────────────────────────
@st.cache_resource
def load_cached_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(config.MODEL_PATH):
        return None, device
    model = load_model(config.MODEL_PATH, device)
    return model, device


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About this model")
    st.markdown("""
**Architecture:** EfficientNet-B0 (transfer learning from ImageNet)

**Dataset:** [Kaggle Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— 5,863 JPEG images, 2 classes

**Explainability:** Grad-CAM highlights the lung regions that drove the prediction

**Typical performance:**
| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| AUC | ~0.98 |
| F1 (macro) | ~0.94 |

**Training:** Two-phase fine-tuning with class-weighted cross-entropy loss
    """)
    st.markdown("---")
    st.markdown("Built for the **Chest X-Ray AI + Explainability** portfolio project.")


# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🫁 Chest X-Ray AI Diagnostic Tool</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a chest X-ray → the model predicts Normal vs Pneumonia '
    'and shows <b>exactly where it looked</b> using Grad-CAM explainability.</p>',
    unsafe_allow_html=True,
)

model, device = load_cached_model()

if model is None:
    st.warning(
        "No trained model found at `outputs/best_model.pth`. "
        "Please run `python -m src.train` first, then relaunch this app."
    )
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_column_width=True)

    with st.spinner("Analysing X-ray…"):
        overlay, pred_class, confidence = generate_gradcam(model, image, device)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, use_column_width=True,
                 caption="Red = high attention | Blue = low attention")

    # Prediction result
    label = config.CLASS_NAMES[pred_class]
    css_class = "pneumonia" if pred_class == 1 else "normal"
    emoji = "🔴" if pred_class == 1 else "🟢"

    st.markdown(
        f'<div class="result-box {css_class}">'
        f'<b style="font-size:1.2rem">{emoji} Prediction: {label}</b><br>'
        f'Confidence: <b>{confidence * 100:.1f}%</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.markdown("#### Confidence breakdown")
    cols = st.columns(2)
    with cols[0]:
        prob_normal = (1 - confidence) if pred_class == 1 else confidence
        st.metric("NORMAL", f"{(1-confidence if pred_class==1 else confidence)*100:.1f}%")
    with cols[1]:
        prob_pneumonia = confidence if pred_class == 1 else 1 - confidence
        st.metric("PNEUMONIA", f"{(confidence if pred_class==1 else 1-confidence)*100:.1f}%")

    # Download side-by-side figure
    sbs = side_by_side(image, overlay)
    sbs_pil = Image.fromarray(sbs)
    import io
    buf = io.BytesIO()
    sbs_pil.save(buf, format="PNG")
    st.download_button(
        label="Download original + heatmap (PNG)",
        data=buf.getvalue(),
        file_name="xray_gradcam.png",
        mime="image/png",
    )

    # Interpretation guide
    with st.expander("How to interpret the Grad-CAM heatmap"):
        st.markdown("""
**What it shows:**
- The heatmap highlights *which pixels* had the greatest influence on the model's decision.
- **Warm colours (red/orange):** the model paid the most attention here.
- **Cool colours (blue/purple):** these regions had little influence.

**For pneumonia:**
The model should focus on the lower lung lobes, where consolidation, infiltrates,
and fluid accumulation are typically found.

**For normal X-rays:**
Attention should be more diffuse across both lung fields, with no concentrated hotspot.

**Caveats:**
- This tool is a research prototype — not a clinical diagnostic device.
- Grad-CAM can produce false confidence on adversarial or low-quality images.
- Always consult a qualified radiologist for medical decisions.
        """)

st.markdown(
    '<p class="disclaimer">⚕️ This application is for educational and research purposes only. '
    'It is not approved for clinical use and should not replace professional medical advice, '
    'diagnosis, or treatment.</p>',
    unsafe_allow_html=True,
)
