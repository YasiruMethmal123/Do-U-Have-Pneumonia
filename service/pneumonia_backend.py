# pneumonia_backend.py
# ─────────────────────────────────────────────────────────────────────────────
# Backend: model loading, inference, report generation, templates
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile
from PIL import Image
import streamlit as st


# ── Report Templates ──────────────────────────────────────────────────────────

REPORT_TEMPLATES = {
    "NORMAL": [
        (
            "The chest X-ray analysis indicates a NORMAL finding. "
            "No significant radiological abnormalities were detected in the lung fields. "
            "Lung parenchyma appears clear with no evidence of consolidation, infiltrates, "
            "or opacification. The cardiac silhouette and costophrenic angles appear within "
            "normal limits. No immediate clinical intervention appears necessary based on "
            "this imaging result alone."
        ),
        (
            "Radiological assessment result: NORMAL. "
            "The AI classification model detected no signs consistent with pneumonia in this "
            "chest X-ray. Lung fields appear clear and well-aerated. Both hemithoraces show "
            "no abnormal densities. Recommend routine follow-up as clinically indicated. "
            "Correlation with patient history and physical examination remains essential."
        ),
        (
            "Imaging review: No acute cardiopulmonary process identified. "
            "The submitted chest X-ray was classified as NORMAL by the model. "
            "There is no radiographic evidence of airspace disease, pleural effusion, "
            "or pneumothorax detected. Clinical correlation is always advised before "
            "any medical decision is made."
        ),
    ],
    "PNEUMONIA": [
        (
            "The chest X-ray analysis flags a PNEUMONIA finding. "
            "Radiological features suggest pulmonary consolidation or infiltration consistent "
            "with pneumonic changes. Patterns such as increased opacity, air bronchograms, "
            "or lobar consolidation may be present. Immediate clinical evaluation is strongly "
            "recommended. Correlation with patient symptoms, lab results, and physician "
            "assessment is essential before any diagnosis is confirmed."
        ),
        (
            "Radiological assessment result: PNEUMONIA DETECTED. "
            "The AI model identified imaging patterns consistent with pneumonic changes in "
            "the lung fields. This may indicate bacterial, viral, or atypical pneumonia. "
            "This result requires urgent clinical review by a licensed radiologist or physician. "
            "Do not use this output as a standalone diagnosis under any circumstances."
        ),
        (
            "Imaging review: Findings suggestive of pulmonary infection. "
            "The submitted chest X-ray was classified as PNEUMONIA by the model. "
            "Areas of increased radiodensity or consolidation may be present, which can be "
            "consistent with an active respiratory infection. Prompt medical attention and "
            "further diagnostic workup are advised. Always verify AI findings with a "
            "qualified healthcare professional."
        ),
    ],
}

# Disclaimer appended to every report
DISCLAIMER = (
    "\n\nDisclaimer: This AI-generated summary is intended for research and educational "
    "purposes only. It does not constitute a medical diagnosis. Always consult a licensed "
    "radiologist or physician for clinical decisions."
)


# ── Model Loaders (Streamlit cached) ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)   # <-- best.pt
    model.to(device)

    return model


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the Sentence Transformer encoder."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(image: Image.Image, yolo_model) -> dict:
    """Run inference using the already-loaded YOLO model object."""
    import torch

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Call predict directly on the loaded object (not re-loading from path)
        results     = yolo_model.predict(source=tmp_path, imgsz=224, verbose=False)
        result      = results[0]
        top1_idx    = result.probs.top1
        label       = result.names[top1_idx]
        confidence  = float(result.probs.top1conf)
        all_probs   = result.probs.data.tolist()
        class_names = list(result.names.values())
    finally:
        os.unlink(tmp_path)

    return {
        "label":       label,
        "confidence":  confidence,
        "class_names": class_names,
        "all_probs":   all_probs,
    }

# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(label: str, confidence: float, st_model) -> str:
    """
    Use Sentence Transformer cosine similarity to select the most
    semantically relevant clinical report template for the prediction.

    Args:
        label:      Predicted class ('NORMAL' or 'PNEUMONIA')
        confidence: Prediction confidence (0.0 – 1.0)
        st_model:   Loaded SentenceTransformer model

    Returns:
        Selected report string with disclaimer appended.
    """
    from sentence_transformers import util

    query      = f"chest xray {label.lower()} diagnosis confidence {confidence:.0%} pulmonary finding"
    candidates = REPORT_TEMPLATES.get(label, REPORT_TEMPLATES["NORMAL"])

    query_embedding     = st_model.encode(query,      convert_to_tensor=True)
    candidate_embeddings = st_model.encode(candidates, convert_to_tensor=True)

    scores   = util.cos_sim(query_embedding, candidate_embeddings)[0]
    best_idx = int(scores.argmax())

    return candidates[best_idx] + DISCLAIMER


# ── Image Metadata ─────────────────────────────────────────────────────────────

def get_image_metadata(uploaded_file, image: Image.Image) -> dict:
    """Extract display metadata from an uploaded file."""
    return {
        "filename":  uploaded_file.name,
        "width":     image.size[0],
        "height":    image.size[1],
        "mode":      image.mode,
        "size_kb":   uploaded_file.size / 1024,
    }