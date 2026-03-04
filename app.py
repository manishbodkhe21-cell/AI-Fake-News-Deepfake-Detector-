# app.py — AI Fake News + Deepfake/AI Image Detector (OpenRouter)
import os
import json
import base64
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

import numpy as np
import cv2
from scipy.stats import kurtosis


# ---------------------------
# Setup
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY missing in .env file")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

st.set_page_config(page_title="AI Fake News & Deepfake Detector", layout="wide")
st.title("🛡️ Detector")
st.caption("Probability-based assessment + verification steps (not absolute proof).")


# ---------------------------
# Sidebar
# ---------------------------
MODEL_TEXT = st.sidebar.selectbox(
    "Text Model",
    [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-7b-instruct",
        "google/gemma-2-9b-it",
    ],
    index=0,
)

MODEL_VISION = st.sidebar.selectbox(
    "Vision Model (Images)",
    ["openai/gpt-4o-mini"],
    index=0,
)

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2, 0.05)


# ---------------------------
# LLM Helpers
# ---------------------------
def call_llm(model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a misinformation analyst. Be careful and conservative."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def pil_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def call_vision(model: str, prompt: str, img: Image.Image) -> str:
    data_uri = pil_to_data_uri(img)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a forensic image analyst. Be grounded; avoid certainty."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content


def render_json_or_raw(out: str) -> None:
    try:
        data = json.loads(out)

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Label", data.get("label", ""))
        with c2:
            conf = int(data.get("confidence", 0) or 0)
            st.metric("Confidence", f"{conf}/100")
            st.progress(max(0, min(conf, 100)) / 100)
        with c3:
            st.write("**Summary**")
            st.write(data.get("summary", ""))

        st.write("### Reasons")
        reasons = data.get("reasons", [])
        if isinstance(reasons, list):
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write(reasons)

        st.write("### How to Verify")
        steps = data.get("verification_steps", [])
        if isinstance(steps, list):
            for s in steps:
                st.write(f"- {s}")
        else:
            st.write(steps)

    except Exception:
        st.code(out)


# ---------------------------
# Forensic Signals (Computer Vision)
# ---------------------------
def compute_forensic_signals(pil_img: Image.Image) -> dict:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1) Noise residual
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_residual = gray.astype(np.float32) - blur.astype(np.float32)
    noise_score = float(np.mean(np.abs(noise_residual)))

    # 2) Edge density
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    # 3) FFT high-frequency ratio
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = max(8, min(h, w) // 6)

    low = magnitude[cy - r:cy + r, cx - r:cx + r]
    high = magnitude.copy()
    high[cy - r:cy + r, cx - r:cx + r] = 0

    hf_ratio = float(high.mean() / (low.mean() + 1e-6))

    # 4) Blockiness proxy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    blockiness = float(
        np.mean(np.abs(np.diff(grad_mag, axis=0))) + np.mean(np.abs(np.diff(grad_mag, axis=1)))
    )

    # 5) Kurtosis of residual
    k = float(kurtosis(noise_residual.flatten(), fisher=False))

    # Heuristic risk scoring (0-100)
    score = 0
    score += 25 if hf_ratio > 1.25 else 0
    score += 20 if noise_score < 6 else 0
    score += 20 if k > 6 else 0
    score += 15 if edge_density < 0.03 else 0
    score += 20 if blockiness > 120 else 0
    score = int(max(0, min(score, 100)))

    # Heatmap (noise residual)
    heat = cv2.normalize(np.abs(noise_residual), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR

    return {
        "noise_score": noise_score,
        "edge_density": edge_density,
        "hf_ratio": hf_ratio,
        "blockiness": blockiness,
        "noise_kurtosis": k,
        "risk_score": score,
        "heatmap_bgr": heat_color,
    }


# ---------------------------
# UI
# ---------------------------
tab1, tab2 = st.tabs(["📰 Fake News (Text)", "🖼️ Deepfake / AI Image"])

with tab1:
    st.subheader("Fake / Misleading News Detector (Text)")
    text = st.text_area(
        "Paste headline / article text",
        height=220,
        placeholder="Paste suspicious news here...",
    )

    if st.button("Analyze Text", use_container_width=True):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            prompt = f"""
Analyze this news text for misinformation / manipulation.
Return ONLY valid JSON with keys:
- label: "Likely Real" or "Likely Fake/Misleading"
- confidence: 0-100
- summary: 1-2 lines
- reasons: 3-6 bullet strings
- verification_steps: 3-6 bullet strings

Text:
{text}
"""
            with st.spinner("Analyzing text..."):
                out = call_llm(MODEL_TEXT, prompt)

            render_json_or_raw(out)

with tab2:
    st.subheader("Deepfake / AI Image Detector (Signals + Explainability)")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

        signals = compute_forensic_signals(img)

        st.write("### Forensic Signals (auto)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Noise", f"{signals['noise_score']:.2f}")
        c2.metric("Edge dens.", f"{signals['edge_density']:.3f}")
        c3.metric("HF ratio", f"{signals['hf_ratio']:.2f}")
        c4.metric("Blockiness", f"{signals['blockiness']:.1f}")
        c5.metric("Risk score", f"{signals['risk_score']}/100")

        heat_rgb = cv2.cvtColor(signals["heatmap_bgr"], cv2.COLOR_BGR2RGB)
        st.image(heat_rgb, caption="Forensic heatmap (noise residual)", use_container_width=True)

        if st.button("Finalize Verdict (AI + Signals)", use_container_width=True):
            prompt = f"""
You are a forensic image analyst.

Signals:
{{
  "noise_score": {signals['noise_score']:.2f},
  "edge_density": {signals['edge_density']:.3f},
  "hf_ratio": {signals['hf_ratio']:.2f},
  "blockiness": {signals['blockiness']:.1f},
  "noise_kurtosis": {signals['noise_kurtosis']:.2f},
  "risk_score": {signals['risk_score']}
}}

Decision rules:
- If risk_score >= 60 => label "Likely AI/Manipulated"
- If risk_score <= 35 => label "Likely Authentic"
- Otherwise => label "Uncertain / Needs Verification"

Return ONLY valid JSON with:
- label
- confidence (0-100)  (use risk_score for AI; use 100-risk_score for authentic; uncertain ~50-65)
- summary (1-2 lines)
- reasons (3-6 bullet strings)
- verification_steps (3-6 bullet strings)

Be conservative. Do not claim certainty.
"""
            with st.spinner("Finalizing verdict..."):
                out = call_vision(MODEL_VISION, prompt, img)

            render_json_or_raw(out)

st.caption("Note: These results are probabilistic signals, not definitive proof.")
