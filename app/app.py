"""Acno demo app: scan your face, understand your skin.

Run from the repo root (trained weights must be in models/):
    streamlit run app/app.py

Privacy rule: photos are analyzed in memory and never written to disk.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import DISCLAIMER, analyze  # noqa: E402

st.set_page_config(page_title="Acno", page_icon=None, layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { max-width: 720px; padding-top: 3rem; }

h1, h2, h3 { font-family: 'Fraunces', serif; color: #2f3e36; letter-spacing: -0.01em; }

.acno-hero { text-align: center; margin-bottom: 0.5rem; }
.acno-hero h1 { font-size: 3rem; margin-bottom: 0.2rem; }
.acno-hero p { color: #5f7268; font-size: 1.1rem; margin-top: 0; }

.acno-card {
  background: #ffffff;
  border: 1px solid #e7e1d6;
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(47, 62, 54, 0.05);
}
.acno-card h3 { margin-top: 0; font-size: 1.15rem; }
.acno-card p { color: #44544b; line-height: 1.55; margin-bottom: 0.4rem; }
.acno-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8a9a90; margin-bottom: 0.2rem; }
.acno-value { font-family: 'Fraunces', serif; font-size: 1.7rem; color: #2f3e36; }
.acno-sub { color: #75857b; font-size: 0.9rem; }

.acno-note {
  background: #f1ede5;
  border-radius: 14px;
  padding: 1rem 1.3rem;
  color: #5a5140;
  font-size: 0.92rem;
  line-height: 1.5;
}
.acno-derm {
  background: #fdf1ec;
  border: 1px solid #f0d9cd;
  border-radius: 14px;
  padding: 1rem 1.3rem;
  color: #7a4a33;
  line-height: 1.5;
}

.stButton > button {
  background: #6d9886; color: white; border: none; border-radius: 999px;
  padding: 0.6rem 2.2rem; font-weight: 600; font-size: 1rem;
}
.stButton > button:hover { background: #5d8574; color: white; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SEVERITY_TEXT = {
    "clear": "Your skin looks clear. Whatever you are doing, it is working.",
    "mild": "A few spots, which is completely normal, especially during puberty.",
    "moderate": "A fair number of active spots. A steady routine can really help here.",
    "severe": "Quite a lot of active spots. This level deserves professional care.",
}


def card(html):
    st.markdown(f'<div class="acno-card">{html}</div>', unsafe_allow_html=True)


st.markdown(
    '<div class="acno-hero"><h1>Acno</h1>'
    "<p>Scan smarter. Skin clearer.</p></div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="acno-note">Take or upload a clear photo of your face in good light. '
    "Acno looks at it right here on the spot and never saves it anywhere. "
    "When you close this page, the photo is gone.</div>",
    unsafe_allow_html=True,
)
st.write("")

tab_upload, tab_camera = st.tabs(["Upload a photo", "Use your camera"])
image_bytes = None
with tab_upload:
    uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    if uploaded is not None:
        image_bytes = uploaded.read()
with tab_camera:
    shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if shot is not None:
        image_bytes = shot.read()


@st.cache_resource(show_spinner=False)
def warm_models():
    from src.pipeline import _load_models
    return _load_models()


if image_bytes is not None:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("That file does not look like a photo we can read. Try a jpg or png.")
        st.stop()

    with st.spinner("Taking a close look at your skin..."):
        warm_models()
        report, annotated = analyze(image)

    st.write("")
    st.markdown("## Your skin report")

    if not report["face_found"]:
        st.markdown(
            '<div class="acno-note">We could not find a face in this photo, so we '
            "analyzed the whole image. For a better reading, try a straight-on photo "
            "of your face in good light.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

    col1, col2 = st.columns(2)
    with col1:
        card(
            '<div class="acno-label">Skin type</div>'
            f'<div class="acno-value">{report["skin_type"].capitalize()}</div>'
            f'<div class="acno-sub">{report["skin_type_confidence"]:.0%} confident</div>'
        )
    with col2:
        card(
            '<div class="acno-label">Severity</div>'
            f'<div class="acno-value">{report["severity"].capitalize()}</div>'
            f'<div class="acno-sub">{report["lesion_count"]} spots found</div>'
        )

    st.markdown(f'<p class="acno-sub" style="margin-top:-0.4rem">'
                f'{SEVERITY_TEXT[report["severity"]]}</p>', unsafe_allow_html=True)

    info = report.get("acne_type_info")
    if info:
        card(
            '<div class="acno-label">Main acne type we see</div>'
            f'<h3>{info["plain_name"]}</h3>'
            f'<p>{info["explanation"]}</p>'
            f'<p><strong>What helps:</strong> {info["care_tip"]}</p>'
        )
    else:
        card(
            '<div class="acno-label">Main acne type we see</div>'
            f'<h3>{report["acne_type"]}</h3>'
        )

    st.markdown("### Where we looked")
    st.image(annotated, caption="Each box is a spot the model found.",
             use_container_width=True)

    routine = report.get("routine")
    if routine:
        st.markdown("### A simple routine for you")
        card(f'<h3>Morning</h3><p>{routine["morning"]}</p>')
        card(f'<h3>Evening</h3><p>{routine["evening"]}</p>')
        card(f'<h3>Things to avoid</h3><p>{routine["avoid"]}</p>')

    if report["see_dermatologist"]:
        st.markdown(
            '<div class="acno-derm"><strong>Please talk to a dermatologist.</strong> '
            "What we detected is the kind of acne that home products alone usually "
            "cannot fix, and a professional can. Seeing one early is the best way to "
            "prevent scarring. You deserve that care.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(f'<div class="acno-note">{DISCLAIMER}</div>', unsafe_allow_html=True)
else:
    st.write("")
    st.markdown(
        '<p class="acno-sub" style="text-align:center">Your report will appear here '
        "after you add a photo.</p>",
        unsafe_allow_html=True,
    )
