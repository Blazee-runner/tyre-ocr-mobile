import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests

# =========================
# Page Config
# =========================
st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("Mobile Camera OCR")

# =========================
# Scanner UI CSS (GUIDE ONLY)
# =========================
st.markdown("""
<style>
.scanner-wrapper {
    position: relative;
    width: 100%;
    max-width: 420px;
    margin: auto;
}

.scanner-box {
    position: absolute;
    top: 10%;
    left: 50%;
    transform: translateX(-50%);
    width: 22%;
    height: 75%;
    border: 3px solid #00ff66;
    border-radius: 10px;
    box-shadow: 0 0 12px rgba(0,255,102,0.7);
    pointer-events: none;
}

.scanner-text {
    text-align: center;
    color: #00ff66;
    font-weight: 600;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SAFE IMAGE DISPLAY (CLOUD)
# =========================
def show_image_bytes(pil_img, caption=None):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf, caption=caption, width=420)

# =========================
# Backend URL
# =========================
OCR_API_URL = st.secrets.get("OCR_API_URL") or os.getenv("OCR_API_URL", "").strip()

with st.sidebar:
    st.header("Backend")
    st.code(OCR_API_URL or "NOT SET")

if not OCR_API_URL:
    st.error("OCR_API_URL is not set")
    st.stop()

# =========================
# Camera Input
# =========================
st.subheader("Scan Barcode / Text")

st.markdown('<div class="scanner-wrapper">', unsafe_allow_html=True)

cam = st.camera_input(
    "Align barcode inside the vertical strip",
    label_visibility="collapsed"
)

st.markdown('<div class="scanner-box"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="scanner-text">Align text inside the vertical strip</div>',
    unsafe_allow_html=True
)

# =========================
# Upload fallback
# =========================
st.subheader("Or Upload Image")
uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
)

if not cam and not uploaded:
    st.info("Capture or upload an image to continue.")
    st.stop()

# =========================
# Load Image
# =========================
src = cam if cam else uploaded
img = Image.open(src)

# Fix EXIF if present
try:
    img = ImageOps.exif_transpose(img)
except Exception:
    pass

img = img.convert("RGB")

# =========================
# FORCE PORTRAIT ORIENTATION
# =========================
def normalize_orientation(pil_img):
    w, h = pil_img.size
    if w > h:
        pil_img = pil_img.rotate(90, expand=True)
    return pil_img

img = normalize_orientation(img)

# =========================
# CROP + OVERLAY FUNCTIONS
# =========================
def crop_vertical_strip(pil_img, strip_ratio=0.22):
    w, h = pil_img.size
    strip_w = int(w * strip_ratio)
    x1 = (w - strip_w) // 2
    x2 = x1 + strip_w
    return pil_img.crop((x1, 0, x2, h))

def draw_crop_overlay(pil_img, strip_ratio=0.22):
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    w, h = pil_img.size
    strip_w = int(w * strip_ratio)
    x1 = (w - strip_w) // 2
    x2 = x1 + strip_w

    # Darken outside area
    draw.rectangle((0, 0, x1, h), fill=(0, 0, 0, 140))
    draw.rectangle((x2, 0, w, h), fill=(0, 0, 0, 140))

    # Green vertical guide
    draw.rectangle((x1, 0, x2, h), outline=(0, 255, 0, 255), width=4)

    return overlay

# =========================
# SHOW OVERLAY + CROPPED ROI
# =========================
st.subheader("Scan Guide (Green Area Will Be Cut)")
overlay_img = draw_crop_overlay(img)
show_image_bytes(overlay_img, "Align text inside green area")

cropped_img = crop_vertical_strip(img)

st.subheader("Actual Cropped ROI (Sent to OCR)")
show_image_bytes(cropped_img, "Cropped vertical strip")

# Use cropped image for OCR
img = cropped_img

# =========================
# OCR API Call
# =========================
def call_ocr_api(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)

    if r.status_code != 200:
        raise RuntimeError(f"OCR Error {r.status_code}: {r.text[:200]}")

    return r.json()

# =========================
# Run OCR
# =========================
run_ocr = st.button("Run OCR")

if run_ocr:
    try:
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        api_out = call_ocr_api(img)

        detections = api_out.get("detections", [])
        status = api_out.get("status")
        message = api_out.get("message")

        st.info(f"OCR status: {status} — {message}")

        rows = []
        detected_texts = []

        for i, det in enumerate(detections, start=1):
            box = det["box"]
            text = det["text"]
            score = float(det["score"])

            pts = [(int(p[0]), int(p[1])) for p in box]
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            draw.text(
                (pts[0][0], max(0, pts[0][1] - 12)),
                text,
                fill=(255, 255, 0),
                font=font
            )

            detected_texts.append(text)

            rows.append({
                "id": i,
                "text": text,
                "score": score
            })

        show_image_bytes(annotated, "OCR Result")

        if detected_texts:
            st.subheader("Detected Text")
            st.code(" | ".join(detected_texts))

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Detections Table")
            st.dataframe(df, use_container_width=True)

            csv = BytesIO()
            df.to_csv(csv, index=False)

            st.download_button(
                "Download CSV",
                csv.getvalue(),
                "ocr_results.csv",
                "text/csv"
            )

        st.success(f"OCR complete — {len(rows)} detections")

    except Exception as e:
        st.error(f"OCR failed: {e}")
