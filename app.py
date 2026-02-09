import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests
import cv2


# ============================================================
# Streamlit compatibility helpers (1.38 vs newer)
# ============================================================
def st_image_auto(img, caption=None):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

def st_df_auto(df):
    try:
        st.dataframe(df, use_container_width=True)
    except TypeError:
        st.dataframe(df)


# ============================================================
# Backend URL from Secrets/ENV
# ============================================================
def normalize_ocr_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    url = url.rstrip("/")
    if not url.endswith("/ocr"):
        url += "/ocr"
    return url

OCR_API_URL = None
if "OCR_API_URL" in st.secrets:
    OCR_API_URL = str(st.secrets["OCR_API_URL"]).strip()
else:
    OCR_API_URL = os.getenv("OCR_API_URL", "").strip()

OCR_API_URL = normalize_ocr_url(OCR_API_URL)

st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("📷 Camera OCR (Scanner-style ROI band)")

with st.sidebar:
    st.header("Scanner ROI")
    band_height_pct = st.slider("Band height (%)", 5, 80, 20, 1)
    band_center_pct = st.slider("Band center Y (%)", 0, 100, 50, 1)

    st.header("Scanner Effects")
    blur_strength = st.slider("Blur strength", 1, 31, 17, 2)
    mask_darkness = st.slider("Mask darkness", 0.0, 0.9, 0.45, 0.05)

    st.header("Display")
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Backend")
    st.code(OCR_API_URL or "NOT SET")

if not OCR_API_URL:
    st.error("OCR_API_URL is not set. Add it in Streamlit secrets or ENV.")
    st.stop()


def call_ocr_api(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Backend HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def compute_band(h: int, band_height_pct: int, band_center_pct: int):
    band_h = int(h * (band_height_pct / 100.0))
    band_h = max(10, min(band_h, h))
    center = int(h * (band_center_pct / 100.0))
    y1 = max(0, center - band_h // 2)
    y2 = min(h, y1 + band_h)
    if y2 - y1 < band_h:
        y1 = max(0, y2 - band_h)
    return y1, y2


def make_scanner_overlay(pil_img: Image.Image, y1: int, y2: int, blur_strength: int, mask_darkness: float):
    """
    Create scanner-look preview:
      - ROI band sharp
      - Outside ROI blurred + dark translucent mask
      - Two thin black horizontal lines
    Returns: overlay PIL (RGB)
    """
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    k = int(blur_strength)
    if k % 2 == 0:
        k += 1
    k = max(1, min(k, 51))

    blurred = cv2.GaussianBlur(bgr, (k, k), 0)
    out = blurred.copy()
    out[y1:y2] = bgr[y1:y2]  # keep band sharp

    # dark mask outside
    alpha = float(mask_darkness)
    if alpha > 0:
        dark = np.zeros_like(out)
        out[:y1] = cv2.addWeighted(out[:y1], 1 - alpha, dark[:y1], alpha, 0)
        out[y2:] = cv2.addWeighted(out[y2:], 1 - alpha, dark[y2:], alpha, 0)

    # thin black lines
    cv2.line(out, (0, y1), (w, y1), (0, 0, 0), 1)
    cv2.line(out, (0, y2), (w, y2), (0, 0, 0), 1)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


# ============================================================
# Camera capture
# ============================================================
st.subheader("Capture")
cam = st.camera_input("Take a photo")

# Optional upload fallback
uploaded = st.file_uploader("Or upload", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

if not cam and not uploaded:
    st.info("Take a photo (or upload) to continue.")
    st.stop()

src = cam if cam else uploaded
img = Image.open(src)

if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

img = img.convert("RGB")
W, H = img.size
st.caption(f"Captured image: **{W} × {H}px**")

# Compute band in this captured image
y1, y2 = compute_band(H, band_height_pct, band_center_pct)

# Scanner-style preview
preview = make_scanner_overlay(img, y1, y2, blur_strength, mask_darkness)

# ROI band to send
roi_pil = img.crop((0, y1, W, y2))

c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader("Scanner preview (after capture)")
    st_image_auto(preview)

with c2:
    st.subheader("ROI band to OCR")
    st_image_auto(roi_pil)
    st.caption(f"Band Y: {y1} → {y2} (height={y2-y1}px)")


run_ocr = st.button("Run OCR (send ROI band)")

if run_ocr:
    try:
        api_out = call_ocr_api(roi_pil)
        status = api_out.get("status", "unknown")
        message = api_out.get("message", "")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        # Annotate ROI only
        roi_annot = roi_pil.copy()
        draw = ImageDraw.Draw(roi_annot)
        font = ImageFont.load_default()

        rows = []
        texts = []

        for idx, det in enumerate(detections, start=1):
            box = det.get("box")
            text = det.get("text", "")
            score = float(det.get("score", 0.0))
            if not box or len(box) < 4:
                continue

            pts = [(int(p[0]), int(p[1])) for p in box]
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
            tx, ty = pts[0]
            draw.text((tx, max(0, ty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            texts.append(text)
            rows.append({"detection_id": idx, "text": text, "score": score})

        st.subheader("ROI band sent to OCR (annotated)")
        st_image_auto(roi_annot)

        if texts:
            st.subheader("Detected Text")
            st.code(" | ".join(texts))

        st.success(f"OCR complete — {len(rows)} detections")

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Detections Table")
            st_df_auto(df)

            csv_buf = BytesIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download OCR CSV",
                data=csv_buf.getvalue(),
                file_name="ocr_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"OCR failed: {e}")
