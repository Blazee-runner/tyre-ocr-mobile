import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests

# --------------------------
# Backend URL from Secrets/ENV
# --------------------------
OCR_API_URL = None
if "OCR_API_URL" in st.secrets:
    OCR_API_URL = str(st.secrets["OCR_API_URL"]).strip()
else:
    OCR_API_URL = os.getenv("OCR_API_URL", "").strip()

def normalize_ocr_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    url = url.rstrip("/")
    # allow user to set either base or full /ocr
    if not url.endswith("/ocr"):
        url = url + "/ocr"
    return url

OCR_API_URL = normalize_ocr_url(OCR_API_URL)

st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("Mobile Camera OCR")

with st.sidebar:
    st.header("Display")
    max_canvas_width = st.slider("Max canvas width (px)", 600, 2500, 1200)
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Scanner ROI (Vertical Strip)")
    st.caption("Move the strip over the barcode/text area. Only this strip is sent to OCR.")

    st.header("Backend")
    st.write("OCR API URL (effective):")
    st.code(OCR_API_URL or "NOT SET")

if not OCR_API_URL:
    st.error("OCR_API_URL is not set. Add it in Streamlit secrets or ENV (base URL or base+/ocr).")
    st.stop()

st.subheader("Capture from Phone Camera")
cam = st.camera_input("Take a photo")

st.subheader("Or Upload Image")
uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
)

if not cam and not uploaded:
    st.info("Capture or upload an image to start.")
    st.stop()

# Choose source
src = cam if cam else uploaded
img = Image.open(src)

if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

img = img.convert("RGB")
orig_w, orig_h = img.size
st.caption(f"Original image: **{orig_w} × {orig_h}px**")

# --------------------------
# ROI controls (in original pixel coords)
# --------------------------
default_strip_w = int(orig_w * 0.22)  # ~22% width
strip_w = st.sidebar.slider(
    "Strip width (px)",
    min_value=max(30, int(orig_w * 0.05)),
    max_value=max(60, int(orig_w * 0.80)),
    value=min(max(60, default_strip_w), int(orig_w * 0.80)),
    step=1
)

center_x = st.sidebar.slider(
    "Strip position (center X)",
    min_value=0,
    max_value=orig_w - 1,
    value=orig_w // 2,
    step=1
)

top_crop_pct = st.sidebar.slider("Top crop (%)", 0, 45, 0, 1)
bottom_crop_pct = st.sidebar.slider("Bottom crop (%)", 0, 45, 0, 1)

# Compute ROI box
x1 = int(max(0, center_x - strip_w / 2))
x2 = int(min(orig_w, x1 + strip_w))
# If we hit right edge, shift left to keep width consistent
if x2 - x1 < strip_w:
    x1 = int(max(0, x2 - strip_w))

y1 = int(orig_h * (top_crop_pct / 100.0))
y2 = int(orig_h * (1.0 - bottom_crop_pct / 100.0))
y1 = max(0, min(y1, orig_h - 1))
y2 = max(y1 + 1, min(y2, orig_h))

roi_pil = img.crop((x1, y1, x2, y2))

def draw_scanner_overlay(pil_img: Image.Image, x1, y1, x2, y2) -> Image.Image:
    out = pil_img.copy()
    d = ImageDraw.Draw(out, "RGBA")

    # Shade outside ROI
    d.rectangle([0, 0, x1, orig_h], fill=(0, 0, 0, 120))
    d.rectangle([x2, 0, orig_w, orig_h], fill=(0, 0, 0, 120))
    d.rectangle([x1, 0, x2, y1], fill=(0, 0, 0, 120))
    d.rectangle([x1, y2, x2, orig_h], fill=(0, 0, 0, 120))

    # ROI outline
    d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=4)

    # Center scan line
    cx = (x1 + x2) // 2
    d.line([cx, y1, cx, y2], fill=(0, 255, 0, 180), width=2)

    return out

overlay_img = draw_scanner_overlay(img, x1, y1, x2, y2)

# Display preview
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader("Preview (scanner strip)")
    st.image(overlay_img, use_container_width=True)

with c2:
    st.subheader("ROI that will be OCR’d")
    st.image(roi_pil, use_container_width=True)
    st.caption(f"ROI box: x={x1}:{x2}, y={y1}:{y2}  →  {x2-x1}×{y2-y1}px")

# --------------------------
# OCR call
# --------------------------
def call_ocr_api(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Backend HTTP {r.status_code}: {r.text[:400]}")
    return r.json()

run_ocr = st.button("Run OCR (ROI only)")

if run_ocr:
    try:
        api_out = call_ocr_api(roi_pil)
        status = api_out.get("status")
        message = api_out.get("message")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        # Annotate both ROI + Full image
        roi_annot = roi_pil.copy()
        full_annot = img.copy()

        draw_roi = ImageDraw.Draw(roi_annot)
        draw_full = ImageDraw.Draw(full_annot)
        font = ImageFont.load_default()

        rows = []
        detected_texts = []

        for idx, det in enumerate(detections, start=1):
            box = det["box"]          # points in ROI coordinates
            text = det["text"]
            score = float(det["score"])

            # ROI points
            pts_roi = [(int(p[0]), int(p[1])) for p in box]
            draw_roi.line(pts_roi + [pts_roi[0]], fill=(0, 255, 0), width=2)

            tx, ty = pts_roi[0]
            draw_roi.text((tx, max(0, ty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            # Full-image points (offset by ROI top-left)
            pts_full = [(px + x1, py + y1) for (px, py) in pts_roi]
            draw_full.line(pts_full + [pts_full[0]], fill=(0, 255, 0), width=2)

            ftx, fty = pts_full[0]
            draw_full.text((ftx, max(0, fty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            detected_texts.append(text)

            rows.append({
                "roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2,
                "detection_id": idx,
                "text": text,
                "score": score,
            })

        # Show results
        r1, r2 = st.columns([2, 1], gap="large")
        with r1:
            st.subheader("Full image (OCR boxes mapped back)")
            # draw ROI outline on full image too
            d = ImageDraw.Draw(full_annot)
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            st.image(full_annot, use_container_width=True)

        with r2:
            st.subheader("ROI annotated")
            st.image(roi_annot, use_container_width=True)

        if detected_texts:
            st.subheader("Detected Text")
            st.code(" | ".join(detected_texts))

        st.success(f"OCR complete — {len(rows)} detections")

        # Table + CSV download
        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Detections Table")
            st.dataframe(df, use_container_width=True)

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
