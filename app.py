# app.py  (Streamlit Frontend - Barcode Scanner Style ROI)
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests


# ============================================================
# Backend URL from Streamlit Secrets or ENV
#   - You can set either:
#       OCR_API_URL = https://xxxx.ngrok-free.app
#     OR
#       OCR_API_URL = https://xxxx.ngrok-free.app/ocr
# ============================================================
def normalize_ocr_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    url = url.rstrip("/")
    if not url.endswith("/ocr"):
        url = url + "/ocr"
    return url


OCR_API_URL = None
if "OCR_API_URL" in st.secrets:
    OCR_API_URL = str(st.secrets["OCR_API_URL"]).strip()
else:
    OCR_API_URL = os.getenv("OCR_API_URL", "").strip()

OCR_API_URL = normalize_ocr_url(OCR_API_URL)

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("Mobile Camera OCR (Barcode-Scanner ROI)")

with st.sidebar:
    st.header("Display")
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Scanner ROI")
    st.caption("Two vertical lines mark the ROI. Only the area between them is sent to OCR.")
    roi_width_pct = st.slider("ROI width (%)", 5, 90, 22, 1)  # default 22%
    roi_center_x_pct = st.slider("ROI center X (%)", 0, 100, 50, 1)  # default middle

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

# ============================================================
# Load image
# ============================================================
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


# ============================================================
# Scanner overlay (two vertical lines)
# ============================================================
def draw_two_line_scanner(pil_img: Image.Image, x1: int, x2: int) -> Image.Image:
    """
    Draws two vertical green lines at x1 and x2 and shades outside the ROI.
    Returns a pure RGB PIL image (Streamlit-safe).
    """
    base = pil_img.convert("RGB").copy()
    w, h = base.size

    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Shade outside ROI
    d.rectangle([0, 0, x1, h], fill=(0, 0, 0, 120))
    d.rectangle([x2, 0, w, h], fill=(0, 0, 0, 120))

    # Two vertical lines
    d.line([x1, 0, x1, h], fill=(0, 255, 0, 255), width=4)
    d.line([x2, 0, x2, h], fill=(0, 255, 0, 255), width=4)

    out = Image.alpha_composite(rgba, overlay).convert("RGB")
    return out


# Compute ROI in original coordinates
roi_w = int(orig_w * (roi_width_pct / 100.0))
roi_w = max(30, min(roi_w, orig_w))

center_x = int(orig_w * (roi_center_x_pct / 100.0))
center_x = max(0, min(center_x, orig_w - 1))

x1 = int(max(0, center_x - roi_w / 2))
x2 = int(min(orig_w, x1 + roi_w))
if x2 - x1 < roi_w:
    x1 = int(max(0, x2 - roi_w))

# Full-height ROI (barcode scanner feel)
y1, y2 = 0, orig_h

roi_pil = img.crop((x1, y1, x2, y2))
overlay_img = draw_two_line_scanner(img, x1, x2)

# Show preview + ROI
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader("Preview (scanner ROI)")
    st.image(np.array(overlay_img, dtype=np.uint8), channels="RGB", use_container_width=True)
    st.caption(f"ROI X range: {x1} → {x2} (width={x2-x1}px)")

with c2:
    st.subheader("ROI that will be OCR’d")
    st.image(np.array(roi_pil, dtype=np.uint8), channels="RGB", use_container_width=True)
    st.caption(f"ROI size: {roi_pil.size[0]}×{roi_pil.size[1]} px")


# ============================================================
# OCR Call
# ============================================================
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

        status = api_out.get("status", "unknown")
        message = api_out.get("message", "")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        # Annotate full image + ROI image
        full_annot = img.copy()
        roi_annot = roi_pil.copy()

        draw_full = ImageDraw.Draw(full_annot)
        draw_roi = ImageDraw.Draw(roi_annot)
        font = ImageFont.load_default()

        # Draw ROI lines on full image too
        draw_full.line([x1, 0, x1, orig_h], fill=(0, 255, 0), width=4)
        draw_full.line([x2, 0, x2, orig_h], fill=(0, 255, 0), width=4)

        rows = []
        detected_texts = []

        for idx, det in enumerate(detections, start=1):
            box = det.get("box")
            text = det.get("text", "")
            score = float(det.get("score", 0.0))

            if not box or len(box) < 4:
                continue

            # Points in ROI coordinates
            pts_roi = [(int(p[0]), int(p[1])) for p in box]
            draw_roi.line(pts_roi + [pts_roi[0]], fill=(0, 255, 0), width=2)
            tx, ty = pts_roi[0]
            draw_roi.text((tx, max(0, ty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            # Map to full image coordinates (offset x by x1, y by y1 which is 0)
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

        # Show outputs
        r1, r2 = st.columns([2, 1], gap="large")
        with r1:
            st.subheader("Full image (OCR mapped back)")
            st.image(np.array(full_annot, dtype=np.uint8), channels="RGB", use_container_width=True)

        with r2:
            st.subheader("ROI annotated")
            st.image(np.array(roi_annot, dtype=np.uint8), channels="RGB", use_container_width=True)

        if detected_texts:
            st.subheader("Detected Text")
            st.code(" | ".join(detected_texts))

        st.success(f"OCR complete — {len(rows)} detections")

        # Table + CSV
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
