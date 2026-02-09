import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests
st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("Mobile Camera OCR")
st.markdown("""
<style>
/* Scanner container */
.scanner-wrapper {
    position: relative;
    width: 100%;
    max-width: 420px;
    margin: auto;
}

/* Scanner frame */
.scanner-box {
    position: absolute;
    top: 18%;
    left: 8%;
    width: 84%;
    height: 38%;
    border: 3px solid #00ff66;
    border-radius: 8px;
    box-shadow: 0 0 12px rgba(0,255,102,0.6);
    pointer-events: none;
}

/* Instruction text */
.scanner-text {
    text-align: center;
    color: #00ff66;
    font-weight: 600;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# Read backend URL from Streamlit Secrets or ENV
OCR_API_URL = None
if "OCR_API_URL" in st.secrets:
    OCR_API_URL = st.secrets["OCR_API_URL"]
else:
    OCR_API_URL = os.getenv("OCR_API_URL", "").strip()



with st.sidebar:
    st.header("Display")
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Backend")
    st.write("OCR API URL:")
    st.code(OCR_API_URL or "NOT SET")

if not OCR_API_URL:
    st.error("OCR_API_URL is not set. Add it in Streamlit secrets or ENV.")
    st.stop()

st.subheader("Scan Image")

st.markdown('<div class="scanner-wrapper">', unsafe_allow_html=True)

cam = st.camera_input(
    "Align text inside the box",
    key="scanner_cam"
)

st.markdown('<div class="scanner-box"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="scanner-text">Align text inside the frame</div>', unsafe_allow_html=True)

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
img_np = np.array(img)

st.caption(f"Original image: **{orig_w} × {orig_h}px**")



def call_ocr_api(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Backend HTTP {r.status_code}: {r.text[:400]}")
    return r.json()

run_ocr = st.button("Run OCR")

if run_ocr:
    try:
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        rows = []
        total = 0

        # FULL IMAGE OCR
        roi_id = 1
        roi_pil = img

        api_out = call_ocr_api(roi_pil)
        status = api_out.get("status")
        message = api_out.get("message")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        detected_texts = []

        for idx, det in enumerate(detections, start=1):
            box = det["box"]
            text = det["text"]
            score = float(det["score"])
            total += 1

            pts = [(int(p[0]), int(p[1])) for p in box]
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            tx, ty = pts[0]
            draw.text(
                (tx, max(0, ty - 12)),
                f"{idx}. {text}",
                fill=(255, 255, 0),
                font=font
            )

            detected_texts.append(text)

            rows.append({
                "ROI": roi_id,
                "detection_id": idx,
                "text": text,
                "score": score,
            })
        st.image(annotated)
        
        if detected_texts:
            st.subheader("Detected Text")
            combined_text = " | ".join(detected_texts)
            st.code(combined_text)


        st.success(f"OCR complete — {total} detections")

        # =========================
        # Table + CSV Download
        # =========================
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







