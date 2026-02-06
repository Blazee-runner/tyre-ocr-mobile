import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests

# Read backend URL from Streamlit Secrets or ENV
OCR_API_URL = None
if "OCR_API_URL" in st.secrets:
    OCR_API_URL = st.secrets["OCR_API_URL"]
else:
    OCR_API_URL = os.getenv("OCR_API_URL", "").strip()

st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("ROI_MRF – Mobile Camera OCR (Hosted)")

with st.sidebar:
    st.header("ROI Settings")
    stroke_width = st.slider("ROI border width", 1, 10, 3)
    stroke_color = st.color_picker("ROI color", "#FF0000")

    st.header("Display")
    max_canvas_width = st.slider("Max canvas width (px)", 600, 2500, 1200)
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Backend")
    st.write("OCR API URL:")
    st.code(OCR_API_URL or "NOT SET")

if not OCR_API_URL:
    st.error("OCR_API_URL is not set. Add it in Streamlit secrets or ENV.")
    st.stop()

st.subheader("Capture from Phone Camera")
cam = st.camera_input("Take a photo (works on HTTPS)")

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

# Canvas scaling
scale = min(1.0, max_canvas_width / orig_w)
canvas_w = int(orig_w * scale)
canvas_h = int(orig_h * scale)
img_display = img.resize((canvas_w, canvas_h), Image.BILINEAR)

canvas = st_canvas(
    background_image=img_display,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="rect",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    fill_color="rgba(0,0,0,0)",
    update_streamlit=True,
    key="roi_canvas",
)

objects = canvas.json_data["objects"] if (canvas.json_data and "objects" in canvas.json_data) else []
st.write(f"**{len(objects)} ROI(s) drawn**")

def call_ocr_api(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Backend HTTP {r.status_code}: {r.text[:400]}")
    return r.json()

run_ocr = st.button("Run OCR (Annotate + CSV)")

if run_ocr:
    try:
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        rows = []
        total = 0

        # If no ROI drawn -> OCR on full image
        rois_to_process = objects if len(objects) > 0 else [{
            "left": 0, "top": 0, "width": canvas_w, "height": canvas_h, "scaleX": 1, "scaleY": 1
        }]

        for roi_id, obj in enumerate(rois_to_process, start=1):
            sx = abs(obj.get("scaleX", 1))
            sy = abs(obj.get("scaleY", 1))

            x1 = int(obj["left"] / scale)
            y1 = int(obj["top"] / scale)
            x2 = int((obj["left"] + obj["width"] * sx) / scale)
            y2 = int((obj["top"] + obj["height"] * sy) / scale)

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            roi_pil = img.crop((x1, y1, x2, y2))

            with st.expander(f"ROI {roi_id} preview ({x2-x1}×{y2-y1})", expanded=False):
                st.image(roi_pil, use_container_width=True)

            # Draw ROI box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            api_out = call_ocr_api(roi_pil)
            detections = api_out.get("detections", [])
            dbg = api_out.get("debug", {})
            st.write(f"ROI {roi_id} backend debug:", dbg)

            for det in detections:
                box = det["box"]
                text = det["text"]
                score = float(det["score"])
                total += 1

                pts = [(int(x1 + p[0]), int(y1 + p[1])) for p in box]

                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
                tx, ty = pts[0]
                draw.text(
                    (tx, max(0, ty - 12)),
                    f"{text} ({score:.2f})",
                    fill=(255, 255, 0),
                    font=font
                )

                rows.append({
                    "ROI": roi_id,
                    "roi_left": x1,
                    "roi_top": y1,
                    "roi_width": x2 - x1,
                    "roi_height": y2 - y1,
                    "text": text,
                    "score": score,
                })

        st.success(f"OCR complete — {total} detections")

        st.subheader("Annotated Image")
        st.image(annotated, use_container_width=True)

        img_buf = BytesIO()
        annotated.save(img_buf, format="PNG")
        st.download_button("Download Annotated Image", img_buf.getvalue(), "annotated.png", "image/png")

        df = pd.DataFrame(rows)
        st.subheader("Detections Table")
        st.dataframe(df, use_container_width=True)

        csv_buf = BytesIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), "ocr_results.csv", "text/csv")

        # If still 0 -> show hint
        if total == 0:
            st.warning(
                "0 detections. Try:\n"
                "- Draw ROI tighter around text\n"
                "- Ensure text is not blurry\n"
                "- Increase light / reduce glare\n"
                "- Try full image OCR (don’t draw ROI)"
            )

    except Exception as e:
        st.error(f"OCR failed: {e}")
