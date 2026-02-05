import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from io import BytesIO
import requests

OCR_API_URL = "https://nigel-orthodontic-unhypothetically.ngrok-free.dev/ocr"


# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("ROI_MRF – Any Image Size OCR")

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("ROI Settings")
    stroke_width = st.slider("ROI border width", 1, 10, 3)
    stroke_color = st.color_picker("ROI color", "#FF0000")

    st.header("Display")
    max_canvas_width = st.slider("Max canvas width (px)", 600, 2500, 1200)
    keep_exif = st.checkbox("Respect EXIF orientation", True)
# =====================================================
# Upload
# =====================================================
camera_image = st.camera_input("Capture image using phone camera")

if not camera_image:
    st.info("Capture an image to start.")
    st.stop()

img = Image.open(camera_image)
if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

img = img.convert("RGB")  # 👈 ADD THIS
# =====================================================
# Load image
# =====================================================
if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

orig_w, orig_h = img.size
img_np = np.array(img)

st.caption(f"Original image size: **{orig_w} × {orig_h}px**")

# =====================================================
# Scale image for canvas (display only)
# =====================================================
scale = min(1.0, max_canvas_width / orig_w)
canvas_w = int(orig_w * scale)
canvas_h = int(orig_h * scale)

img_display = img.resize((canvas_w, canvas_h), Image.BILINEAR)

# =====================================================
# Canvas
# =====================================================
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

objects = canvas.json_data["objects"] if canvas.json_data else []
st.write(f"**{len(objects)} ROI(s) drawn**")

def call_ocr_api(roi_np):
    import cv2

    roi_pil = Image.fromarray(roi_np).convert("L")
    roi_pil = ImageOps.autocontrast(roi_pil)

    w, h = roi_pil.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        roi_pil = roi_pil.resize(
            (int(w * scale), int(h * scale)),
            Image.BICUBIC
        )

    roi_np = np.array(roi_pil)

    # CLAHE only (NO threshold)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_np = clahe.apply(roi_np)

    roi_pil = Image.fromarray(roi_np)
    buf = BytesIO()
    roi_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    files = {"file": ("roi.jpg", buf, "image/jpeg")}
    r = requests.post(OCR_API_URL, files=files, timeout=60)
    r.raise_for_status()
    return r.json()["detections"]


# =====================================================
# OCR
# =====================================================
run_ocr = st.button("Run OCR (Annotate + CSV)")

if run_ocr:
    try:
        # 🔑 ALWAYS convert to RGB before drawing
        annotated = img.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        rows = []
        total = 0

        for roi_id, obj in enumerate(objects, start=1):
            # Canvas → original coordinates
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width = int(obj["width"] * obj.get("scaleX", 1) / scale)
            height = int(obj["height"] * obj.get("scaleY", 1) / scale)



            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(orig_w, x1 + width)
            y2 = min(orig_h, y1 + height)

            if x2 <= x1 or y2 <= y1:
                continue
            roi = img_np[y1:y2, x1:x2]
            
            h, w = roi.shape[:2]
            if h < 40 or w < 40:
                continue
            
            # 🔄 Auto-rotate vertical tyre text
            if h > w * 1.2:
                roi = np.rot90(roi, k=1)


            # debug (remove later)
            st.image(roi, caption=f"ROI {roi_id}", width=300)
            
            detections = call_ocr_api(roi)


            for det in detections:
                box = det["box"]
                text = det["text"]
                score = det["score"]

                total += 1
                # Map ROI-local box → global image
                pts = [(int(x1 + p[0]), int(y1 + p[1])) for p in box]

                # Draw polygon
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

                # Draw text label
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
                    "gx1": pts[0][0], "gy1": pts[0][1],
                    "gx2": pts[1][0], "gy2": pts[1][1],
                    "gx3": pts[2][0], "gy3": pts[2][1],
                    "gx4": pts[3][0], "gy4": pts[3][1],
                    "text": text,
                    "score": float(score),
                })

        st.success(f"OCR complete — {total} detections")

        # =====================================================
        # Outputs
        # =====================================================
        st.subheader("Annotated Image (Original Resolution)")
        st.image(annotated, use_column_width=True)

        img_buf = BytesIO()
        annotated.save(img_buf, format="PNG")

        st.download_button(
            "Download Annotated Image",
            img_buf.getvalue(),
            file_name="annotated.png",
            mime="image/png"
        )

        df = pd.DataFrame(rows)
        csv_buf = BytesIO()
        df.to_csv(csv_buf, index=False)

        st.download_button(
            "Download OCR CSV",
            csv_buf.getvalue(),
            file_name="ocr_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(
            "OCR failed.\n\n"
            f"Details: {e}\n\n"
            "Make sure the OCR backend is running and reachable."
        )











