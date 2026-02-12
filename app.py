import os
import json
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="ROI → OCR Pipeline", layout="wide")
st.title("ROI → CRAFT → OCR Pipeline")

OCR_API_URL = st.secrets.get(
    "OCR_API_URL",
    os.getenv("OCR_API_URL", "")
).rstrip("/")

if not OCR_API_URL:
    st.error("❌ OCR_API_URL not set in secrets or env")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("ROI Settings")
    stroke_width = st.slider("ROI border width", 1, 10, 3)
    stroke_color = st.color_picker("ROI color", "#FF0000")

    st.header("Display")
    max_canvas_width = st.slider("Max canvas width", 600, 2500, 1200)
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Backend")
    st.code(OCR_API_URL)

# =====================================================
# IMAGE INPUT
# =====================================================
uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
)

if not uploaded:
    st.info("Upload an image to start")
    st.stop()

# =====================================================
# LOAD IMAGE SAFELY
# =====================================================
img = Image.open(uploaded)

if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

img = img.convert("RGB")

# Re-encode (important for canvas stability)
buf = BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
img = Image.open(buf).convert("RGB")

orig_w, orig_h = img.size
img_np = np.array(img)

st.caption(f"Original image size: **{orig_w} × {orig_h}px**")

# =====================================================
# SCALE FOR CANVAS
# =====================================================
scale = min(1.0, max_canvas_width / orig_w)
canvas_w = int(orig_w * scale)
canvas_h = int(orig_h * scale)

img_display = img.resize((canvas_w, canvas_h), Image.BILINEAR)

# =====================================================
# DRAW ROI CANVAS
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
    key="roi_canvas"
)

objects = canvas.json_data["objects"] if canvas.json_data else []
st.write(f"**{len(objects)} ROI(s) drawn**")

# =====================================================
# RUN PIPELINE
# =====================================================
if st.button("Run CRAFT + OCR Pipeline 🚀"):

    if not objects:
        st.warning("Draw at least one ROI")
        st.stop()

    # -----------------------------
    # Convert ROIs to backend coords
    # -----------------------------
    rois = []

    for obj in objects:
        left = int(obj["left"] / scale)
        top = int(obj["top"] / scale)
        width = int(obj["width"] * obj.get("scaleX", 1) / scale)
        height = int(obj["height"] * obj.get("scaleY", 1) / scale)

        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(orig_w, x1 + width)
        y2 = min(orig_h, y1 + height)

        if x2 > x1 and y2 > y1:
            rois.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    if not rois:
        st.error("Invalid ROIs")
        st.stop()

    # -----------------------------
    # Send to FastAPI
    # -----------------------------
    img_buf = BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    files = {
        "file": ("image.jpg", img_buf, "image/jpeg")
    }

    data = {
        "rois": json.dumps(rois)
    }

    with st.spinner("Running full OCR pipeline... ⏳"):
        r = requests.post(
            f"{OCR_API_URL}/pipeline",
            files=files,
            data=data,
            timeout=600
        )

    if r.status_code != 200:
        st.error(f"Backend failed: {r.text}")
        st.stop()

    out = r.json()

    st.success("Pipeline completed 🎉")
    st.json(out)

