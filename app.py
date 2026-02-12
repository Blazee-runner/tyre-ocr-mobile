import os
import json
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="ROI → CRAFT → OCR", layout="wide")
st.title("ROI → CRAFT → OCR Pipeline")

# =====================================================
# BACKEND URL
# =====================================================
OCR_API_URL = st.secrets.get(
    "OCR_API_URL",
    os.getenv("OCR_API_URL", "")
).rstrip("/")

if not OCR_API_URL:
    st.error("❌ OCR_API_URL not set")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("ROI Settings")
    stroke_width = st.slider("ROI border width", 1, 10, 3)
    stroke_color = st.color_picker("ROI color", "#FF0000")

    st.header("Display")
    max_canvas_width = st.slider(
        "Max canvas width (px)",
        600, 1400, 1000   # 🔥 DO NOT exceed 1400
    )
    keep_exif = st.checkbox("Respect EXIF orientation", True)

    st.header("Backend")
    st.code(OCR_API_URL)

# =====================================================
# IMAGE SOURCE
# =====================================================
st.subheader("📸 Image Source")

src_mode = st.radio(
    "Choose input source",
    ["Upload Image", "Capture from Camera"],
    horizontal=True
)

img = None
src_key = "none"

if src_mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    )
    if uploaded:
        img = Image.open(uploaded)
        src_key = uploaded.name

else:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam)
        src_key = "camera"

if img is None:
    st.info("Upload or capture an image to continue")
    st.stop()

# =====================================================
# SAFE IMAGE LOAD
# =====================================================
if keep_exif:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

img = img.convert("RGB")

# 🔥 HARD RESET IMAGE (canvas fix)
buf = BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
img = Image.open(buf).convert("RGB")

orig_w, orig_h = img.size
st.caption(f"Original image size: **{orig_w} × {orig_h}px**")

# =====================================================
# SCALE FOR CANVAS (SAFE)
# =====================================================
scale = min(1.0, max_canvas_width / orig_w)
canvas_w = int(orig_w * scale)
canvas_h = int(orig_h * scale)

# 🔥 FORCE NumPy → PIL roundtrip
img_np = np.array(img)
img_display = Image.fromarray(img_np).resize(
    (canvas_w, canvas_h),
    Image.BILINEAR
)

# 🔍 SANITY CHECK (comment out later if you want)
st.image(img_display, caption="Canvas background preview")

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
    key=f"roi_canvas_{src_key}"
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
        st.error("Invalid ROI selection")
        st.stop()

    # -----------------------------
    # Send to FastAPI
    # -----------------------------
    img_buf = BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    files = {"file": ("image.jpg", img_buf, "image/jpeg")}
    data = {"rois": json.dumps(rois)}

    with st.spinner("Running OCR pipeline... ⏳"):
        r = requests.post(
            f"{OCR_API_URL}/pipeline",
            files=files,
            data=data,
            timeout=600
        )

    if r.status_code != 200:
        st.error(f"Backend error: {r.text}")
        st.stop()

    st.success("Pipeline completed 🎉")
    st.json(r.json())
