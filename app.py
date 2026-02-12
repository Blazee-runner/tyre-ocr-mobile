import os
import requests
import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Auto ROI OCR", layout="centered")
st.title("📸 Auto ROI → CRAFT → OCR")

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
# IMAGE SOURCE
# =====================================================
st.subheader("Image Source")

src_mode = st.radio(
    "Choose input source",
    ["Upload Image", "Capture from Camera"],
    horizontal=True
)

img = None

if src_mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    )
    if uploaded:
        img = Image.open(uploaded)

else:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam)

if img is None:
    st.info("Upload or capture an image to continue")
    st.stop()

# =====================================================
# SAFE IMAGE LOAD
# =====================================================
try:
    img = ImageOps.exif_transpose(img)
except Exception:
    pass

img = img.convert("RGB")

st.image(img, caption="Input Image", use_column_width=True)

# =====================================================
# RUN AUTO OCR PIPELINE
# =====================================================
if st.button("Run Auto OCR 🚀"):

    img_buf = BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    files = {"file": ("image.jpg", img_buf, "image/jpeg")}

    with st.spinner("Running CRAFT + OCR pipeline... ⏳"):
        r = requests.post(
            f"{OCR_API_URL}/pipeline",
            files=files,
            timeout=600
        )

    if r.status_code != 200:
        st.error(f"Backend error: {r.text}")
        st.stop()

    result = r.json()

    st.success("OCR completed 🎉")

    # =================================================
    # SHOW RESULTS
    # =================================================
    if "stitched_image_url" in result:
        st.subheader("📄 OCR Result Image")
        st.image(result["stitched_image_url"], use_column_width=True)

    if "text" in result:
        st.subheader("📝 Extracted Text")
        st.text_area("OCR Output", result["text"], height=200)

    if "excel_url" in result:
        st.subheader("📊 Download")
        st.download_button(
            "Download Excel",
            data=requests.get(result["excel_url"]).content,
            file_name="ocr_output.xlsx"
        )
