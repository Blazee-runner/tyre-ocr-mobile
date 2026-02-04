import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
import cv2

from streamlit_drawable_canvas import st_canvas
from paddleocr import PaddleOCR


# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Tyre OCR Mobile Tool", layout="wide")
st.title("📷 Tyre OCR Annotation Tool (Mobile Ready)")

st.write("✅ Works on OnePlus 13R Mobile Browser")

# -------------------------------
# Step 1: Capture Image from Mobile Camera
# -------------------------------
st.header("Step 1: Capture Tyre Image")

camera_photo = st.camera_input("Take a Tyre Photo")

if camera_photo is None:
    st.warning("Please capture a tyre image to continue.")
    st.stop()

# Convert captured photo to PIL
image_orig = Image.open(camera_photo).convert("RGB")
st.image(image_orig, caption="Captured Image", use_container_width=True)

img_np = np.array(image_orig)
h, w = img_np.shape[:2]

# -------------------------------
# Step 2: Load PaddleOCR
# -------------------------------
st.header("Step 2: OCR Detection")

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

ocr = load_ocr()

st.info("Running OCR...")

result = ocr.ocr(img_np)

detected_texts = []

if result and result[0]:
    st.success("✅ Text Detected!")

    for line in result[0]:
        text = line[1][0]
        conf = float(line[1][1])

        detected_texts.append({
            "text": text,
            "confidence": conf
        })

    df = pd.DataFrame(detected_texts)

    st.subheader("Detected Text Output")
    st.dataframe(df)

else:
    st.error("No text detected.")
    st.stop()

# -------------------------------
# Step 3: ROI Annotation (Optional)
# -------------------------------
st.header("Step 3: Draw ROI Boxes (Optional)")

st.write("Draw rectangles on image for focused OCR detection.")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=3,
    stroke_color="red",
    background_image=image_orig,
    update_streamlit=True,
    height=500,
    drawing_mode="rect",
    key="canvas",
)

# -------------------------------
# Step 4: Extract ROI and OCR Again
# -------------------------------
if canvas_result.json_data and canvas_result.json_data["objects"]:

    st.subheader("ROI OCR Results")

    roi_rows = []
    idx = 1

    for obj in canvas_result.json_data["objects"]:
        left = int(obj["left"])
        top = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])

        x1, y1 = left, top
        x2, y2 = left + width, top + height

        roi = img_np[y1:y2, x1:x2]

        roi_result = ocr.ocr(roi)

        if roi_result and roi_result[0]:
            best = roi_result[0][0]
            txt = best[1][0]
            score = best[1][1]

            roi_rows.append({
                "ROI": idx,
                "Text": txt,
                "Score": score
            })

        idx += 1

    roi_df = pd.DataFrame(roi_rows)
    st.dataframe(roi_df)

    # Download CSV
    csv_buf = BytesIO()
    roi_df.to_csv(csv_buf, index=False)
    st.download_button("⬇ Download ROI OCR CSV",
                       csv_buf.getvalue(),
                       file_name="roi_ocr.csv",
                       mime="text/csv")

