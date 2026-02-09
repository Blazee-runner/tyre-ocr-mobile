import warnings
warnings.filterwarnings("ignore")

import os
import threading
from io import BytesIO

import cv2
import av
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings


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

OCR_API_URL = (st.secrets.get("OCR_API_URL", "") if hasattr(st, "secrets") else "") or os.getenv("OCR_API_URL", "")
OCR_API_URL = normalize_ocr_url(OCR_API_URL)

st.set_page_config(page_title="ROI_MRF", layout="wide")
st.title("Live Camera OCR (Horizontal Scanner ROI)")

with st.sidebar:
    st.header("Scanner ROI (Horizontal)")
    st.caption("Only the band between the 2 lines is visible + sent to OCR.")
    band_height_pct = st.slider("Band height (%)", 5, 80, 20, 1)
    band_center_pct = st.slider("Band center Y (%)", 0, 100, 50, 1)

    st.header("Backend")
    st.code(OCR_API_URL or "OCR_API_URL NOT SET")

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


# ============================================================
# Live video processor
# ============================================================
class ScannerProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_bgr = None
        self.band_height_pct = 20
        self.band_center_pct = 50

    def _compute_band(self, h: int):
        band_h = int(h * (self.band_height_pct / 100.0))
        band_h = max(10, min(band_h, h))

        center = int(h * (self.band_center_pct / 100.0))
        y1 = max(0, center - band_h // 2)
        y2 = min(h, y1 + band_h)
        if y2 - y1 < band_h:
            y1 = max(0, y2 - band_h)
        return y1, y2

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # store latest original frame for OCR capture
        with self.lock:
            self.latest_bgr = img.copy()

        y1, y2 = self._compute_band(h)

        # ✅ Black out everything except the ROI band
        out = np.zeros_like(img)               # whole frame black
        out[y1:y2, :, :] = img[y1:y2, :, :]    # only band visible

        # draw the 2 horizontal lines on the band boundary
        cv2.line(out, (0, y1), (w, y1), (0, 255, 0), 3)
        cv2.line(out, (0, y2), (w, y2), (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_bgr is None:
                return None
            return self.latest_bgr.copy()


# ============================================================
# WebRTC settings
# ============================================================
client_settings = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.subheader("Live Camera Preview (only ROI band visible)")

webrtc_ctx = webrtc_streamer(
    key="scanner",
    client_settings=client_settings,
    video_processor_factory=ScannerProcessor,
    async_processing=True,
)

# Update overlay controls live
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.band_height_pct = band_height_pct
    webrtc_ctx.video_processor.band_center_pct = band_center_pct

st.divider()

run_ocr = st.button("Run OCR on band (capture latest frame)")

if run_ocr:
    if not webrtc_ctx.video_processor:
        st.error("Camera not ready. Start the camera stream first.")
        st.stop()

    frame_bgr = webrtc_ctx.video_processor.get_latest_frame()
    if frame_bgr is None:
        st.error("No frame received yet. Wait 1–2 seconds and try again.")
        st.stop()

    # Compute ROI on captured frame
    h, w = frame_bgr.shape[:2]
    band_h = int(h * (band_height_pct / 100.0))
    band_h = max(10, min(band_h, h))
    center = int(h * (band_center_pct / 100.0))
    y1 = max(0, center - band_h // 2)
    y2 = min(h, y1 + band_h)
    if y2 - y1 < band_h:
        y1 = max(0, y2 - band_h)

    # ROI is full width between the 2 horizontal lines
    roi_bgr = frame_bgr[y1:y2, :, :]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)

    try:
        api_out = call_ocr_api(roi_pil)
        status = api_out.get("status", "unknown")
        message = api_out.get("message", "")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        # Prepare annotated FULL image ONLY (no ROI preview)
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        full_pil = Image.fromarray(full_rgb)
        full_annot = full_pil.copy()

        draw_full = ImageDraw.Draw(full_annot)
        font = ImageFont.load_default()

        # Draw band boundary lines
        W, H = full_annot.size
        draw_full.line([(0, y1), (W, y1)], fill=(0, 255, 0), width=4)
        draw_full.line([(0, y2), (W, y2)], fill=(0, 255, 0), width=4)

        rows = []
        detected_texts = []

        for idx, det in enumerate(detections, start=1):
            box = det.get("box")
            text = det.get("text", "")
            score = float(det.get("score", 0.0))
            if not box or len(box) < 4:
                continue

            # box is ROI coordinates → map to full image by adding y1
            pts_roi = [(int(p[0]), int(p[1])) for p in box]
            pts_full = [(px, py + y1) for (px, py) in pts_roi]

            draw_full.line(pts_full + [pts_full[0]], fill=(0, 255, 0), width=2)
            ftx, fty = pts_full[0]
            draw_full.text((ftx, max(0, fty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            detected_texts.append(text)
            rows.append({
                "band_y1": y1, "band_y2": y2,
                "detection_id": idx,
                "text": text,
                "score": score,
            })

        st.subheader("Captured frame (OCR mapped back)")
        st_image_auto(full_annot)

        if detected_texts:
            st.subheader("Detected Text")
            st.code(" | ".join(detected_texts))

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
