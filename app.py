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
from PIL import Image, ImageDraw, ImageFont

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


# ============================================================
# UI styling (scanner look)
# ============================================================
st.set_page_config(page_title="ROI_MRF", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      .scanner-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
      .scanner-sub { color: #9aa0a6; margin-top: 0; }
      .scan-pill {
        display:inline-block; padding:6px 10px; border-radius:999px;
        background: rgba(0,255,0,0.10); border: 1px solid rgba(0,255,0,0.25);
        color: #c8facc; font-size: 0.9rem;
      }
      .hint { color:#b0b6bd; font-size:0.95rem; }
      .stButton > button {
        border-radius: 14px;
        padding: 0.6rem 1rem;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="scanner-title">📷 Live Camera OCR Scanner</div>', unsafe_allow_html=True)
st.markdown('<p class="scanner-sub">Align text/barcode inside the band. Only that band goes to OCR.</p>', unsafe_allow_html=True)

if not OCR_API_URL:
    st.error("OCR_API_URL is not set. Add it in Streamlit secrets or ENV.")
    st.stop()

with st.sidebar:
    st.header("Scanner ROI")
    band_height_pct = st.slider("Band height (%)", 5, 80, 20, 1)
    band_center_pct = st.slider("Band center Y (%)", 0, 100, 50, 1)

    st.header("Scanner Effects")
    blur_strength = st.slider("Blur strength", 1, 31, 17, 2)  # must be odd-ish, we’ll fix below
    mask_darkness = st.slider("Mask darkness", 0.0, 0.9, 0.45, 0.05)
    show_scan_line = st.checkbox("Animated scan line", True)
    scan_speed = st.slider("Scan speed", 1, 25, 12, 1)

    st.header("Backend")
    st.code(OCR_API_URL)


def call_ocr_api(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("roi.png", buf, "image/png")}
    r = requests.post(OCR_API_URL, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Backend HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def compute_band(h: int, band_height_pct: int, band_center_pct: int):
    band_h = int(h * (band_height_pct / 100.0))
    band_h = max(10, min(band_h, h))
    center = int(h * (band_center_pct / 100.0))
    y1 = max(0, center - band_h // 2)
    y2 = min(h, y1 + band_h)
    if y2 - y1 < band_h:
        y1 = max(0, y2 - band_h)
    return y1, y2


# ============================================================
# Live video processor with blur + translucent mask + scan line
# ============================================================
class ScannerProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_bgr = None

        # updated from Streamlit
        self.band_height_pct = 20
        self.band_center_pct = 50
        self.blur_strength = 17
        self.mask_darkness = 0.45
        self.show_scan_line = True
        self.scan_speed = 12

        # scan line animation state
        self._scan_y = None
        self._scan_dir = 1

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # store latest original frame for capture
        with self.lock:
            self.latest_bgr = img.copy()

        y1, y2 = compute_band(h, self.band_height_pct, self.band_center_pct)

        # --- Blur background (outside ROI) ---
        k = int(self.blur_strength)
        if k % 2 == 0:
            k += 1
        k = max(1, min(k, 51))
        blurred = cv2.GaussianBlur(img, (k, k), 0)

        # Start with blurred everywhere
        out = blurred.copy()

        # Keep ROI band SHARP by copying from original
        out[y1:y2, :, :] = img[y1:y2, :, :]

        # Apply translucent dark mask outside ROI (scanner overlay)
        if self.mask_darkness > 0:
            dark = np.zeros_like(out)
            alpha = float(self.mask_darkness)
            # top
            out[:y1] = cv2.addWeighted(out[:y1], 1 - alpha, dark[:y1], alpha, 0)
            # bottom
            out[y2:] = cv2.addWeighted(out[y2:], 1 - alpha, dark[y2:], alpha, 0)

        # --- Draw ROI boundary lines ---
        line_color = (0, 255, 0)
        cv2.line(out, (0, y1), (w, y1), line_color, 3)
        cv2.line(out, (0, y2), (w, y2), line_color, 3)

        # --- Corner brackets (scanner style) ---
        pad = 14
        br = 30  # bracket length
        thickness = 3
        # top-left
        cv2.line(out, (pad, y1 + pad), (pad + br, y1 + pad), line_color, thickness)
        cv2.line(out, (pad, y1 + pad), (pad, y1 + pad + br), line_color, thickness)
        # top-right
        cv2.line(out, (w - pad, y1 + pad), (w - pad - br, y1 + pad), line_color, thickness)
        cv2.line(out, (w - pad, y1 + pad), (w - pad, y1 + pad + br), line_color, thickness)
        # bottom-left
        cv2.line(out, (pad, y2 - pad), (pad + br, y2 - pad), line_color, thickness)
        cv2.line(out, (pad, y2 - pad), (pad, y2 - pad - br), line_color, thickness)
        # bottom-right
        cv2.line(out, (w - pad, y2 - pad), (w - pad - br, y2 - pad), line_color, thickness)
        cv2.line(out, (w - pad, y2 - pad), (w - pad, y2 - pad - br), line_color, thickness)

        # --- Animated scan line ---
        if self.show_scan_line:
            if self._scan_y is None or not (y1 <= self._scan_y <= y2):
                self._scan_y = y1

            self._scan_y += self._scan_dir * int(max(1, self.scan_speed))
            if self._scan_y >= y2:
                self._scan_y = y2
                self._scan_dir = -1
            elif self._scan_y <= y1:
                self._scan_y = y1
                self._scan_dir = 1

            # glow effect: draw 3 lines
            cv2.line(out, (0, self._scan_y), (w, self._scan_y), (0, 255, 0), 2)
            cv2.line(out, (0, self._scan_y - 2), (w, self._scan_y - 2), (0, 255, 0), 1)
            cv2.line(out, (0, self._scan_y + 2), (w, self._scan_y + 2), (0, 255, 0), 1)

        # Small label
        cv2.putText(out, "ALIGN CODE INSIDE BAND", (14, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 255, 160), 2, cv2.LINE_AA)

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

colA, colB = st.columns([3, 2], gap="large")
with colA:
    st.markdown('<span class="scan-pill">LIVE</span> <span class="hint">Allow camera access in your browser.</span>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="scanner",
        client_settings=client_settings,
        video_processor_factory=ScannerProcessor,
        async_processing=True,
    )

with colB:
    st.markdown("### ✅ Scan")
    st.markdown("- Keep barcode/text within the green band\n- Press **Run OCR**")
    run_ocr = st.button("Run OCR (send ROI band)")

# Push slider settings into processor (live)
if webrtc_ctx.video_processor:
    vp = webrtc_ctx.video_processor
    vp.band_height_pct = band_height_pct
    vp.band_center_pct = band_center_pct
    vp.blur_strength = blur_strength
    vp.mask_darkness = mask_darkness
    vp.show_scan_line = show_scan_line
    vp.scan_speed = scan_speed

st.divider()


# ============================================================
# OCR action — show ONLY ROI band result (annotated)
# ============================================================
if run_ocr:
    if not webrtc_ctx.video_processor:
        st.error("Camera not ready. Start the camera stream first.")
        st.stop()

    frame_bgr = webrtc_ctx.video_processor.get_latest_frame()
    if frame_bgr is None:
        st.error("No frame received yet. Wait 1–2 seconds and try again.")
        st.stop()

    h, w = frame_bgr.shape[:2]
    y1, y2 = compute_band(h, band_height_pct, band_center_pct)

    roi_bgr = frame_bgr[y1:y2, :, :]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)

    try:
        api_out = call_ocr_api(roi_pil)
        status = api_out.get("status", "unknown")
        message = api_out.get("message", "")
        detections = api_out.get("detections", [])

        st.info(f"OCR status: {status} — {message}")

        roi_annot = roi_pil.copy()
        draw_roi = ImageDraw.Draw(roi_annot)
        font = ImageFont.load_default()

        rows = []
        detected_texts = []

        for idx, det in enumerate(detections, start=1):
            box = det.get("box")
            text = det.get("text", "")
            score = float(det.get("score", 0.0))
            if not box or len(box) < 4:
                continue

            pts = [(int(p[0]), int(p[1])) for p in box]
            draw_roi.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            tx, ty = pts[0]
            draw_roi.text((tx, max(0, ty - 12)), f"{idx}. {text}", fill=(255, 255, 0), font=font)

            detected_texts.append(text)
            rows.append({
                "band_y1": y1,
                "band_y2": y2,
                "detection_id": idx,
                "text": text,
                "score": score,
            })

        st.subheader("ROI band sent to OCR (annotated)")
        st_image_auto(roi_annot)

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
