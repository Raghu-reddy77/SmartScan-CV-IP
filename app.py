"""
app.py
------
SmartScan – Intelligent Document Scanner & Image Enhancement System
Main Streamlit Application

Run with:
    streamlit run app.py

Navigation:
    • Home / Dashboard
    • Document Scanner
    • Image Enhancement
    • About Project
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from scanner import scan_document
from image_processing import process_image, PROCESSING_MODES
from utils import (
    validate_image_file,
    pil_to_cv2,
    cv2_to_pil,
    resize_for_display,
    image_to_download_bytes,
    generate_filename,
    get_image_info,
)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartScan – Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – Blue & White professional theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Roboto, sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d47a1 0%, #1565c0 60%, #1976d2 100%);
    color: #ffffff;
}
[data-testid="stSidebar"] .css-1d391kg, 
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e3f2fd !important;
}

/* ---- Header banner ---- */
.smartscan-header {
    background: linear-gradient(135deg, #0d47a1, #1976d2);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(13,71,161,0.25);
}
.smartscan-header h1 { margin: 0; font-size: 2.4rem; }
.smartscan-header p  { margin: 0.4rem 0 0; font-size: 1.05rem; opacity: 0.9; }

/* ---- Metric cards ---- */
.metric-card {
    background: #ffffff;
    border: 1px solid #e3f2fd;
    border-left: 5px solid #1976d2;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
.metric-card h3 { margin: 0; color: #1565c0; font-size: 1.5rem; }
.metric-card p  { margin: 0; color: #546e7a; font-size: 0.9rem; }

/* ---- Section headers ---- */
.section-header {
    color: #0d47a1;
    border-bottom: 3px solid #1976d2;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ---- Upload area ---- */
[data-testid="stFileUploader"] {
    border: 2px dashed #1976d2;
    border-radius: 10px;
    padding: 1rem;
    background: #f8fbff;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(21,101,192,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d47a1, #1565c0);
    box-shadow: 0 4px 12px rgba(21,101,192,0.4);
    transform: translateY(-1px);
}

/* ---- Download buttons ---- */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- Info / success / warning boxes ---- */
.info-box {
    background: #e3f2fd;
    border-left: 4px solid #1976d2;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #0d47a1;
}
.success-box {
    background: #e8f5e9;
    border-left: 4px solid #388e3c;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #1b5e20;
}
.warning-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e65100;
}

/* ---- Image captions ---- */
.img-label {
    text-align: center;
    font-weight: 600;
    color: #1565c0;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

/* ---- Radio buttons ---- */
.stRadio > div { flex-wrap: wrap; gap: 0.4rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "original_cv2":  None,   # original BGR image (OpenCV)
        "scanned_cv2":   None,   # perspective-corrected BGR image
        "processed_cv2": None,   # enhancement-processed BGR image
        "scan_success":  False,
        "scan_message":  "",
        "uploaded_name": "",
        "current_mode":  "Enhanced Scan",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem;'>
                <span style='font-size:3rem;'>📄</span>
                <h2 style='margin:0.3rem 0 0; color:#e3f2fd;'>SmartScan</h2>
                <p style='color:#bbdefb; font-size:0.85rem; margin:0;'>
                    Intelligent Document Scanner
                </p>
            </div>
            <hr style='border-color:#42a5f5; margin:1rem 0;'>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📷 Document Scanner", "🎨 Image Enhancement", "ℹ️ About Project"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#42a5f5; margin:1rem 0;'>", unsafe_allow_html=True)

        # Quick stats
        st.markdown("**Session Info**")
        if st.session_state.uploaded_name:
            st.markdown(f"📁 `{st.session_state.uploaded_name}`")
            st.markdown(
                f"{'✅ Document Detected' if st.session_state.scan_success else '⚠️ Manual Mode'}"
            )
        else:
            st.markdown("_No image loaded yet_")

        st.markdown(
            """
            <hr style='border-color:#42a5f5; margin:1rem 0;'>
            <p style='color:#90caf9; font-size:0.78rem; text-align:center;'>
                B.Tech CS — CV & IP Project<br>
                Built with Python + OpenCV
            </p>
            """,
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Image loader (shared between Scanner and Enhancement pages)
# ---------------------------------------------------------------------------

def render_upload_section(key_prefix="main"):
    """Render file uploader and return a loaded BGR image (or None)."""
    st.markdown(
        """
        <div class='info-box'>
            📤 <strong>Upload a photo</strong> of your document (JPG, JPEG, PNG).
            For best results, place the document on a contrasting background.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key=f"{key_prefix}_uploader",
        label_visibility="collapsed",
    )

    if uploaded is None:
        return None

    # Validate
    valid, msg = validate_image_file(uploaded)
    if not valid:
        st.error(f"❌ {msg}")
        return None

    # Load
    try:
        pil_img = Image.open(uploaded)
        pil_img = pil_img.convert("RGB")  # strip alpha / palette
        bgr     = pil_to_cv2(pil_img)
        st.session_state.uploaded_name = uploaded.name
        return bgr
    except Exception as exc:
        st.error(f"❌ Failed to load image: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page: Home / Dashboard
# ---------------------------------------------------------------------------

def page_home():
    # Header
    st.markdown(
        """
        <div class='smartscan-header'>
            <h1>📄 SmartScan</h1>
            <p>Intelligent Document Scanner & Image Enhancement System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📐", "Document Detection",   "Auto-detect and straighten tilted documents"),
        ("🔄", "Perspective Fix",      "Correct camera angle with homography transforms"),
        ("✨", "Image Enhancement",    "CLAHE, denoising, adaptive thresholding"),
        ("💾", "Export Ready",         "Download as PNG or JPEG instantly"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h3>{icon}</h3>
                    <p><strong>{title}</strong><br><small>{desc}</small></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Processing pipeline overview
    st.markdown("<h3 class='section-header'>🔬 Processing Pipeline</h3>", unsafe_allow_html=True)

    steps = [
        ("1", "Upload", "JPG/JPEG/PNG document photo", "#1565c0"),
        ("2", "Detect", "Canny edges + contour detection", "#0277bd"),
        ("3", "Correct", "Four-point perspective warp", "#00695c"),
        ("4", "Enhance", "CLAHE + denoising + sharpening", "#2e7d32"),
        ("5", "Export", "PNG or JPEG download", "#4a148c"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc, color) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style='text-align:center; padding:1rem;
                            background:#f8fbff; border-radius:10px;
                            border-top:4px solid {color};'>
                    <div style='font-size:1.8rem; font-weight:800;
                                color:{color};'>{num}</div>
                    <div style='font-weight:700; color:#263238;
                                margin:0.3rem 0 0.2rem;'>{title}</div>
                    <div style='font-size:0.8rem; color:#607d8b;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Algorithms table
    st.markdown("<h3 class='section-header'>📚 Algorithms Used</h3>", unsafe_allow_html=True)
    algo_data = {
        "Algorithm": [
            "Gaussian Blur", "Canny Edge Detection", "Contour Detection",
            "Approx Polygon", "Perspective Transform", "CLAHE",
            "Adaptive Thresholding", "NL-Means Denoising", "Unsharp Mask",
        ],
        "OpenCV Function": [
            "cv2.GaussianBlur()", "cv2.Canny()", "cv2.findContours()",
            "cv2.approxPolyDP()", "cv2.warpPerspective()", "cv2.createCLAHE()",
            "cv2.adaptiveThreshold()", "cv2.fastNlMeansDenoising()", "cv2.addWeighted()",
        ],
        "Purpose": [
            "Noise reduction before edge detection",
            "Find document boundary edges",
            "Extract closed regions from edges",
            "Simplify contour to 4 corners",
            "Flatten tilted document",
            "Local contrast enhancement",
            "Binarise text under uneven light",
            "Remove noise while preserving text",
            "Enhance fine details / text crispness",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(algo_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Document Scanner
# ---------------------------------------------------------------------------

def page_scanner():
    st.markdown(
        "<h2 class='section-header'>📷 Document Scanner</h2>", unsafe_allow_html=True
    )

    # --- Upload ---
    st.markdown("### Step 1 — Upload Your Document")
    bgr = render_upload_section(key_prefix="scanner")

    if bgr is None:
        st.markdown(
            """
            <div class='info-box'>
                👆 Upload a document image above to begin scanning.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Store original
    st.session_state.original_cv2 = bgr

    # --- Scan ---
    st.markdown("### Step 2 — Automatic Document Detection")

    if st.button("🔍 Detect & Scan Document", type="primary"):
        with st.spinner("Detecting document boundaries…"):
            scan_result = scan_document(bgr)
            st.session_state.scanned_cv2  = scan_result["warped"]
            st.session_state.scan_success = scan_result["success"]
            st.session_state.scan_message = scan_result["message"]

    # Always ensure scanned_cv2 is populated (use original as fallback)
    if st.session_state.scanned_cv2 is None:
        st.session_state.scanned_cv2 = bgr

    # Show message if available
    if st.session_state.scan_message:
        box_class = "success-box" if st.session_state.scan_success else "warning-box"
        st.markdown(
            f"<div class='{box_class}'>{st.session_state.scan_message}</div>",
            unsafe_allow_html=True,
        )

    # --- Side-by-side comparison ---
    st.markdown("### Step 3 — Preview")
    col_orig, col_scan = st.columns(2)

    orig_disp = resize_for_display(bgr, 700)
    scan_disp = resize_for_display(st.session_state.scanned_cv2, 700)

    with col_orig:
        st.markdown("<div class='img-label'>📸 Original Image</div>", unsafe_allow_html=True)
        st.image(cv2_to_pil(orig_disp), use_container_width=True)
        info = get_image_info(bgr, "Original")
        st.caption(f"Size: {info['width']}×{info['height']} px | Channels: {info['channels']}")

    with col_scan:
        st.markdown(
            f"<div class='img-label'>{'✅ Perspective Corrected' if st.session_state.scan_success else '📄 Original (No Correction)'}</div>",
            unsafe_allow_html=True,
        )
        st.image(cv2_to_pil(scan_disp), use_container_width=True)
        info2 = get_image_info(st.session_state.scanned_cv2, "Scanned")
        st.caption(f"Size: {info2['width']}×{info2['height']} px | Channels: {info2['channels']}")

    # --- Download scanned image ---
    st.markdown("### Step 4 — Download Scan")
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        png_bytes = image_to_download_bytes(st.session_state.scanned_cv2, "PNG")
        st.download_button(
            "⬇️ Download PNG",
            data=png_bytes,
            file_name=generate_filename("smartscan_doc", "Scan", "png"),
            mime="image/png",
        )
    with dl_col2:
        jpg_bytes = image_to_download_bytes(st.session_state.scanned_cv2, "JPEG", quality=92)
        st.download_button(
            "⬇️ Download JPEG",
            data=jpg_bytes,
            file_name=generate_filename("smartscan_doc", "Scan", "jpg"),
            mime="image/jpeg",
        )


# ---------------------------------------------------------------------------
# Page: Image Enhancement
# ---------------------------------------------------------------------------

def page_enhancement():
    st.markdown(
        "<h2 class='section-header'>🎨 Image Enhancement</h2>", unsafe_allow_html=True
    )

    # --- Upload (or use already-scanned image) ---
    source_bgr = None

    if st.session_state.scanned_cv2 is not None:
        use_prev = st.checkbox(
            "✅ Use scanned image from Document Scanner page", value=True
        )
        if use_prev:
            source_bgr = st.session_state.scanned_cv2

    if source_bgr is None:
        st.markdown("### Upload an Image")
        source_bgr = render_upload_section(key_prefix="enhance")

    if source_bgr is None:
        st.markdown(
            "<div class='info-box'>👆 Upload an image or run the scanner first.</div>",
            unsafe_allow_html=True,
        )
        return

    # --- Mode selection ---
    st.markdown("### Choose Enhancement Mode")
    mode_col, info_col = st.columns([1, 2])

    with mode_col:
        mode = st.radio(
            "Processing Mode",
            list(PROCESSING_MODES.keys()),
            index=list(PROCESSING_MODES.keys()).index(st.session_state.current_mode),
            label_visibility="collapsed",
        )
        st.session_state.current_mode = mode

    with info_col:
        descriptions = {
            "Original Color Scan":
                "🎨 **Original Color Scan** — Keeps colour information intact. "
                "Applies CLAHE on the LAB L-channel and mild sharpening.",
            "Grayscale Scan":
                "⚫ **Grayscale Scan** — Converts to single-channel greyscale. "
                "Applies CLAHE for local contrast, then unsharp mask.",
            "Black & White Scan":
                "📃 **Black & White Scan** — Adaptive Gaussian thresholding produces "
                "a crisp, printer-ready B&W document. Handles uneven lighting.",
            "Enhanced Scan":
                "✨ **Enhanced Scan** — Full pipeline: denoise → CLAHE → gamma "
                "correction → unsharp mask. Best all-round quality.",
        }
        st.markdown(
            f"<div class='info-box'>{descriptions.get(mode, '')}</div>",
            unsafe_allow_html=True,
        )

    # --- Process ---
    if st.button("⚡ Apply Enhancement", type="primary"):
        with st.spinner(f"Applying '{mode}'…"):
            try:
                processed = process_image(source_bgr, mode)
                st.session_state.processed_cv2 = processed
                st.markdown(
                    f"<div class='success-box'>✅ Enhancement applied: <strong>{mode}</strong></div>",
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"❌ Processing error: {exc}")
                st.session_state.processed_cv2 = None

    # --- Comparison ---
    if st.session_state.processed_cv2 is not None:
        st.markdown("### Before vs After")
        col_a, col_b = st.columns(2)

        src_disp  = resize_for_display(source_bgr, 700)
        proc_disp = resize_for_display(st.session_state.processed_cv2, 700)

        with col_a:
            st.markdown("<div class='img-label'>📸 Before</div>", unsafe_allow_html=True)
            st.image(cv2_to_pil(src_disp), use_container_width=True)

        with col_b:
            st.markdown(
                f"<div class='img-label'>✨ After — {mode}</div>", unsafe_allow_html=True
            )
            st.image(cv2_to_pil(proc_disp), use_container_width=True)

        # --- Histogram comparison ---
        with st.expander("📊 View Intensity Histograms (before vs after)"):
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            gray_src  = cv2.cvtColor(source_bgr,                        cv2.COLOR_BGR2GRAY)
            gray_proc = cv2.cvtColor(st.session_state.processed_cv2,    cv2.COLOR_BGR2GRAY)

            axes[0].hist(gray_src.ravel(),  256, [0, 256], color="#1565c0", alpha=0.8)
            axes[0].set_title("Before — Intensity Distribution", fontsize=10)
            axes[0].set_xlabel("Pixel Value"); axes[0].set_ylabel("Count")

            axes[1].hist(gray_proc.ravel(), 256, [0, 256], color="#2e7d32", alpha=0.8)
            axes[1].set_title("After  — Intensity Distribution", fontsize=10)
            axes[1].set_xlabel("Pixel Value"); axes[1].set_ylabel("Count")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # --- Download ---
        st.markdown("### Download Enhanced Image")
        dl1, dl2, _ = st.columns([1, 1, 2])
        safe_mode = mode.replace(" ", "_")
        with dl1:
            png_bytes = image_to_download_bytes(st.session_state.processed_cv2, "PNG")
            st.download_button(
                "⬇️ Download PNG",
                data=png_bytes,
                file_name=generate_filename("smartscan", safe_mode, "png"),
                mime="image/png",
            )
        with dl2:
            jpg_bytes = image_to_download_bytes(
                st.session_state.processed_cv2, "JPEG", quality=95
            )
            st.download_button(
                "⬇️ Download JPEG",
                data=jpg_bytes,
                file_name=generate_filename("smartscan", safe_mode, "jpg"),
                mime="image/jpeg",
            )


# ---------------------------------------------------------------------------
# Page: About Project
# ---------------------------------------------------------------------------

def page_about():
    st.markdown(
        "<h2 class='section-header'>ℹ️ About SmartScan</h2>", unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='smartscan-header'>
            <h1>📄 SmartScan</h1>
            <p>Intelligent Document Scanner & Image Enhancement System</p>
            <p style='opacity:0.75; font-size:0.9rem;'>
                B.Tech Computer Science · Computer Vision & Image Processing Subject Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_intro, tab_algo, tab_viva, tab_tech = st.tabs(
        ["📖 Introduction", "🔬 Algorithms", "🎓 Viva Q&A", "⚙️ Tech Stack"]
    )

    with tab_intro:
        st.markdown(
            """
            ## Problem Statement
            Photographs of documents captured by smartphones or webcams often suffer from:
            - **Perspective distortion** – the camera is rarely perfectly overhead.
            - **Uneven lighting & shadows** – ambient light rarely falls uniformly.
            - **Low contrast** – text blends into the background.
            - **Noise** – sensor noise makes text edges rough.
            - **Poor focus** – slight motion blur reduces readability.

            ## Project Objectives
            1. Automatically detect document boundaries using edge detection and contour analysis.
            2. Correct perspective distortion via homography (four-point transform).
            3. Enhance image quality through classical image-processing operations.
            4. Provide a clean, browser-based interface that requires no external APIs.

            ## Real-World Applications
            | Application | Description |
            |---|---|
            | 📋 Office Document Archival | Scan and digitise paper documents |
            | 📚 Student Note Digitisation | Convert handwritten notes to digital PDFs |
            | 🏦 KYC / ID Scanning | Capture and clean identity document photos |
            | 🏥 Medical Record Archival | Digitise prescriptions and reports |
            | 📜 Historical Document Preservation | Restore and clean old documents |
            """
        )

    with tab_algo:
        st.markdown(
            """
            ## Document Detection Pipeline

            ### 1. Gaussian Blur — `cv2.GaussianBlur(img, (5,5), 0)`
            Convolution with a Gaussian kernel smooths high-frequency noise.
            Without blurring, tiny texture variations produce thousands of false edges in Canny.
            The kernel size (5×5) and sigma (auto) are chosen to suppress sensor noise while
            keeping large edge transitions intact.

            ### 2. Canny Edge Detection — `cv2.Canny(blur, 75, 200)`
            John Canny's 1986 algorithm operates in four steps:
            1. Gradient magnitude & direction (Sobel operators).
            2. Non-maximum suppression – thins edges to single-pixel width.
            3. Double thresholding (75, 200) – classifies pixels as strong/weak/non-edge.
            4. Edge tracking by hysteresis – retains weak pixels connected to strong ones.
            Result: clean, thin edges that precisely trace document boundaries.

            ### 3. Contour Detection — `cv2.findContours()`
            Contours are curves that join all continuous points of the same intensity along a boundary.
            `RETR_LIST` retrieves all contours without hierarchy; `CHAIN_APPROX_SIMPLE` compresses
            horizontal/vertical/diagonal segments, keeping only endpoints.

            ### 4. Polygon Approximation — `cv2.approxPolyDP()`
            Douglas–Peucker algorithm simplifies a contour to fewer vertices while preserving shape.
            Epsilon = 2 % of perimeter → a document viewed from a slight angle approximates to
            exactly 4 vertices (the four corners).

            ### 5. Perspective Transform — `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`
            A 3×3 homography matrix M is computed from the four detected corners (source) and four
            corners of the target rectangle (destination).
            `warpPerspective` maps every source pixel to its destination position using bilinear
            interpolation, producing a flat, top-down view of the document.

            ## Enhancement Algorithms

            ### CLAHE — Contrast Limited Adaptive Histogram Equalization
            Standard histogram equalization redistributes pixel intensities globally.
            CLAHE divides the image into small tiles (default 8×8) and equalises each tile locally.
            A clip limit prevents over-amplification of noise in nearly-uniform regions.
            Result: much better contrast in shadowed or bright areas without washing out the image.

            ### Adaptive Thresholding — `cv2.adaptiveThreshold()`
            Instead of a single global threshold, a local threshold is computed for each pixel as
            the weighted average of a neighbourhood (block size = 11 × 11), minus a constant C = 5.
            This handles documents where lighting changes across the page.

            ### Non-Local Means Denoising — `cv2.fastNlMeansDenoising()`
            For each pixel, NLM searches for similar patches across the whole image and averages
            them. Because text strokes are repeated patterns, they are preserved; random noise
            averages away. Parameter h = 8–10 is a mild setting to avoid blurring text edges.
            """
        )

    with tab_viva:
        qa_pairs = [
            ("What is the purpose of Gaussian Blur before Canny?",
             "Gaussian Blur reduces high-frequency noise in the image. Canny detects intensity "
             "gradients; without blurring, sensor noise produces false edges. Blurring ensures "
             "only significant edges (like document boundaries) are detected."),
            ("Explain Canny Edge Detection.",
             "Canny (1986) uses Sobel operators to compute gradients, applies non-maximum "
             "suppression to thin edges, then double thresholding (low and high) to classify "
             "edges. Hysteresis connects weak edges to strong ones, producing clean, thin boundaries."),
            ("What is a contour in OpenCV?",
             "A contour is a curve joining all continuous points along a boundary that share the "
             "same colour or intensity. cv2.findContours() returns these as lists of (x,y) "
             "coordinate arrays."),
            ("How does perspective transformation work?",
             "Given four source corner points of a quadrilateral (the detected document corners) "
             "and four destination points (corners of a rectangle), cv2.getPerspectiveTransform() "
             "computes a 3×3 homography matrix. cv2.warpPerspective() applies this matrix to every "
             "pixel, mapping the skewed document to a flat rectangle."),
            ("What is CLAHE and how does it differ from equalizeHist?",
             "Global histogram equalization (equalizeHist) uses a single cumulative distribution "
             "function for the whole image, which can over-brighten some regions. CLAHE divides the "
             "image into tiles, equalises each locally with a clip limit that prevents noise "
             "amplification, then blends the tiles. Result: better local contrast, especially under "
             "uneven lighting."),
            ("Why use Adaptive Thresholding for the B&W scan?",
             "A global threshold fails when lighting varies across the document (e.g., one corner "
             "in shadow). Adaptive thresholding computes a local threshold for each pixel from its "
             "neighbourhood, so text remains visible even where the background is darker or brighter."),
            ("What is cv2.approxPolyDP and why is epsilon 2% of perimeter?",
             "approxPolyDP implements the Douglas–Peucker algorithm, reducing a contour's vertices "
             "while preserving its shape. Epsilon controls approximation tolerance. 2% of the "
             "perimeter gives a good balance: a rectangular document becomes 4 vertices, while "
             "noise/detail is removed."),
            ("What colour space does CLAHE enhancement use and why?",
             "LAB colour space. LAB separates luminance (L) from chrominance (A, B). By applying "
             "CLAHE only to the L channel, we enhance contrast without shifting the hues "
             "(colours remain natural)."),
            ("What is Non-Local Means Denoising?",
             "NLM denoising searches for similar small patches (templates) across the image and "
             "replaces the central pixel with a weighted average of matching patches. Similar "
             "patches (like repeating text strokes) reinforce each other; random noise averages "
             "out. It preserves edges better than Gaussian blur."),
            ("What is a homography matrix?",
             "A 3×3 matrix that maps points from one plane to another via a projective "
             "transformation. For document scanning, it maps the four corners of a tilted "
             "document in the camera plane to four corners of an axis-aligned rectangle, "
             "effectively removing perspective distortion."),
        ]
        for i, (q, a) in enumerate(qa_pairs, 1):
            with st.expander(f"Q{i}. {q}"):
                st.markdown(
                    f"<div class='success-box'><strong>Answer:</strong> {a}</div>",
                    unsafe_allow_html=True,
                )

    with tab_tech:
        st.markdown(
            """
            | Component | Technology |
            |---|---|
            | Language | Python 3.9+ |
            | UI Framework | Streamlit |
            | Computer Vision | OpenCV (cv2) |
            | Numerical Computing | NumPy |
            | Image I/O | Pillow (PIL) |
            | Plotting | Matplotlib |
            | Data Tables | Pandas |

            ## File Structure
            ```
            SmartScan/
            ├── app.py               ← Main Streamlit app & UI
            ├── scanner.py           ← Document detection & perspective transform
            ├── image_processing.py  ← Enhancement modes (CLAHE, B&W, etc.)
            ├── utils.py             ← Validation, conversion, download helpers
            ├── requirements.txt     ← pip dependencies
            ├── README.md            ← Full documentation
            ├── uploads/             ← Uploaded files (auto-created)
            └── output/              ← Exported scans (auto-created)
            ```

            ## System Architecture
            ```
            User Browser
                │
                ▼
            Streamlit (app.py)
            ├── Upload & Validate (utils.py)
            ├── Document Scanner (scanner.py)
            │   ├── Grayscale + Gaussian Blur
            │   ├── Canny Edge Detection
            │   ├── Contour Detection & Selection
            │   ├── Polygon Approximation (4 corners)
            │   └── Perspective Warp
            └── Image Enhancement (image_processing.py)
                ├── Color Mode (CLAHE on LAB-L)
                ├── Grayscale Mode (CLAHE + sharpen)
                ├── B&W Mode (Adaptive Threshold)
                └── Enhanced Mode (Full pipeline)
            ```
            """
        )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()

    if   "Home"        in page: page_home()
    elif "Scanner"     in page: page_scanner()
    elif "Enhancement" in page: page_enhancement()
    elif "About"       in page: page_about()


if __name__ == "__main__":
    main()
