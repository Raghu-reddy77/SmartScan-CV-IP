"""
utils.py
--------
Utility helpers for SmartScan.

Provides:
  - Image validation
  - PIL ↔ OpenCV (NumPy) conversion
  - Smart resizing with aspect-ratio preservation
  - In-memory download buffer generation (PNG / JPEG)
  - Filename generation
"""

import io
import os
import time

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIN_DIMENSION      = 100   # px — reject images smaller than this
MAX_DIMENSION      = 8000  # px — reject images larger than this (usually corrupt)
MAX_FILE_SIZE_MB   = 20    # MB


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image_file(uploaded_file):
    """
    Validate a Streamlit UploadedFile object.

    Checks:
        - File extension is in ALLOWED_EXTENSIONS.
        - File size ≤ MAX_FILE_SIZE_MB.
        - File can be opened as an image by PIL.
        - Image dimensions are within [MIN_DIMENSION, MAX_DIMENSION].

    Args:
        uploaded_file: Streamlit UploadedFile.

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # Extension check
    name = uploaded_file.name.lower()
    ext  = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '.{ext}'. Upload JPG, JPEG or PNG."

    # Size check
    uploaded_file.seek(0, 2)          # seek to end
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)             # reset
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."

    # Try to open with PIL
    try:
        pil_img = Image.open(uploaded_file)
        pil_img.verify()          # checks for corruption
        uploaded_file.seek(0)     # verify() consumes the stream
    except Exception as exc:
        return False, f"Cannot open image: {exc}"

    # Dimension check
    try:
        pil_img2 = Image.open(uploaded_file)
        w, h = pil_img2.size
        uploaded_file.seek(0)
    except Exception:
        return False, "Cannot read image dimensions."

    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return False, (
            f"Image too small ({w}×{h} px). "
            f"Minimum required: {MIN_DIMENSION}×{MIN_DIMENSION} px."
        )
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        return False, (
            f"Image too large ({w}×{h} px). "
            f"Maximum allowed: {MAX_DIMENSION}×{MAX_DIMENSION} px."
        )

    return True, "Image is valid."


# ---------------------------------------------------------------------------
# Conversion: PIL ↔ OpenCV
# ---------------------------------------------------------------------------

def pil_to_cv2(pil_image):
    """
    Convert a PIL Image to an OpenCV BGR NumPy array.

    PIL uses RGB; OpenCV uses BGR, so we reverse the channel order.

    Args:
        pil_image (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: BGR image.
    """
    # Ensure the image is in RGB mode (handles RGBA, P, L, etc.)
    pil_rgb = pil_image.convert("RGB")
    np_rgb  = np.array(pil_rgb, dtype=np.uint8)
    bgr     = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    return bgr


def cv2_to_pil(cv2_image):
    """
    Convert an OpenCV BGR NumPy array to a PIL Image (RGB).

    Args:
        cv2_image (np.ndarray): BGR image.

    Returns:
        PIL.Image.Image: RGB PIL Image.
    """
    if len(cv2_image.shape) == 2:
        # Grayscale
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_with_aspect_ratio(image, max_width=None, max_height=None, inter=cv2.INTER_AREA):
    """
    Resize an image while preserving its aspect ratio.

    If both max_width and max_height are given, the image is scaled so that
    neither dimension exceeds the respective maximum.

    Args:
        image      (np.ndarray): Input BGR image.
        max_width  (int | None): Maximum allowed width.
        max_height (int | None): Maximum allowed height.
        inter               : OpenCV interpolation flag.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = image.shape[:2]

    if max_width is None and max_height is None:
        return image

    scale_w = (max_width  / w) if max_width  else float("inf")
    scale_h = (max_height / h) if max_height else float("inf")
    scale   = min(scale_w, scale_h)

    if scale >= 1.0:
        return image   # already within bounds

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=inter)


def resize_for_display(image, max_side=800):
    """
    Resize an image so its longest side is at most `max_side` pixels.
    Useful for displaying previews in Streamlit without lagging.

    Args:
        image    (np.ndarray): BGR image.
        max_side (int): Maximum pixels on any side.

    Returns:
        np.ndarray: Display-ready image.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Download buffer
# ---------------------------------------------------------------------------

def image_to_download_bytes(cv2_image, fmt="PNG", quality=95):
    """
    Convert an OpenCV image to an in-memory byte buffer for Streamlit download.

    Args:
        cv2_image (np.ndarray): BGR image.
        fmt       (str):        'PNG' or 'JPEG'.
        quality   (int):        JPEG quality (1–100); ignored for PNG.

    Returns:
        bytes: Encoded image data.
    """
    pil_img = cv2_to_pil(cv2_image)
    buf     = io.BytesIO()

    if fmt.upper() == "PNG":
        pil_img.save(buf, format="PNG", optimize=True)
    else:
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)

    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def generate_filename(base="smartscan", mode="Enhanced_Scan", ext="png"):
    """
    Generate a meaningful timestamped filename.

    Example:  smartscan_Enhanced_Scan_20250601_143022.png

    Args:
        base (str): Base name prefix.
        mode (str): Processing mode label.
        ext  (str): File extension without dot.

    Returns:
        str: Filename string.
    """
    ts   = time.strftime("%Y%m%d_%H%M%S")
    safe = mode.replace(" ", "_").replace("/", "-")
    return f"{base}_{safe}_{ts}.{ext}"


# ---------------------------------------------------------------------------
# Image info
# ---------------------------------------------------------------------------

def get_image_info(cv2_image, label="Image"):
    """
    Return a dict of basic metadata about an OpenCV image.

    Args:
        cv2_image (np.ndarray): Input image.
        label     (str):        Human-readable label.

    Returns:
        dict: Keys — label, width, height, channels, dtype, size_kb.
    """
    h, w   = cv2_image.shape[:2]
    ch     = cv2_image.shape[2] if cv2_image.ndim == 3 else 1
    size_b = cv2_image.nbytes
    return {
        "label":    label,
        "width":    w,
        "height":   h,
        "channels": ch,
        "dtype":    str(cv2_image.dtype),
        "size_kb":  round(size_b / 1024, 1),
    }
