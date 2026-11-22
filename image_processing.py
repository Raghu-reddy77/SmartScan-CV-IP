"""
image_processing.py
-------------------
Image Enhancement Module for SmartScan.

Implements classical image-processing algorithms using OpenCV and NumPy:
  - Grayscale conversion
  - Gaussian Blur
  - Histogram Equalization
  - CLAHE  (Contrast Limited Adaptive Histogram Equalization)
  - Adaptive Thresholding  (Black & White scan)
  - Noise reduction (Non-Local Means Denoising)
  - Sharpening

Each function includes explanatory comments suitable for a viva voce discussion.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mode 1 – Original Color Scan (light clean-up only)
# ---------------------------------------------------------------------------

def enhance_color(image):
    """
    Produce a lightly enhanced full-color scan.

    Steps:
        1. Mild Gaussian Blur to reduce JPEG noise.
        2. CLAHE on the L-channel of LAB colour space for local contrast boost.
        3. Slight unsharp mask to restore perceived sharpness.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        np.ndarray: Enhanced BGR image.
    """
    # Step 1 – Mild blur (3×3) to suppress compression artefacts
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Step 2 – Convert to LAB colour space
    # LAB separates luminance (L) from colour (A, B), so we can enhance
    # contrast without shifting hues.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L-channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_eq  = cv2.merge([l_eq, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Step 3 – Unsharp mask: sharpened = original + weight*(original – blur)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened


# ---------------------------------------------------------------------------
# Mode 2 – Grayscale Scan
# ---------------------------------------------------------------------------

def enhance_grayscale(image):
    """
    Convert to grayscale and apply histogram equalization + CLAHE.

    Why CLAHE instead of plain equalizeHist?
        Global histogram equalization (equalizeHist) can over-brighten or
        over-darken uniform regions.  CLAHE divides the image into small tiles,
        equalizes each tile locally, and then blends them — giving much better
        results on documents with shadows or uneven lighting.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        np.ndarray: 3-channel grayscale (all channels equal) BGR image.
    """
    # Convert BGR → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise before equalization
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE: clipLimit prevents over-amplification of noise
    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Unsharp mask for sharpness
    blurred_g   = cv2.GaussianBlur(gray_clahe, (0, 0), 2.0)
    sharpened_g = cv2.addWeighted(gray_clahe, 1.5, blurred_g, -0.5, 0)

    # Return as 3-channel so Streamlit/PIL treats it uniformly
    return cv2.cvtColor(sharpened_g, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Mode 3 – Black & White Scan (high-contrast document look)
# ---------------------------------------------------------------------------

def enhance_bw(image):
    """
    Produce a clean black-and-white (thresholded) document scan.

    Algorithm:
        1. Convert to grayscale.
        2. Denoise with fastNlMeansDenoising (Non-Local Means).
        3. Apply Adaptive Thresholding.

    Why Adaptive Thresholding?
        Global thresholding uses a single cutoff value for the whole image.
        If lighting varies (shadows, uneven background), some text will be
        clipped.  Adaptive thresholding computes a local threshold for each
        small neighbourhood, so text under shadow remains legible.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        np.ndarray: 3-channel black-and-white BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise: h=10 is a mild filter — good for printed documents
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive Gaussian Thresholding
    # blockSize=11 → neighbourhood radius for local mean calculation
    # C=5          → constant subtracted from the local mean
    bw = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5,
    )

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Mode 4 – Enhanced Scan (best-quality all-round)
# ---------------------------------------------------------------------------

def enhance_advanced(image):
    """
    Apply a full enhancement pipeline for maximum document quality.

    Pipeline:
        1. Grayscale conversion.
        2. Non-Local Means denoising.
        3. CLAHE for contrast.
        4. Gamma correction to brighten mid-tones.
        5. Unsharp mask for final sharpness.
        6. Adaptive threshold → overlay on the CLAHE result for crisp text.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        np.ndarray: 3-channel enhanced BGR image (looks like a good scanner output).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)

    # CLAHE
    clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(denoised)

    # Gamma correction: gamma < 1 → brighten; gamma > 1 → darken
    # We use 0.85 to slightly brighten the mid-tones (mimics a bright scanner lamp).
    gamma     = 0.85
    inv_gamma = 1.0 / gamma
    table     = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype="uint8"
    )
    gamma_corrected = cv2.LUT(gray_clahe, table)

    # Unsharp mask
    blur_for_sharp = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
    sharpened      = cv2.addWeighted(gamma_corrected, 1.6, blur_for_sharp, -0.6, 0)

    # Adaptive threshold (for reference — not blended in this mode)
    # We produce a slightly softer result than pure B&W so images look good too.
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return result_bgr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PROCESSING_MODES = {
    "Original Color Scan": enhance_color,
    "Grayscale Scan":      enhance_grayscale,
    "Black & White Scan":  enhance_bw,
    "Enhanced Scan":       enhance_advanced,
}


def process_image(image, mode="Enhanced Scan"):
    """
    Process a BGR image with the requested enhancement mode.

    Args:
        image (np.ndarray): BGR image (as returned by OpenCV or converted from PIL).
        mode  (str):        One of PROCESSING_MODES keys.

    Returns:
        np.ndarray: Processed BGR image.

    Raises:
        ValueError: If the mode is not recognised.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    if mode not in PROCESSING_MODES:
        raise ValueError(
            f"Unknown processing mode '{mode}'. "
            f"Choose from: {list(PROCESSING_MODES.keys())}"
        )

    fn = PROCESSING_MODES[mode]
    return fn(image)
