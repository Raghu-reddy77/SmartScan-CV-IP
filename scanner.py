"""
scanner.py
----------
Document Detection and Perspective Transformation Module for SmartScan.

This module implements traditional OpenCV-based document detection:
  1. Preprocessing (grayscale, blur, edge detection)
  2. Contour detection and selection
  3. Four-point perspective transformation

Algorithms used are purely classical computer vision — no ML models required.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helper: Order four corner points consistently
# ---------------------------------------------------------------------------

def order_points(pts):
    """
    Order four corner points of a quadrilateral in a fixed sequence:
        [top-left, top-right, bottom-right, bottom-left]

    Why we need this:
        getPerspectiveTransform() requires the source and destination points
        in a consistent order.  Contour vertices can come in any order, so
        we sort them here before computing the transform.

    Args:
        pts (np.ndarray): Array of shape (4, 2) with four (x, y) coordinates.

    Returns:
        np.ndarray: Ordered array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest sum (x+y); bottom-right the largest.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    # Top-right has the smallest difference (y-x); bottom-left the largest.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# ---------------------------------------------------------------------------
# Helper: Compute output dimensions from ordered corners
# ---------------------------------------------------------------------------

def compute_output_size(ordered_pts):
    """
    Compute the width and height of the output (de-warped) image.

    Strategy:
        Width  = max of the two horizontal edge lengths.
        Height = max of the two vertical edge lengths.
    This preserves the full content without cropping.

    Args:
        ordered_pts (np.ndarray): (4, 2) ordered corner array.

    Returns:
        tuple: (width, height) as integers.
    """
    (tl, tr, br, bl) = ordered_pts

    width_bottom = np.linalg.norm(br - bl)
    width_top    = np.linalg.norm(tr - tl)
    max_width    = max(int(width_bottom), int(width_top))

    height_right = np.linalg.norm(br - tr)
    height_left  = np.linalg.norm(bl - tl)
    max_height   = max(int(height_right), int(height_left))

    return max_width, max_height


# ---------------------------------------------------------------------------
# Core: Detect document contour
# ---------------------------------------------------------------------------

def detect_document_contour(image):
    """
    Detect the largest rectangular (four-sided) contour in an image.

    Pipeline:
        1. Resize to a small working copy (≤ 500 px tall) for speed.
        2. Convert to grayscale   → cv2.cvtColor()
        3. Gaussian Blur          → reduces noise before edge detection.
        4. Canny Edge Detection   → finds sharp brightness transitions.
        5. Find contours          → cv2.findContours()
        6. Sort by area (largest first).
        7. Approximate each contour to a polygon → cv2.approxPolyDP()
        8. Accept the first contour with exactly 4 vertices.

    Args:
        image (np.ndarray): BGR image (as loaded by OpenCV / converted from PIL).

    Returns:
        tuple:
            screen_cnt (np.ndarray | None): (4, 1, 2) contour array in ORIGINAL
                image coordinates, or None if not found.
            debug_image (np.ndarray): Annotated image showing detected edges /
                contour (useful for display).
    """
    orig_h, orig_w = image.shape[:2]

    # --- 1. Resize for processing efficiency ---
    # We scale down to at most 500 px tall to keep Canny fast.
    target_h = 500
    scale    = target_h / orig_h if orig_h > target_h else 1.0
    proc_w   = int(orig_w * scale)
    proc_h   = int(orig_h * scale)
    resized  = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    # --- 2. Grayscale ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # --- 3. Gaussian Blur (kernel 5×5, sigma=0 → auto) ---
    # Blurring smooths texture/noise so Canny focuses on document edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 4. Canny Edge Detection ---
    # Lower threshold = 75, upper = 200.  These values work well for documents.
    edged = cv2.Canny(blurred, 75, 200)

    # --- 5 & 6. Find and sort contours by area (descending) ---
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # --- Debug image (draw edges) ---
    debug_image = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    screen_cnt = None

    # --- 7 & 8. Approximate each contour; pick the first 4-sided one ---
    for c in contours:
        # Perimeter of the contour
        peri = cv2.arcLength(c, True)
        # Approximate to a polygon; epsilon = 2 % of perimeter
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is not None:
        # Scale contour coordinates back to original image size
        screen_cnt = (screen_cnt / scale).astype("float32")

        # Draw detected contour on debug image (scaled to proc size for display)
        cv2.drawContours(debug_image, [screen_cnt.astype(int)], -1, (0, 255, 0), 3)

        # Also draw on a copy of the original
        annotated = image.copy()
        cv2.drawContours(annotated, [screen_cnt.astype(int)], -1, (0, 255, 0), 3)
        # Draw corner circles
        for pt in screen_cnt.reshape(-1, 2):
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        debug_image = annotated

    return screen_cnt, debug_image


# ---------------------------------------------------------------------------
# Core: Four-point perspective transformation
# ---------------------------------------------------------------------------

def four_point_transform(image, pts):
    """
    Apply a perspective (bird's-eye) transformation to straighten a document.

    How it works:
        Given four corner points of a document in a tilted/angled photo,
        cv2.getPerspectiveTransform() computes a 3×3 homography matrix M such
        that M maps the source corners to the four corners of a rectangle.
        cv2.warpPerspective() then applies that matrix to every pixel, producing
        a flat, properly-oriented scan.

    Args:
        image (np.ndarray): Source BGR image.
        pts (np.ndarray):   Four corner points, shape (4, 2) or (4, 1, 2).

    Returns:
        np.ndarray: Warped (de-perspectived) BGR image.

    Raises:
        ValueError: If pts does not contain exactly 4 points.
    """
    pts = pts.reshape(4, 2).astype("float32")

    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}")

    # Order points: tl, tr, br, bl
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute output dimensions
    max_width, max_height = compute_output_size(rect)

    # Destination rectangle corners
    dst = np.array(
        [
            [0,            0           ],
            [max_width - 1, 0          ],
            [max_width - 1, max_height - 1],
            [0,            max_height - 1],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix M
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the warp
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


# ---------------------------------------------------------------------------
# High-level: Full document scan pipeline
# ---------------------------------------------------------------------------

def scan_document(image):
    """
    Run the complete document-scanning pipeline on a BGR image.

    Steps:
        1. Detect the document contour.
        2. If found  → apply perspective transform.
        3. If not found → return the original image with a warning flag.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        dict with keys:
            "success"       (bool)        : True if document was detected.
            "warped"        (np.ndarray)  : Perspective-corrected image.
            "contour"       (np.ndarray | None): Detected contour.
            "debug_image"   (np.ndarray)  : Annotated debug image.
            "message"       (str)         : Human-readable status message.
    """
    result = {
        "success": False,
        "warped": image.copy(),
        "contour": None,
        "debug_image": image.copy(),
        "message": "",
    }

    try:
        contour, debug_image = detect_document_contour(image)
        result["debug_image"] = debug_image

        if contour is None:
            result["message"] = (
                "⚠️ Document boundary not detected automatically. "
                "The image will be processed without perspective correction. "
                "Try uploading a photo with clearer document edges."
            )
            return result

        # Perspective correction
        warped = four_point_transform(image, contour)

        result["success"]  = True
        result["warped"]   = warped
        result["contour"]  = contour
        result["message"]  = "✅ Document detected and perspective corrected successfully."

    except Exception as exc:
        result["message"] = f"❌ Error during document detection: {exc}"

    return result
