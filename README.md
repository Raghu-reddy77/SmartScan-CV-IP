# SmartScan – Intelligent Document Scanner & Image Enhancement System

> **B.Tech Computer Science | Computer Vision & Image Processing Project**  
> Built entirely with Python, OpenCV, NumPy, and Streamlit — no external APIs required.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Real-World Applications](#4-real-world-applications)
5. [Technology Stack](#5-technology-stack)
6. [System Architecture](#6-system-architecture)
7. [Image Processing Workflow](#7-image-processing-workflow)
8. [Algorithms Used](#8-algorithms-used)
9. [Installation Instructions](#9-installation-instructions)
10. [How to Run](#10-how-to-run)
11. [Project Structure](#11-project-structure)
12. [Features](#12-features)
13. [Future Enhancements](#13-future-enhancements)
14. [Viva Questions & Answers](#14-viva-questions--answers)
15. [Conclusion](#15-conclusion)

---

## 1. Introduction

**SmartScan** is an intelligent document scanner and image enhancement system that transforms
ordinary photographs of documents (taken with a smartphone or webcam) into clean, properly aligned,
publication-quality digital scans — entirely through classical computer vision and image-processing
algorithms implemented in Python and OpenCV.

The application provides a modern, browser-based interface powered by Streamlit, requiring only a
standard Python environment with no paid APIs, cloud services, or machine-learning frameworks.

---

## 2. Problem Statement

Photographs of documents captured in real-world conditions commonly suffer from:

| Problem | Description |
|---|---|
| **Perspective distortion** | Camera held at an angle instead of directly overhead |
| **Uneven lighting** | Shadows cast by the photographer or surrounding objects |
| **Low contrast** | Light-coloured ink on light paper, faded documents |
| **Sensor noise** | Grainy images from low-light smartphone cameras |
| **Motion blur** | Slight hand-shake during capture |
| **Background clutter** | Table surface, objects around the document |

These defects make photos unsuitable for archiving, sharing, or OCR processing.
SmartScan addresses all of them using traditional image-processing techniques.

---

## 3. Project Objectives

1. Detect document boundaries automatically using edge detection and contour analysis.
2. Correct perspective distortion by computing and applying a homography transformation.
3. Enhance image quality through contrast enhancement, denoising, and sharpening.
4. Provide multiple enhancement modes to suit different document types.
5. Allow lossless export in PNG or high-quality JPEG format.
6. Demonstrate practical implementation of the core CV & IP algorithms taught in class.

---

## 4. Real-World Applications

| Domain | Use Case |
|---|---|
| 📋 **Office / Enterprise** | Scan and archive paper documents to a DMS |
| 📚 **Education** | Digitise handwritten notes and lecture slides |
| 🏦 **Banking / KYC** | Capture and clean identity document photographs |
| 🏥 **Healthcare** | Digitise prescriptions, lab reports, and consent forms |
| 📜 **Cultural Heritage** | Restore and clean old historical documents |
| 🏗️ **Engineering** | Archive blueprints and hand-drawn schematics |
| 📱 **Mobile Apps** | Backend engine for iOS/Android document scanner apps |

---

## 5. Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Programming Language | Python | 3.9+ |
| UI Framework | Streamlit | ≥ 1.32 |
| Computer Vision | OpenCV (`cv2`) | ≥ 4.9 |
| Numerical Computing | NumPy | ≥ 1.26 |
| Image I/O | Pillow (PIL) | ≥ 10.2 |
| Plotting | Matplotlib | ≥ 3.8 |
| Data Tables | Pandas | ≥ 2.2 |

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User's Web Browser                    │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP (localhost:8501)
┌───────────────────────────▼─────────────────────────────┐
│                   Streamlit (app.py)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Navigation │  │   Upload &  │  │   Download &    │  │
│  │  Sidebar    │  │   Validate  │  │   Export        │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────┘  │
└─────────────────────────-┼──────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼──────────┐           ┌──────────▼─────────────┐
│   scanner.py        │           │  image_processing.py    │
│  ─────────────────  │           │  ─────────────────────  │
│  • Grayscale conv.  │           │  • Color enhancement    │
│  • Gaussian Blur    │           │  • Grayscale + CLAHE    │
│  • Canny edges      │           │  • B&W (Adapt. Thresh.) │
│  • Contour detect.  │           │  • Enhanced pipeline    │
│  • approxPolyDP     │           └────────────────────────┘
│  • Perspective warp │
└─────────────────────┘
          │
┌─────────▼──────────┐
│      utils.py       │
│  ─────────────────  │
│  • Validation       │
│  • PIL ↔ cv2        │
│  • Resize helpers   │
│  • Download buffer  │
└─────────────────────┘
```

---

## 7. Image Processing Workflow

```
Input Image (JPG/JPEG/PNG)
        │
        ▼
  Validation & Load (utils.py)
        │
        ▼
  Resize to ≤500 px height (for speed)
        │
        ▼
  Grayscale Conversion (cv2.cvtColor)
        │
        ▼
  Gaussian Blur — 5×5 kernel (cv2.GaussianBlur)
        │
        ▼
  Canny Edge Detection — thresholds 75, 200 (cv2.Canny)
        │
        ▼
  Find Contours (cv2.findContours)
        │
        ▼
  Sort by area → Approximate Polygon (cv2.approxPolyDP)
        │
     ┌──┴──┐
   4 pts    Not found
     │          │
     ▼          ▼
  Order      Use original
  corners    image as-is
     │
     ▼
  getPerspectiveTransform + warpPerspective
        │
        ▼
  Enhancement Mode (user selection)
  ┌─────┬──────┬──────┬────────┐
  │Color│Gray  │B&W   │Enhanced│
  └──┬──┴──┬───┴──┬───┴──┬─────┘
     ▼     ▼      ▼      ▼
  CLAHE  CLAHE  Adapt  Full
  +sharp +sharp Thresh Pipeline
        │
        ▼
  Download (PNG / JPEG)
```

---

## 8. Algorithms Used

### 8.1 Gaussian Blur — `cv2.GaussianBlur(image, (5,5), 0)`

The Gaussian filter convolves the image with a 2D Gaussian kernel:

```
G(x,y) = (1 / 2πσ²) · exp(-(x²+y²) / 2σ²)
```

It acts as a low-pass filter, suppressing high-frequency noise.
Using it before Canny prevents spurious edges from sensor noise.

---

### 8.2 Canny Edge Detection — `cv2.Canny(blur, 75, 200)`

Steps (John Canny, 1986):
1. **Gradient computation** — Sobel operators in X and Y directions.
2. **Non-maximum suppression** — Thin edges to 1-pixel width.
3. **Double thresholding** — Strong (>200), Weak (75–200), None (<75).
4. **Hysteresis** — Weak pixels connected to strong ones are kept.

---

### 8.3 Contour Detection — `cv2.findContours()`

A contour is a continuous curve of pixels along a boundary.
`RETR_LIST` returns all contours; `CHAIN_APPROX_SIMPLE` removes redundant collinear points.
Contours are sorted by area descending so the document (usually the largest object) is first.

---

### 8.4 Polygon Approximation — `cv2.approxPolyDP(contour, ε, closed=True)`

Douglas–Peucker recursive subdivision:
- If the maximum distance from any point to the line connecting the first and last point > ε,
  that point is kept and the algorithm recurses on both sub-curves.
- ε = 0.02 × perimeter → a quadrilateral document becomes exactly 4 vertices.

---

### 8.5 Perspective Transform

**Four-point ordering:** Top-left = min(x+y); Bottom-right = max(x+y);
Top-right = min(y−x); Bottom-left = max(y−x).

**Output size:** Width = max of bottom-edge length and top-edge length;
Height = max of left-edge length and right-edge length.

**Matrix computation:**
```
M = cv2.getPerspectiveTransform(src_pts, dst_pts)   # 3×3 homography
warped = cv2.warpPerspective(image, M, (W, H))       # apply warp
```

---

### 8.6 CLAHE — Contrast Limited Adaptive Histogram Equalization

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
result = clahe.apply(gray)
```

The image is divided into 8×8 tiles. Each tile's histogram is equalised independently.
A clip limit prevents excessive amplification of near-uniform regions (noise).
Tiles are blended with bilinear interpolation to avoid boundary artefacts.

---

### 8.7 Adaptive Thresholding — `cv2.adaptiveThreshold()`

```python
bw = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    blockSize=11, C=5
)
```

For each pixel, the threshold = Gaussian-weighted mean of an 11×11 neighbourhood − 5.
Works correctly even when lighting varies dramatically across the image.

---

### 8.8 Non-Local Means Denoising — `cv2.fastNlMeansDenoising()`

For every pixel, NLM searches for similar small patches in the image.
The denoised value = weighted average of matching-patch centres.
Text strokes (self-similar) are preserved; random noise averages to grey.
Parameter `h` controls the filter strength (8–10 is mild).

---

## 9. Installation Instructions

### Prerequisites

- Python 3.9 or later ([python.org](https://www.python.org/downloads/))
- pip (bundled with Python)

### Steps

```bash
# 1. Navigate to the project directory
cd SmartScan

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 10. How to Run

```bash
streamlit run app.py
```

The application opens automatically in your default browser at **http://localhost:8501**.

---

## 11. Project Structure

```
SmartScan/
│
├── app.py               ← Main Streamlit application and UI router
├── scanner.py           ← Document detection & perspective transformation
├── image_processing.py  ← Enhancement modes (Color, Grayscale, B&W, Enhanced)
├── utils.py             ← Validation, conversion, resizing, download helpers
├── requirements.txt     ← pip dependencies
├── README.md            ← This documentation
│
├── uploads/             ← (auto-created) temporary uploaded files
└── output/              ← (auto-created) exported scan files
```

---

## 12. Features

### Implemented

- [x] JPG / JPEG / PNG image upload with validation
- [x] File size and dimension checks
- [x] Automatic document boundary detection (Canny + contours)
- [x] Perspective correction (four-point homography warp)
- [x] Graceful fallback when detection fails
- [x] Four enhancement modes: Color / Grayscale / B&W / Enhanced
- [x] Before vs after comparison view
- [x] Intensity histogram visualisation
- [x] PNG and JPEG download with timestamped filenames
- [x] Professional blue & white Streamlit UI
- [x] In-app viva Q&A and algorithm documentation

---

## 13. Future Enhancements

| Enhancement | Description |
|---|---|
| 🔵 Manual corner selection | Let users drag corners when auto-detection fails |
| 🔵 OCR integration | Extract text from scanned documents (Tesseract) |
| 🔵 Batch processing | Process multiple documents simultaneously |
| 🔵 PDF export | Save scans directly as PDF files |
| 🔵 Webcam capture | Live webcam feed with real-time edge preview |
| 🔵 Deep learning detector | EAST or DocTR for more robust document detection |
| 🔵 Deskewing | Auto-detect and correct small rotations |
| 🔵 Watermark removal | Remove timestamps or logos from scans |

---

## 14. Viva Questions & Answers

**Q: What is the purpose of Gaussian Blur before Canny?**  
A: Gaussian Blur removes high-frequency noise. Canny detects intensity gradients, so without
blurring, sensor noise would produce thousands of false edges. Blurring ensures only major
transitions (like document boundaries) are detected.

**Q: Explain the four steps of Canny Edge Detection.**  
A: (1) Compute gradient magnitude and direction using Sobel operators. (2) Non-maximum suppression
thins edges to one pixel wide. (3) Double thresholding classifies pixels as strong (>high),
weak (between low and high), or non-edge (<low). (4) Hysteresis retains weak pixels only if
connected to a strong pixel.

**Q: Why is the contour sorted by area?**  
A: The document is typically the largest object in the frame. Sorting by area descending and
checking the first few contours for a 4-vertex approximation efficiently finds the document
without exhaustive search.

**Q: What is a homography matrix?**  
A: A 3×3 projective transformation matrix that maps points between two planes.
`getPerspectiveTransform` computes it from four source↔destination point pairs.
`warpPerspective` applies it to every pixel, correcting the camera's viewing angle.

**Q: What is CLAHE and how does it differ from equalizeHist?**  
A: `equalizeHist` redistributes pixel intensities using a single global cumulative histogram,
which can over-brighten uniform regions. CLAHE works tile by tile (adaptive), clips each
tile's histogram to avoid noise amplification, and blends tiles bilinearly. CLAHE produces
much better results on unevenly lit documents.

**Q: Why LAB colour space for colour enhancement?**  
A: LAB separates luminance (L) from chrominance (A for green-red, B for blue-yellow). By applying
CLAHE only to the L channel, we enhance contrast without altering the colour information.

**Q: What is Adaptive Thresholding and when is it preferred?**  
A: Global thresholding uses one fixed value for the whole image. Adaptive thresholding computes a
local threshold per pixel based on a neighbourhood mean (minus a constant C). It is preferred
when the document has uneven lighting, shadows, or varying background colour.

**Q: Describe Non-Local Means Denoising.**  
A: NLM replaces each pixel with a weighted average of pixels whose small surrounding patches
(templates) are similar to the target pixel's patch. Similar patches reinforce each other;
random noise has no matching patches and averages out. It preserves edges better than blur.

---

## 15. Conclusion

SmartScan successfully demonstrates how classical computer vision algorithms can solve a practical,
real-world problem without relying on neural networks or paid APIs.

The pipeline — from edge detection through perspective correction to histogram-based enhancement —
showcases the foundational techniques of Computer Vision and Image Processing:
Gaussian filtering, Canny edge detection, contour analysis, projective geometry, and
adaptive histogram operations.

The result is a production-quality document scanner built entirely on first principles,
demonstrating the power and practicality of traditional image processing.

---

*SmartScan | B.Tech CS | CV & IP Project | Built with Python + OpenCV + Streamlit*
