import cv2
import numpy as np


class PupilDetector:
    """Detect and draw pupils in cropped eye images."""

    def detect(self, eye_frame):
        return detect_pupil(eye_frame)

    def draw(self, eye_frame, pupil_data):
        return draw_pupil(eye_frame, pupil_data)


def detect_pupil(eye_frame):
    """
    Detect pupil center and radius from a cropped eye BGR image.
    Uses adaptive thresholding so it works with glasses, lighting
    changes, and dark frames.

    Returns:
        ((cx, cy), radius) or None.
    """
    if eye_frame is None or eye_frame.size == 0:
        return None

    # Resize small eye crops for better detection
    h, w = eye_frame.shape[:2]
    if w < 20 or h < 20:
        return None

    scale = 1
    if w < 60:
        scale = 3
        eye_frame = cv2.resize(eye_frame, (w * scale, h * scale),
                               interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    # Equalise histogram to handle glasses / lighting variation
    gray = cv2.equalizeHist(gray)

    # Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Method 1: fixed dark-region threshold ---
    _, thresh_fixed = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # --- Method 2: adaptive threshold (works better with glasses) ---
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )

    # Combine both masks — use whichever gives better contours
    combined = cv2.bitwise_or(thresh_fixed, thresh_adapt)

    # Morphological close to fill gaps caused by glasses frames
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    result = _best_pupil_contour(combined, scale)

    # Fallback to fixed threshold alone if combined fails
    if result is None:
        result = _best_pupil_contour(thresh_fixed, scale)

    return result


def _best_pupil_contour(thresh, scale):
    """Find the best circular contour and return ((cx,cy), radius)."""
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Filter noise and extreme sizes
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > thresh.shape[0] * thresh.shape[1] * 0.4:
            continue
        # Prefer roughly circular contours
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.2:   # 1.0 = perfect circle
            valid.append((c, circularity))

    if not valid:
        return None

    # Pick the most circular contour
    best_contour = max(valid, key=lambda x: x[1])[0]

    (x, y), radius = cv2.minEnclosingCircle(best_contour)

    # Scale coordinates back to original size
    center = (int(x / scale), int(y / scale))
    radius = max(1, int(radius / scale))

    return center, radius


def draw_pupil(eye_frame, pupil_data):
    """Draw pupil circle and center dot. Returns modified copy."""
    if pupil_data is None:
        return eye_frame

    out = eye_frame.copy()
    center, radius = pupil_data

    cv2.circle(out, center, radius, (0, 255, 0), 2)   # green outline
    cv2.circle(out, center, 2,      (0, 0, 255), -1)  # red center dot

    return out