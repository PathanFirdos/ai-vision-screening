import cv2
import numpy as np

LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]


def get_eye_region(frame, landmarks, indices, padding=0.3):
    """
    Crop an eye region from the frame using landmark indices.
    padding: fraction of eye size to add around the crop (bigger = safer)
    """
    h, w = frame.shape[:2]

    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        # Clamp to frame bounds
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        points.append([x, y])

    points = np.array(points)

    bx, by, bw, bh = cv2.boundingRect(points)

    # Safety check — bounding box must be valid
    if bw <= 0 or bh <= 0:
        return None

    # Add padding
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    # Final safety check
    if x2 <= x1 or y2 <= y1:
        return None

    eye = frame[y1:y2, x1:x2]

    if eye is None or eye.size == 0:
        return None

    return eye


def extract_eyes(frame, face_landmarks):
    """
    Extract left and right eye crops from a frame.

    Args:
        frame:          BGR image
        face_landmarks: MediaPipe face_landmarks object
                        (results.multi_face_landmarks[i])

    Returns:
        (left_eye, right_eye) — both BGR crops
        Returns (None, None) if extraction fails.
    """
    landmarks = face_landmarks.landmark

    left_eye  = get_eye_region(frame, landmarks, LEFT_EYE_IDX)
    right_eye = get_eye_region(frame, landmarks, RIGHT_EYE_IDX)

    if left_eye is None or right_eye is None:
        return None, None

    return left_eye, right_eye