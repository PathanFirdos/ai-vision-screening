import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices
LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]


class EyeTracker:
    """
    Step 5: Eye Tracking — extracts cropped eye images from a frame
    using FaceMesh landmarks. Also measures blink rate via EAR.
    """

    EAR_BLINK_THRESHOLD = 0.20

    def __init__(self):
        self._blink_count = 0
        self._eye_closed  = False

    def get_eyes(self, frame, landmarks):
        """
        Crop left and right eye regions from frame.

        Returns:
            (left_eye_img, right_eye_img) or None if extraction fails.
        """
        try:
            left  = self._crop_eye(frame, landmarks, LEFT_EYE_IDX)
            right = self._crop_eye(frame, landmarks, RIGHT_EYE_IDX)

            if left is None or right is None:
                return None
            if left.size == 0 or right.size == 0:
                return None

            return left, right

        except Exception:
            return None

    def update_blink(self, landmarks, frame_shape):
        """
        Call once per frame to update blink count using EAR formula.
        Returns current blink count.
        """
        ear = self._eye_aspect_ratio(landmarks, frame_shape)

        if ear is not None:
            if ear < self.EAR_BLINK_THRESHOLD:
                if not self._eye_closed:
                    self._blink_count += 1
                    self._eye_closed = True
            else:
                self._eye_closed = False

        return self._blink_count

    def reset_blink_count(self):
        self._blink_count = 0
        self._eye_closed  = False

    def _crop_eye(self, frame, landmarks, indices):
        h, w = frame.shape[:2]
        points = np.array([
            [int(landmarks[i].x * w), int(landmarks[i].y * h)]
            for i in indices
        ])
        bx, by, bw, bh = cv2.boundingRect(points)

        # 15% padding
        px, py = int(bw * 0.15), int(bh * 0.15)
        x1 = max(0, bx - px);      y1 = max(0, by - py)
        x2 = min(w, bx + bw + px); y2 = min(h, by + bh + py)

        region = frame[y1:y2, x1:x2]
        return region if region.size > 0 else None

    def _eye_aspect_ratio(self, landmarks, frame_shape):
        """
        Eye Aspect Ratio (EAR) for blink detection.
        EAR = (vertical distances) / (2 * horizontal distance)
        """
        try:
            h, w = frame_shape[:2]

            def pt(idx):
                return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

            # Left eye
            A = np.linalg.norm(pt(159) - pt(145))
            B = np.linalg.norm(pt(158) - pt(153))
            C = np.linalg.norm(pt(133) - pt(33))
            ear_left = (A + B) / (2.0 * C) if C > 0 else 0

            # Right eye
            D = np.linalg.norm(pt(386) - pt(374))
            E = np.linalg.norm(pt(385) - pt(380))
            F = np.linalg.norm(pt(263) - pt(362))
            ear_right = (D + E) / (2.0 * F) if F > 0 else 0

            return (ear_left + ear_right) / 2.0

        except Exception:
            return None