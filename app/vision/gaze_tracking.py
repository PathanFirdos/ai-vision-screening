import numpy as np


class GazeTracker:
    """
    Step 5: Gaze direction estimation.
    Classifies gaze as: center / left / right / up / down / unknown.

    Uses normalized pupil offset ratios so results are
    distance-independent.
    """

    HORIZONTAL_THRESHOLD = 0.20   # 20% of half eye-width
    VERTICAL_THRESHOLD   = 0.15

    def estimate(self, left_eye, right_eye, left_pupil, right_pupil):
        """
        Args:
            left_eye / right_eye:     Cropped BGR eye images
            left_pupil / right_pupil: ((cx,cy), radius) from pupil_detection

        Returns:
            str: "center" | "left" | "right" | "up" | "down" | "unknown"
        """
        try:
            lh, lv = self._ratio(left_eye,  left_pupil)
            rh, rv = self._ratio(right_eye, right_pupil)

            avg_h = (lh + rh) / 2
            avg_v = (lv + rv) / 2

            if abs(avg_h) > self.HORIZONTAL_THRESHOLD or \
               abs(avg_v) > self.VERTICAL_THRESHOLD:
                if abs(avg_h) >= abs(avg_v):
                    return "right" if avg_h > 0 else "left"
                else:
                    return "down" if avg_v > 0 else "up"

            return "center"

        except Exception:
            return "unknown"

    def _ratio(self, eye_img, pupil_data):
        """
        Normalized horizontal + vertical offset for one eye.
        Returns (ratio_h, ratio_v) in range [-1, 1].
        """
        if pupil_data is None:
            return 0.0, 0.0

        (px, py), _ = pupil_data
        h, w = eye_img.shape[:2]

        if w == 0 or h == 0:
            return 0.0, 0.0

        ratio_h = (px - w / 2) / (w / 2)
        ratio_v = (py - h / 2) / (h / 2)

        return ratio_h, ratio_v