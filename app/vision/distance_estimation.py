import numpy as np


class FaceDistanceEstimator:

    # Average human interpupillary distance (cm)
    REAL_EYE_DISTANCE = 6.3

    # Camera focal length (approx, tune later)
    FOCAL_LENGTH = 700

    LEFT_EYE = 33
    RIGHT_EYE = 263

    def estimate(self, landmarks, frame_shape):

        h, w = frame_shape[:2]

        left = landmarks[self.LEFT_EYE]
        right = landmarks[self.RIGHT_EYE]

        x1 = int(left.x * w)
        y1 = int(left.y * h)

        x2 = int(right.x * w)
        y2 = int(right.y * h)

        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if pixel_distance == 0:
            return None

        distance = (self.REAL_EYE_DISTANCE * self.FOCAL_LENGTH) / pixel_distance

        return round(distance, 1)