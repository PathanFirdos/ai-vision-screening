import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Step 3: Camera Activation — confirms a user is present.
    Step 4: Distance Verification — checks user is at correct distance (40-80 cm).
    """

    REAL_FACE_WIDTH_CM = 16.0
    FOCAL_LENGTH = 600
    MIN_DISTANCE_CM = 40.0
    MAX_DISTANCE_CM = 80.0

    def __init__(self, model_selection=0, min_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self._detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect(self, frame):
        """
        Detect all faces in a BGR frame.

        Returns:
            List of dicts: x, y, width, height, confidence,
                           distance_cm, distance_status
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces = []
        if not results.detections:
            return faces

        h, w = frame.shape[:2]

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x   = int(bbox.xmin * w)
            y   = int(bbox.ymin * h)
            bw  = int(bbox.width * w)
            bh  = int(bbox.height * h)

            distance = self._estimate_distance(bw)
            status   = self._distance_status(distance)

            faces.append({
                "x": x, "y": y,
                "width": bw, "height": bh,
                "confidence": round(float(det.score[0]), 3),
                "distance_cm": distance,
                "distance_status": status
            })

        return faces

    def _estimate_distance(self, face_width_pixels):
        if face_width_pixels <= 0:
            return None
        return round((self.REAL_FACE_WIDTH_CM * self.FOCAL_LENGTH) / face_width_pixels, 1)

    def _distance_status(self, distance_cm):
        if distance_cm is None:
            return "unknown"
        if distance_cm < self.MIN_DISTANCE_CM:
            return "too_close"
        if distance_cm > self.MAX_DISTANCE_CM:
            return "too_far"
        return "ok"

    def draw(self, frame, faces):
        """Draw bounding boxes + distance status on frame copy."""
        out = frame.copy()
        for face in faces:
            x, y, bw, bh = face["x"], face["y"], face["width"], face["height"]
            color = (0, 255, 0) if face["distance_status"] == "ok" else (0, 100, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = f"{face['distance_cm']} cm | {face['distance_status'].replace('_',' ').upper()}"
            cv2.putText(out, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return out

    def close(self):
        self._detector.close()


def run():
    """Standalone test: Steps 3 & 4 from PDF."""
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    print("[INFO] Press ESC to quit. Ideal distance: 40-80 cm.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        faces = detector.detect(frame)
        annotated = detector.draw(frame, faces)

        if faces:
            s = faces[0]["distance_status"]
            d = faces[0]["distance_cm"]
            if s == "too_close":
                msg, col = "Move BACK — too close", (0, 60, 255)
            elif s == "too_far":
                msg, col = "Move CLOSER — too far", (0, 140, 255)
            else:
                msg, col = f"Good position  ({d} cm)", (0, 220, 100)
        else:
            msg, col = "No face detected", (0, 60, 255)

        cv2.putText(annotated, msg, (20, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
        cv2.imshow("Vision Screening — Face Detection (Steps 3 & 4)", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run()