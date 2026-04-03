import cv2
import numpy as np

from app.vision.distance_estimation import FaceDistanceEstimator
from app.vision.face_detection import FaceDetector
from app.vision.face_mesh import FaceMeshDetector
from app.vision.eye_tracking import EyeTracker
from app.vision.gaze_tracking import GazeTracker
from app.vision.pupil_detection import PupilDetector
from app.reporting.vision_report import VisionReportGenerator


class EyeAlignmentScreening:

    def __init__(self):

        self.face_detector = FaceDetector()
        self.mesh_detector = FaceMeshDetector()
        self.eye_tracker = EyeTracker()
        self.gaze_tracker = GazeTracker()
        self.pupil_detector = PupilDetector()
        self.distance_estimator = FaceDistanceEstimator()

    def compute_alignment(self, left_eye, right_eye, left_pupil, right_pupil):

        lh, lw = left_eye.shape[:2]
        rh, rw = right_eye.shape[:2]

        left_center = np.array([lw // 2, lh // 2])
        right_center = np.array([rw // 2, rh // 2])

        left_offset = np.array(left_pupil) - left_center
        right_offset = np.array(right_pupil) - right_center

        diff = np.linalg.norm(left_offset - right_offset)

        if diff < 10:
            return "aligned"
        else:
            return "possible_strabismus"

    def analyze_frame(self, frame):

        results = {
            "face_detected": False,
            "eyes_detected": False,
            "pupils_detected": False,
            "gaze_direction": "unknown",
            "alignment": "unknown",
            "distance_cm": None
        }

        # 1️⃣ Face detection
        faces = self.face_detector.detect(frame)

        if not faces:
            return results

        results["face_detected"] = True

        # 2️⃣ Face mesh
        mesh = self.mesh_detector.process(frame)

        if mesh is None:
            return results

        # 3️⃣ Distance estimation
        distance = self.distance_estimator.estimate(mesh, frame.shape)
        results["distance_cm"] = distance

        # 4️⃣ Eye extraction
        eyes = self.eye_tracker.get_eyes(frame, mesh)

        if eyes is None:
            return results

        results["eyes_detected"] = True

        left_eye, right_eye = eyes

        # 5️⃣ Pupil detection
        left_pupil = self.pupil_detector.detect(left_eye)
        right_pupil = self.pupil_detector.detect(right_eye)

        if left_pupil is None or right_pupil is None:
            return results

        results["pupils_detected"] = True

        # 6️⃣ Gaze estimation
        gaze = self.gaze_tracker.estimate(left_eye, right_eye, left_pupil, right_pupil)
        results["gaze_direction"] = gaze

        # 7️⃣ Alignment
        alignment = self.compute_alignment(
            left_eye,
            right_eye,
            left_pupil,
            right_pupil
        )

        results["alignment"] = alignment

        return results


def run_camera_screening():

    cap = cv2.VideoCapture(0)

    screening = EyeAlignmentScreening()
    reporter = VisionReportGenerator()

    final_result = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        result = screening.analyze_frame(frame)
        final_result = result

        text = f"Align:{result['alignment']} | Gaze:{result['gaze_direction']} | Dist:{result['distance_cm']}cm"

        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("AI Vision Screening", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate report after closing camera
    if final_result:
        report_file = reporter.generate(final_result)
        print("Report saved:", report_file)


if __name__ == "__main__":
    run_camera_screening()