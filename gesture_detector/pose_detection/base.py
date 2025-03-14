import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, source: str, keyPoint_names: list, cap: cv2.VideoCapture, show_feed=False):
        self.source = source
        self.keyPoint_names = keyPoint_names
        self.cap = cap
        self.show_feed = show_feed
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_pose(self):
        if self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                return None

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            assert hasattr(results, "pose_landmarks")

            if self.show_feed:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                cv2.imshow('MediaPipe Pose', image)
                cv2.waitKey(1)

            if results.pose_landmarks is None:
                return None

            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            results = results.pose_landmarks.landmark
            out = {}
            for i, landmark in enumerate(results):
                out[self.keyPoint_names[i]] = landmark
                out["timestamp"] = timestamp
            return out

    def release_cap(self):
        self.cap.release()

    def __del__(self):
        self.release_cap()
