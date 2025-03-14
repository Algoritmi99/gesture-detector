import cv2

from gesture_detector.pose_detection.base import PoseDetector


class OffFeedPoseDetector(PoseDetector):
    def __init__(self, source: str, keyPoint_names: list):
        super().__init__(source, keyPoint_names, cv2.VideoCapture(source))
