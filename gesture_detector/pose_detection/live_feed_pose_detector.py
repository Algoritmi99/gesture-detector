import cv2

from gesture_detector.pose_detection.base import PoseDetector


class LiveFeedPoseDetector(PoseDetector):
    def __init__(self, source, keyPoint_names: list, show_feed=False):
        super().__init__(source, keyPoint_names, cv2.VideoCapture(index=int(source)), show_feed=show_feed)
