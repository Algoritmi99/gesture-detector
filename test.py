import numpy as np
import yaml

from gesture_detector.feature_extraction import FeatureExtractor
from gesture_detector.pose_detection import LiveFeedPoseDetector
from gesture_detector.buffer import Buffer


def read_keypoint_names():
    with open("keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"]
        keypoint_names += mappings["body"]
    return keypoint_names

def show_cam_pose():
    keypoint_names = read_keypoint_names()
    posedetector = LiveFeedPoseDetector("2", keypoint_names, show_feed=True)
    feature_extractor = FeatureExtractor()

    while True:
        pose = posedetector.get_pose()
        feature_vector = feature_extractor.extract_features(pose)
        print(len(feature_vector))

def main():
    buffer  = Buffer(5)
    for i in range(100):
        buffer.add(np.array([i, i ** 2, i ** 3]))
        print(buffer.get_flatten())

if __name__ == '__main__':
    # main()
    show_cam_pose()
