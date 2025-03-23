import numpy as np
import pandas as pd
import yaml

from gesture_detector.feature_extraction import FeatureExtractor
from gesture_detector.pose_detection import LiveFeedPoseDetector
from gesture_detector.buffer import Buffer


def read_keypoint_names():
    with open("./keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"]
        keypoint_names += mappings["body"]
    return keypoint_names

def show_cam_pose():
    keypoint_names = read_keypoint_names()
    posedetector = LiveFeedPoseDetector("0", keypoint_names, show_feed=True)
    feature_extractor = FeatureExtractor()

    while True:
        pose = posedetector.get_pose()
        feature_vector = feature_extractor.extract_features(pose)
        print(len(feature_vector) if feature_vector is not None else None)

def main():
    buffer  = Buffer(150//29)
    for i in range(100):
        buffer.add(i)
        print(buffer.get_flatten())

if __name__ == '__main__':
    # print(type(150 // 29))
    main()
    # show_cam_pose()
    # df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(df)
