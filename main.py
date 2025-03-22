import argparse
import time

import pandas as pd
import websockets
import yaml
from tqdm import tqdm

from gesture_detector.pose_detection import LiveFeedPoseDetector
from gesture_detector.trainer import train_new
from gesture_detector.pipeline import load_pipeline


async def send_command(command):
    uri = "ws://localhost:8000/events"
    async with websockets.connect(uri) as websocket:
        await websocket.send(command)
        print(f"Sent command: {command}")


def read_keypoint_names():
    with open("keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"]
        keypoint_names += mappings["body"]
    return keypoint_names

class Pose:
    def __init__(self, x, y, z, confidence):
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence


def train_and_save():
    keypoint_names = read_keypoint_names()

    df = pd.read_csv("data/combined_output.csv")
    y = df.iloc[:, [-1]]
    x_raw = df.iloc[:, :-1]
    print("Read dataset with length", len(y))

    print("Preprocessing data...")
    x = pd.DataFrame(columns=keypoint_names + ["timestamp"])
    for idx, row in tqdm(x_raw.iterrows()):
        next_line = []
        for pose_name in keypoint_names:
            next_line.append(
                Pose(
                    float(row[pose_name + "_x"]),
                    float(row[pose_name + "_y"]),
                    float(row[pose_name + "_z"]),
                    float(row[pose_name + "_confidence"]),
                )
            )
        next_line.append(int(row["timestamp"]))
        x.loc[len(x)] = next_line


    print("Running Model Trainer...")
    pipeline = train_new((x, y), None, 700, 0.4)
    pipeline.save("./pipelines", "gesture_detector700_0.4")


def run_app(path):
    pipeline = load_pipeline(path)
    pipeline.set_pose_detector(LiveFeedPoseDetector("0", read_keypoint_names(), show_feed=True))
    while True:
        gesture = pipeline.process()
        print(gesture)
        if gesture != "idle" and gesture is not None:
            send_command(gesture)
            # time.sleep(2.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose a mode to run")
    subparsers.add_parser("train", help="train a new pipeline and save it (no extra argument needed)")

    mode2_parser = subparsers.add_parser("run", help="Run a given pipeline (requires an argument)")
    mode2_parser.add_argument("path", type=str, help="path to the pipeline to run")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save()
    elif args.mode == "run":
        run_app(args.path)
