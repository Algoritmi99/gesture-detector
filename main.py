import yaml
import asyncio
import websockets
import pandas as pd

from gesture_detector.trainer import train_new
from gesture_detector.pose_detection import LiveFeedPoseDetector


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


df = pd.read_csv("data/zosia_csv_with_ground_truth_rotate.csv")
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

keypoint_names = read_keypoint_names()
pose_detector = LiveFeedPoseDetector("0", keypoint_names, show_feed=True)

model = train_new((X, y), pose_detector)
label = model.process()
print(label)

asyncio.run(send_command(label))
msg = "right"
asyncio.run(send_command(msg))
