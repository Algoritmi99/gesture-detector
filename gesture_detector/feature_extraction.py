import copy
from itertools import combinations
import numpy as np

# List of important body landmarks
important_poses = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb"
]

pose_pairs = list(combinations(important_poses, 2))

# Small constant to prevent division by zero
EPSILON = 1e-6

class FeatureExtractor:
    def __init__(self):
        self.last_time = None
        self.last_vector = None

    def extract_features(self, raw_landmarks: dict) -> np.ndarray:
        """
        Extracts numerical features from landmark positions.
        :param raw_landmarks: Dictionary containing landmark coordinates and timestamp.
        :return: Feature vector as a NumPy array.
        """
        if raw_landmarks is None:
            return None

        if self.last_vector is None or self.last_time is None:
            assert self.last_vector is None and self.last_time is None

        out = []
        xs, ys = [], []

        # Iterate through each joint
        for pose_name in important_poses:
            x = float(raw_landmarks[pose_name].x)
            y = float(raw_landmarks[pose_name].y)
            xs.append(x)
            ys.append(y)

            out.append(x)
            out.append(y)

            # Compute velocity and acceleration
            if self.last_time is None or self.last_vector is None:
                out.extend([0] * 6)  # If first frame, set velocity & acceleration to 0
            else:
                lastX = self.last_vector[len(out) - 2]
                lastY = self.last_vector[len(out) - 1]
                delta_t = max(raw_landmarks["timestamp"] - self.last_time, EPSILON)

                # Velocity
                v_x = (x - lastX) / delta_t
                v_y = (y - lastY) / delta_t
                v_d = float(np.sqrt((x - lastX) ** 2 + (y - lastY) ** 2) / delta_t)

                out.extend([v_x, v_y, v_d])

                # Acceleration
                last_v_x = self.last_vector[len(out) - 3]
                last_v_y = self.last_vector[len(out) - 2]
                last_v_d = self.last_vector[len(out) - 1]

                a_x = (v_x - last_v_x) / delta_t
                a_y = (v_y - last_v_y) / delta_t
                a_d = (v_d - last_v_d) / delta_t

                out.extend([a_x, a_y, a_d])

        # Compute pairwise distances and angles
        for j1, j2 in pose_pairs:
            x1, y1 = raw_landmarks[j1].x, raw_landmarks[j1].y
            x2, y2 = raw_landmarks[j2].x, raw_landmarks[j2].y

            # Distance between joints
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            last_dist = self.last_vector[len(out) - 1] if self.last_vector is not None else 0
            out.append(dist)
            out.append(dist - last_dist)

            # Angle calculation (prevent invalid input to arccos)
            dot_product = (x1 * x2) + (y1 * y2)
            norm1 = np.sqrt(x1**2 + y1**2)
            norm2 = np.sqrt(x2**2 + y2**2)
            angle = np.arccos(np.clip(dot_product / (norm1 * norm2 + EPSILON), -1.0, 1.0))

            last_angle = self.last_vector[len(out) - 1] if self.last_vector is not None else 0
            out.append(angle)
            out.append(angle - last_angle)

        # Compute mean position
        mean_x = np.nan_to_num(np.mean(xs))
        mean_y = np.nan_to_num(np.mean(ys))
        out.extend([mean_x, mean_y])

        # Ensure vector length consistency
        if self.last_vector is not None:
            assert len(out) == len(self.last_vector)

        # Store the latest values
        self.last_vector = copy.deepcopy(out)
        self.last_time = raw_landmarks["timestamp"]

        return np.array(out)

# import copy
# from itertools import combinations
#
# import numpy as np
#
# important_poses = [
#     "left_shoulder",
#     "right_shoulder",
#     "left_elbow",
#     "right_elbow",
#     "left_wrist",
#     "right_wrist",
#     "left_pinky",
#     "right_pinky",
#     "left_index",
#     "right_index",
#     "left_thumb",
#     "right_thumb",
# ]
#
# pose_pairs = list(combinations(important_poses, 2))
#
#
# class FeatureExtractor:
#     def __init__(self):
#         self.last_time = None
#         self.last_vector = None
#
#     def extract_features(self, raw_landmarks: dict) -> np.ndarray:
#         """
#         extract features from raw landmarks
#         :param raw_landmarks: dictionary that names the landmarks.
#         :return: extracted features.
#         """
#         if raw_landmarks is None:
#             return None
#
#         # consistency assertion
#         if self.last_vector is None or self.last_time is None:
#             assert self.last_vector is None and self.last_time is None
#
#         out = []
#         xs, ys = [], []
#         # - Per joint:
#         #       raw_pos_x, raw_pos_y, velocity, acceleration
#         for pose_name in important_poses:
#             out.append(float(raw_landmarks[pose_name].x))
#             out.append(float(raw_landmarks[pose_name].y))
#             xs.append(float(raw_landmarks[pose_name].x))
#             ys.append(float(raw_landmarks[pose_name].y))
#
#             # Add velocity and acceleration if this is not the first vector being made, otherwise 0
#             if self.last_time is None and self.last_vector is None:
#                 out.extend([0] * 6)
#             else:
#                 # Velocity
#                 x = raw_landmarks[pose_name].x
#                 y = raw_landmarks[pose_name].y
#                 lastX = self.last_vector[len(out) - 2]
#                 lastY = self.last_vector[len(out) - 1]
#                 out.append((x - lastX) / (raw_landmarks["timestamp"] - self.last_time))
#                 out.append((y - lastY) / (raw_landmarks["timestamp"] - self.last_time))
#                 out.append(
#                     float(np.sqrt(
#                         ((x - lastX) ** 2) +
#                         ((y - lastY) ** 2)
#                     ) / (raw_landmarks["timestamp"] - self.last_time)
#                 ))
#
#                 # Acceleration
#                 v_x = out[len(out) - 3]
#                 v_y = out[len(out) - 2]
#                 v_d = out[len(out) - 1]
#                 last_v_x = self.last_vector[len(out) - 3]
#                 last_v_y = self.last_vector[len(out) - 2]
#                 last_v_d = self.last_vector[len(out) - 1]
#                 out.append((v_x - last_v_x) / (raw_landmarks["timestamp"] - self.last_time))
#                 out.append((v_y - last_v_y) / (raw_landmarks["timestamp"] - self.last_time))
#                 out.append(
#                     (v_d - last_v_d) / (raw_landmarks["timestamp"] - self.last_time)
#                 )
#
#         # - Per joint pair:
#         #       Distance between joints, Angle between joints, change in angles
#         for j1, j2 in pose_pairs:
#             # distance between joints
#             out.append(
#                 float(np.sqrt(
#                     ((raw_landmarks[j1].x - raw_landmarks[j2].x) ** 2) +
#                     ((raw_landmarks[j1].y - raw_landmarks[j2].y) ** 2)
#                 )
#             ))
#             out.append(out[len(out) - 1] - self.last_vector[len(out) - 1] if self.last_vector is not None else 0)
#             # Angle between joints
#             out.append(
#                 float(np.arccos(
#                     ((raw_landmarks[j1].x * raw_landmarks[j2].x) + (raw_landmarks[j1].y * raw_landmarks[j2].y)) /
#                     (
#                         float(np.sqrt(
#                             (raw_landmarks[j1].x ** 2) + (raw_landmarks[j1].y ** 2)
#                         ))
#                         *
#                         float(np.sqrt(
#                             (raw_landmarks[j2].x ** 2) + (raw_landmarks[j2].y ** 2)
#                         ))
#                      )
#                 )
#             ))
#             out.append((out[len(out) - 1] - self.last_vector[len(out) - 1]) if self.last_vector is not None else 0)
#
#
#         # - For all joints together:
#         #       Mean pos
#         out.append(sum(xs) / len(xs))
#         out.append(sum(ys) / len(ys))
#
#         # consistency assertion
#         if self.last_vector is not None:
#             assert len(out) == len(self.last_vector)
#
#         # save output and last time
#         self.last_vector = copy.deepcopy(out)
#         self.last_time = raw_landmarks["timestamp"]
#
#         return np.array(out)
