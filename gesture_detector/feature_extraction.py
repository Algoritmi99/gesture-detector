import copy
from itertools import combinations

import numpy as np

important_poses = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
]

pose_pairs = list(combinations(important_poses, 2))


class FeatureExtractor:
    def __init__(self):
        self.last_time = None
        self.last_vector = None

    def extract_features(self, raw_landmarks: dict) -> np.ndarray:
        """
        extract features from raw landmarks
        :param raw_landmarks: dictionary that names the landmarks.
        :return: extracted features.
        """
        if raw_landmarks is None:
            return None

        # consistency assertion
        if self.last_vector is None or self.last_time is None:
            assert self.last_vector is None and self.last_time is None

        out = []
        xs, ys = [], []
        # - Per joint:
        #       raw_pos_x, raw_pos_y, velocity, acceleration
        for pose_name in important_poses:
            if 'left_shoulder_x' not in raw_landmarks.keys():
                out.append(float(raw_landmarks[pose_name].x))
                out.append(float(raw_landmarks[pose_name].y))
                xs.append(float(raw_landmarks[pose_name].x))
                ys.append(float(raw_landmarks[pose_name].y))
            else:
                out.append(float(raw_landmarks[f"{pose_name}_x"]))
                out.append(float(raw_landmarks[f"{pose_name}_y"]))
                xs.append(float(raw_landmarks[f"{pose_name}_x"]))
                ys.append(float(raw_landmarks[f"{pose_name}_y"]))

            # Add velocity and acceleration if this is not the first vector being made, otherwise 0
            if self.last_time is None and self.last_vector is None:
                out.extend([0] * 6)
            else:
                # Velocity
                if 'left_shoulder_x' not in raw_landmarks.keys():
                    x = raw_landmarks[pose_name].x
                    y = raw_landmarks[pose_name].y
                else:
                    x = raw_landmarks[f"{pose_name}_x"]
                    y = raw_landmarks[f"{pose_name}_y"]
                lastX = self.last_vector[len(out) - 2]
                lastY = self.last_vector[len(out) - 1]

                denominator = raw_landmarks["timestamp"] - self.last_time
                if denominator == 0:
                    denominator = 10e6

                out.append((x - lastX) / denominator)
                out.append((y - lastY) / denominator)
                out.append(
                    float(np.sqrt(
                        ((x - lastX) ** 2) +
                        ((y - lastY) ** 2)
                    ) / denominator
                ))

                # Acceleration
                v_x = out[len(out) - 3]
                v_y = out[len(out) - 2]
                v_d = out[len(out) - 1]
                last_v_x = self.last_vector[len(out) - 3]
                last_v_y = self.last_vector[len(out) - 2]
                last_v_d = self.last_vector[len(out) - 1]
                out.append((v_x - last_v_x) / denominator)
                out.append((v_y - last_v_y) / denominator)
                out.append(
                    (v_d - last_v_d) / denominator
                )

        # - Per joint pair:
        #       Distance between joints, Angle between joints, change in angles
        for j1, j2 in pose_pairs:
            # distance between joints
            if 'left_shoulder_x' not in raw_landmarks.keys():
                out.append(
                    float(np.sqrt(

                        ((raw_landmarks[j1].x - raw_landmarks[j2].x) ** 2) +
                        ((raw_landmarks[j1].y - raw_landmarks[j2].y) ** 2)
                        # ((raw_landmarks[f"{j1}_x"] - raw_landmarks[f"{j2}_x"]) ** 2) +
                        # ((raw_landmarks[f"{j1}_y"] - raw_landmarks[f"{j2}_y"]) ** 2)
                    )
                    ))
            else:
                out.append(
                    float(np.sqrt(

                        # ((raw_landmarks[j1].x - raw_landmarks[j2].x) ** 2) +
                        # ((raw_landmarks[j1].y - raw_landmarks[j2].y) ** 2)
                        ((raw_landmarks[f"{j1}_x"] - raw_landmarks[f"{j2}_x"]) ** 2) +
                        ((raw_landmarks[f"{j1}_y"] - raw_landmarks[f"{j2}_y"]) ** 2)
                    )
                ))
            out.append(out[len(out) - 1] - self.last_vector[len(out) - 1] if self.last_vector is not None else 0)
            # Angle between joints
        if 'left_shoulder_x' not in raw_landmarks.keys():
            out.append(
                float(np.arccos(
                    ((raw_landmarks[j1].x * raw_landmarks[j2].x) + (raw_landmarks[j1].y * raw_landmarks[j2].y)) /
                    # ((raw_landmarks[f"{j1}_x"] * raw_landmarks[f"{j2}_x"]) + (
                    #             raw_landmarks[f"{j1}_y"] * raw_landmarks[f"{j2}_y"])) /
                    (
                            float(np.sqrt(
                                (raw_landmarks[j1].x ** 2) + (raw_landmarks[j1].y ** 2)
                                # (raw_landmarks[f"{j1}_x"] ** 2) + (raw_landmarks[f"{j1}_y"] ** 2)
                            ))
                            *
                            float(np.sqrt(
                                (raw_landmarks[j2].x ** 2) + (raw_landmarks[j2].y ** 2)
                                # (raw_landmarks[f"{j2}_x"] ** 2) + (raw_landmarks[f"{j2}_y"] ** 2)
                            ))
                    )
                )
                ))
        else:
            out.append(
                float(np.arccos(
                    # ((raw_landmarks[j1].x * raw_landmarks[j2].x) + (raw_landmarks[j1].y * raw_landmarks[j2].y)) /
                    ((raw_landmarks[f"{j1}_x"] * raw_landmarks[f"{j2}_x"]) + (raw_landmarks[f"{j1}_y"] * raw_landmarks[f"{j2}_y"])) /
                    (
                        float(np.sqrt(
                            # (raw_landmarks[j1].x ** 2) + (raw_landmarks[j1].y ** 2)
                            (raw_landmarks[f"{j1}_x"] ** 2) + (raw_landmarks[f"{j1}_y"] ** 2)
                        ))
                        *
                        float(np.sqrt(
                            # (raw_landmarks[j2].x ** 2) + (raw_landmarks[j2].y ** 2)
                            (raw_landmarks[f"{j2}_x"] ** 2) + (raw_landmarks[f"{j2}_y"] ** 2)
                        ))
                     )
                )
            ))
        out.append((out[len(out) - 1] - self.last_vector[len(out) - 1]) if self.last_vector is not None else 0)


        # - For all joints together:
        #       Mean pos
        out.append(sum(xs) / len(xs))
        out.append(sum(ys) / len(ys))

        # consistency assertion
        if self.last_vector is not None:
            assert len(out) == len(self.last_vector)

        # save output and last time
        self.last_vector = copy.deepcopy(out)
        self.last_time = raw_landmarks["timestamp"]

        return np.array(out)
