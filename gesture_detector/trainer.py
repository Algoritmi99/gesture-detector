from collections import Counter

import light
import pandas as pd
import numpy as np
from light.trainer import Trainer

from gesture_detector.buffer import Buffer
from gesture_detector.classifier import FFNClassifier
from gesture_detector.feature_extraction import FeatureExtractor
from gesture_detector.pipeline import GestureDetectorPipeline
from gesture_detector.pose_detection.base import PoseDetector


def train_new(dataset: tuple[pd.DataFrame, pd.DataFrame], pose_detector: PoseDetector) -> GestureDetectorPipeline:
    """
    Build a new GestureDetectorPipeline given a tuple of (pd.DataFrame: raw features, pd.DataFrame: gesture labels)
    :param pose_detector: instance of PoseDetector to be used in the GestureDetectorPipeline
    :param dataset: tuple of (pd.DataFrame: raw features, pd.DataFrame: gesture labels)
    :return: trained GestureDetectorPipeline
    """
    x, y = dataset[0], dataset[1]
    assert len(x) == len(y)

    feature_extractor = FeatureExtractor()
    features = pd.DataFrame(x.apply(lambda row: feature_extractor.extract_features(row.to_dict()), axis=1).tolist())

    pca = light.PCA(variance_threshold=0.99)
    # reduced_features = pd.DataFrame(pca.fit_transform(features.to_numpy()))
    reduced_features = pca.fit_transform(features.to_numpy())
    reduced_features = np.real(reduced_features)
    # reduced_features = np.nan_to_num(reduced_features)
    reduced_features = pd.DataFrame(reduced_features)
    assert len(reduced_features) == len(y)

    # one_hot_encoder = light.OneHotEncoder(y.iloc[:, 0].unique())
    one_hot_encoder = light.OneHotEncoder(y.unique())

    buffer_size = 100 // pca.n_components
    buffer_size = int(buffer_size)

    buffer_x = Buffer(buffer_size)
    buffer_y = Buffer(buffer_size)

    # nn_dataset_x = pd.DataFrame()
    # nn_dataset_y = pd.DataFrame()
    num_features = reduced_features.shape[1]
    nn_dataset_x = pd.DataFrame(columns=[f'feature_{i}' for i in range(num_features)])
    num_classes = len(y.unique())
    nn_dataset_y = pd.DataFrame(columns=[f'class_{i}' for i in range(num_classes)])

    for idx in range(len(reduced_features)):
        buffer_x.add(reduced_features.iloc[idx].to_numpy())
        buffer_y.add(y.iloc[idx])
        # buffer_y.add(y.iloc[idx].numpy())

        next_x = buffer_x.get_flatten()
        next_y = buffer_y.get_flatten()

        if next_x is not None and next_y is not None:
            label = Counter(next_y.tolist()).most_common(1)[0][0]
            next_y = one_hot_encoder.encode(label)

            # nn_dataset_x.loc[len(nn_dataset_x)] = next_x
            nn_dataset_x = pd.concat([nn_dataset_x, pd.DataFrame([next_x])], ignore_index=True)
            # nn_dataset_y.loc[len(nn_dataset_y)] = next_y
            nn_dataset_y = pd.concat([nn_dataset_y, pd.DataFrame([next_y], columns=nn_dataset_y.columns)],
                                     ignore_index=True)

    X_train, X_test, y_train, y_test = light.train_test_split(nn_dataset_x, nn_dataset_y, 0.8)

    # Train the NN
    net = FFNClassifier(
        X_train.shape[1],
        int(X_train.shape[1] * 0.2),
        # len(y.iloc[:, 0].unique())
        len(y.unique())
    )

    optimizer = light.SGD(net, light.CrossEntropyLoss(), 0.01)

    trainer = Trainer(optimizer, plot=True)
    X_train = X_train.fillna(float(0))

    net = trainer.train((X_train, y_train), 10, progress_bar=True)
    assert isinstance(net, FFNClassifier)

    # Accuracy on unseen data
    correct = 0
    false = 0
    for idx in range(len(X_test)):
        # X_test = X_test.fillna(float(0))
        # print(X_test.head())
        pred_label = one_hot_encoder.decode(X_test.iloc[idx].to_numpy())
        # if pred_label == one_hot_encoder.decode(y_test.iloc[idx].to_numpy()):
        if pred_label == one_hot_encoder.decode(y_test.to_numpy()):
            correct += 1
        else:
            false += 1
    print("Accuracy: {:.2f}%".format(correct * 100 / len(X_test)))

    out = GestureDetectorPipeline(
        pose_detector,
        Buffer(buffer_size),
        pca,
        # y.iloc[:, 0].unique().tolist(),
        y.unique().tolist(),
        net
    )

    return out
