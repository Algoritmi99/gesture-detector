from collections import Counter

import light
import pandas as pd
from light.trainer import Trainer
from tqdm import tqdm

from gesture_detector.buffer import Buffer
from gesture_detector.classifier import FFNClassifier
from gesture_detector.feature_extraction import FeatureExtractor
from gesture_detector.pipeline import GestureDetectorPipeline
from gesture_detector.pose_detection.base import PoseDetector


def train_new(dataset: tuple[pd.DataFrame, pd.DataFrame], pose_detector: PoseDetector, model_input_length: int, model_hidden_ratio: float) -> GestureDetectorPipeline:
    """
    Build a new GestureDetectorPipeline given a tuple of (pd.DataFrame: raw features, pd.DataFrame: gesture labels)
    :param model_hidden_ratio: The ratio of hidden layers to input layer
    :param model_input_length: The length of input layer
    :param pose_detector: instance of PoseDetector to be used in the GestureDetectorPipeline
    :param dataset: tuple of (pd.DataFrame: raw features, pd.DataFrame: gesture labels)
    :return: trained GestureDetectorPipeline
    """
    # Build the datasets
    x, y = dataset[0], dataset[1]
    assert len(x) == len(y)

    print("Running feature extractor on data...")
    feature_extractor = FeatureExtractor()
    features = pd.DataFrame(x.apply(lambda row: feature_extractor.extract_features(row.to_dict()), axis=1).tolist())

    print("Running PCA on data...")
    pca = light.PCA(variance_threshold=0.99)
    reduced_features = pd.DataFrame(pca.fit_transform(features.to_numpy()))

    assert len(reduced_features) == len(y)

    one_hot_encoder = light.OneHotEncoder(y.iloc[:, 0].unique())

    buffer_size = model_input_length // pca.n_components
    buffer_x = Buffer(buffer_size)
    buffer_y = Buffer(buffer_size)

    nn_dataset_x = pd.DataFrame()
    nn_dataset_y = pd.DataFrame()

    print("Running buffering on data...")
    for idx in tqdm(range(len(reduced_features))):
        buffer_x.add(reduced_features.iloc[idx].to_numpy())
        buffer_y.add(y.iloc[idx].to_numpy())

        next_x = buffer_x.get_flatten()
        next_y = buffer_y.get_flatten()

        if next_x is not None and next_y is not None:
            label = Counter(next_y.tolist()).most_common(1)[0][0]
            next_y = one_hot_encoder.encode(label)

            nn_dataset_x.loc[len(nn_dataset_x)] = next_x
            nn_dataset_y.loc[len(nn_dataset_y)] = next_y

    X_train, X_test, y_train, y_test = light.train_test_split(nn_dataset_x, nn_dataset_y, 0.8)

    # Train the NN
    net = FFNClassifier(
        X_train.shape[1],
        int(X_train.shape[1] * model_hidden_ratio),
        len(y.iloc[:, 0].unique())
    )

    optimizer = light.SGD(net, light.CrossEntropyLoss(), 0.01)

    print("Training model...")
    trainer = Trainer(optimizer, plot=True)
    net = trainer.train((X_train, y_train), 600, progress_bar=True)
    assert isinstance(net, FFNClassifier)

    # Accuracy on unseen data
    print("Measuring accuracy...")
    correct = 0
    false = 0
    for idx in tqdm(range(len(X_test))):
        pred_label = one_hot_encoder.decode(net(X_test.iloc[idx].to_numpy()))
        if pred_label == one_hot_encoder.decode(y_test.iloc[idx].to_numpy()):
            correct += 1
        else:
            false += 1
    print("Accuracy: {:.2f}%".format(correct * 100 / len(X_test)))

    out = GestureDetectorPipeline(
        pose_detector,
        Buffer(buffer_size),
        pca,
        y.iloc[:, 0].unique().tolist(),
        net
    )

    return out
