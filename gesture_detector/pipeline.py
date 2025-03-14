from light import PCA, OneHotEncoder

from gesture_detector.buffer import Buffer
from gesture_detector.classifier import FFNClassifier
from gesture_detector.pose_detection.base import PoseDetector
from gesture_detector.feature_extraction import FeatureExtractor


class GestureDetectorPipeline:
    def __init__(
            self,
            pose_detector: PoseDetector,
            buffer: Buffer,
            pca: PCA,
            gesture_list: list[str],
            classifier: FFNClassifier
    ):
        self.pose_detector = pose_detector
        self.buffer = buffer
        self.pca = pca
        self.feature_extractor = FeatureExtractor()
        self.classifier = classifier
        self.one_hot_encoder = OneHotEncoder(gesture_list)

        assert self.classifier.in_dim == pca.n_components * buffer.buffer_size
        assert self.classifier.out_dim == len(gesture_list)


    def process(self):
        pose = self.pose_detector.get_pose()
        feature_vector = self.feature_extractor.extract_features(pose)
        feature_vector = self.pca.transform(feature_vector)
        self.buffer.add(feature_vector)
        in_next = self.buffer.get_flatten()

        if in_next is None:
            return None

        label = self.one_hot_encoder.decode(self.classifier(in_next))

        return label
