"""Load a validated feature contract and trained classifier for inference."""

import numpy as np
import torch

from .features import FEATURE_VERSION, LABELS
from .network import make_classifier


def fit_standardizer(training_features):
    x = np.asarray(training_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 63 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite training features with 63 columns")
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


class Predictor:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("purpose") != "trained_gesture_classifier":
            raise ValueError("This is not a final gesture classifier (smoke models are not supported)")
        if checkpoint.get("feature_version") != FEATURE_VERSION or checkpoint.get("labels") != list(LABELS):
            raise ValueError("Classifier feature version or label mapping is incompatible")
        self.mean = checkpoint["mean"].cpu().numpy()
        self.scale = checkpoint["scale"].cpu().numpy()
        if (self.mean.shape != (63,) or self.scale.shape != (63,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or (self.scale <= 0).any()):
            raise ValueError("Invalid saved normalization")
        self.model = make_classifier().eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.threshold = float(checkpoint["confidence_threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid confidence threshold")

    def probabilities(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 63 or not np.isfinite(x).all():
            raise ValueError("Expected finite 63-dimensional features")
        with torch.inference_mode():
            probabilities = self.model(torch.from_numpy((x - self.mean) / self.scale)).softmax(dim=1).numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite classifier output")
        return probabilities
