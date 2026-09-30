"""Explicit denominators, including detector failures in end-to-end evaluation."""

import numpy as np

from .features import LABELS


def evaluate_predictions(truth, predictions, confidence, threshold=0.6):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    n_classes = len(LABELS)
    if not (truth.shape == predictions.shape == confidence.shape) or truth.ndim != 1 or not len(truth):
        raise ValueError("Expected nonempty matching one-dimensional evaluation arrays")
    if ((truth < 0) | (truth >= n_classes)).any() or ((predictions < -1) | (predictions >= n_classes)).any():
        raise ValueError("Unknown class index")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Invalid confidence")
    usable = predictions >= 0
    correct = usable & (predictions == truth)
    accepted = usable & (confidence >= threshold)
    cm = np.zeros((n_classes, n_classes + 1), dtype=int)
    np.add.at(cm, (truth, np.where(usable, predictions, n_classes)), 1)
    tp = np.diag(cm[:, :n_classes]).astype(float)
    actual = cm[:, :n_classes].sum(axis=1)
    predicted = cm[:, :n_classes].sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros(n_classes), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes), where=(precision + recall) != 0)
    n_usable, n_accepted, n_correct = int(usable.sum()), int(accepted.sum()), int(correct.sum())
    return {
        "original_images": len(truth), "usable_images": n_usable, "correct_predictions": n_correct,
        "feature_coverage": n_usable / len(truth),
        "accuracy_on_usable": n_correct / n_usable if n_usable else None,
        "end_to_end_accuracy": n_correct / len(truth),
        "macro_f1_on_usable": float(f1.mean()) if n_usable else None,
        "confidence_threshold": threshold, "accepted_predictions": n_accepted,
        "accepted_coverage_original": n_accepted / len(truth),
        "accuracy_on_accepted": int((correct & accepted).sum()) / n_accepted if n_accepted else None,
        "confusion_matrix_with_detection_failure": cm.tolist(),
        "per_class": [{"label": label, "original_images": int(cm[i].sum()), "usable_images": int(actual[i]),
                       "correct_predictions": int(tp[i]), "precision_on_usable": float(precision[i]),
                       "recall_on_usable": float(recall[i]), "f1_on_usable": float(f1[i]),
                       "end_to_end_recall": float(tp[i] / cm[i].sum()) if cm[i].sum() else None}
                      for i, label in enumerate(LABELS)],
    }
