"""Small landmark classifier; the hand detector remains pretrained."""

from torch import nn

from .features import LABELS


def make_classifier():
    return nn.Sequential(
        nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(64, len(LABELS)),
    )
