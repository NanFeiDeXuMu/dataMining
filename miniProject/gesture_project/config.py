"""Versioned defaults shared by training and the live application."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_epochs: int = 150
    patience: int = 20
    min_delta: float = 0.0001
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    cpu_threads: int = 4
    confidence_threshold: float = 0.60


DETECTOR = DetectorConfig()
TRAINING = TrainingConfig()
