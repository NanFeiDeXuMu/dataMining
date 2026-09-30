"""A single landmark/feature contract for image extraction and live video."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "matplotlib-cache"))

import mediapipe as mp
import numpy as np

from .config import DETECTOR
from .features import normalize_landmarks


@dataclass
class Detection:
    status: str
    hand_count: int
    landmarks: list
    feature: np.ndarray | None
    handedness: str = ""
    milliseconds: float = 0.0


class HandDetector:
    def __init__(self, model_path, video=False, config=DETECTOR):
        self.video = video
        self.last_timestamp = -1
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE,
            **asdict(config),
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        started = time.perf_counter()
        if self.video:
            timestamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1_000_000)
            self.last_timestamp = timestamp
            result = self.detector.detect_for_video(image, timestamp)
        else:
            result = self.detector.detect(image)
        elapsed = (time.perf_counter() - started) * 1000
        hands = [np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
                 for points in result.hand_landmarks]
        if len(hands) != 1:
            return Detection("no_hand" if not hands else "ambiguous_multiple_hands", len(hands), hands, None,
                             milliseconds=elapsed)
        try:
            feature = normalize_landmarks(hands[0], rgb.shape[1], rgb.shape[0])
        except ValueError:
            return Detection("invalid_geometry", 1, hands, None, milliseconds=elapsed)
        handedness = result.handedness[0][0].category_name if result.handedness else ""
        return Detection("usable", 1, hands, feature, handedness, elapsed)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
