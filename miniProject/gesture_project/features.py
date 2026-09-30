"""Shared feature contract for offline images and future webcam inference."""

import numpy as np

LABELS = (
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm",
    "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three",
    "three2", "two_up", "two_up_inverted",
)
FEATURE_VERSION = "wrist-relative-aspect-corrected-63-v1"


def normalize_landmarks(points, width: int, height: int) -> np.ndarray:
    """Center on wrist and scale by wrist-to-middle-MCP distance.

    Image x and z use image-width units; y uses image-height units.
    Preserve direction, signed depth and chirality: labels distinguish these.
    No rotation, reflection, handedness canonicalization or image mirroring.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite xyz landmarks")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    relative = (points - points[0]) * np.array([width, height, width])
    scale = float(np.linalg.norm(relative[9]))
    if scale <= 1e-6:
        raise ValueError("Degenerate wrist-to-middle-MCP distance")
    return (relative / scale).reshape(63).astype(np.float32)
