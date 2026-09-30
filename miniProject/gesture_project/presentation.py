"""Frame-local overlays and temporal smoothing that clears on hand loss."""

from collections import deque
import time

import cv2
import numpy as np

from .features import LABELS

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class PredictionSmoother:
    def __init__(self, window=5, max_gap=0.5, max_wrist_jump=0.25):
        self.history = deque(maxlen=window)
        self.max_gap, self.max_wrist_jump = max_gap, max_wrist_jump
        self.last_time, self.last_wrist, self.last_hand = None, None, None

    def reset(self):
        self.history.clear()
        self.last_time = self.last_wrist = self.last_hand = None

    def update(self, probabilities, wrist=None, handedness="", now=None):
        now = time.monotonic() if now is None else now
        if probabilities is None:
            self.reset()
            return None
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (18,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
            raise ValueError("Expected 18 probabilities summing to one")
        wrist = None if wrist is None else np.asarray(wrist, dtype=float)
        stale = self.last_time is not None and (now - self.last_time > self.max_gap or now < self.last_time)
        moved = wrist is not None and self.last_wrist is not None and np.linalg.norm(wrist - self.last_wrist) > self.max_wrist_jump
        swapped = bool(handedness and self.last_hand and handedness != self.last_hand)
        if stale or moved or swapped:
            self.reset()
        self.history.append(values.copy())
        self.last_time, self.last_wrist, self.last_hand = now, wrist, handedness
        return np.mean(self.history, axis=0)


def render_overlay(bgr, detection, probabilities, threshold, fps=None, mirror=False, subtitle="Live camera"):
    view = cv2.flip(bgr, 1) if mirror else bgr.copy()
    h, w = view.shape[:2]
    for landmarks in detection.landmarks:
        points = [(int(np.clip((1 - p[0]) if mirror else p[0], 0, 1) * (w - 1)),
                   int(np.clip(p[1], 0, 1) * (h - 1))) for p in landmarks]
        for start, end in CONNECTIONS:
            cv2.line(view, points[start], points[end], (60, 220, 90), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(view, point, 4, (0, 210, 255), -1, cv2.LINE_AA)
    canvas = np.full((max(h, 600) + 72, w + 330, 3), (28, 23, 20), dtype=np.uint8)
    canvas[72:72 + h, :w] = view
    cv2.putText(canvas, "GESTURE LAB", (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 245, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle[:100], (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 183, 188), 1, cv2.LINE_AA)
    x = w + 20
    def text(value, y, color=(225, 228, 232), scale=0.55):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text("PREDICTION", 105, (140, 180, 175), 0.55)
    if probabilities is None:
        label = {"no_hand": "No hand detected", "ambiguous_multiple_hands": "Show one hand",
                 "invalid_geometry": "Adjust hand position"}.get(detection.status, "Waiting")
        text(label, 150, (100, 190, 255), 0.63)
    else:
        ranked = np.argsort(probabilities)[::-1]
        score = float(probabilities[ranked[0]])
        label = LABELS[ranked[0]] if score >= threshold else "Uncertain"
        text(label, 148, (115, 235, 130) if score >= threshold else (100, 190, 255), 0.67)
        text(f"Score: {score:.1%}", 182, scale=0.65)
        text("Top candidates", 230, (140, 180, 175))
        for i, index in enumerate(ranked[:3]):
            text(f"{LABELS[index]:<18} {probabilities[index]:.0%}", 264 + 34 * i, scale=0.48)
    text("Image preview" if fps is None else f"FPS: {fps:.1f}", 405)
    text(f"Detector: {detection.milliseconds:.1f} ms", 435)
    text(f"Threshold: {threshold:.2f}", 465)
    text("Score is not calibrated", 500, (140, 150, 155), 0.45)
    if fps is None:
        text("One image, one prediction", 548, scale=0.47)
        text("Try another photo to explore", 577, scale=0.45)
    else:
        text("Q / Esc  Quit", 548, scale=0.5)
        text("M  Mirror display", 575, scale=0.5)
        text("S  Save screenshot", 602, scale=0.5)
    return canvas
