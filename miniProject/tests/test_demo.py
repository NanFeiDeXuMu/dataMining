import numpy as np
import pytest

from gesture_project import demo
from gesture_project.detector import Detection
from gesture_project.presentation import PredictionSmoother, render_overlay
from gesture_project.gallery import select_examples
from gesture_project.features import LABELS


def probability(label):
    values = np.zeros(18)
    values[label] = 1
    return values


def test_no_hand_clears_stale_prediction_immediately():
    smoother = PredictionSmoother()
    smoother.update(probability(0), now=1)
    assert smoother.update(None, now=1.1) is None
    np.testing.assert_array_equal(smoother.update(probability(1), now=1.2), probability(1))


@pytest.mark.parametrize("new_time,new_wrist,new_hand", [(2, [0.1, 0.1], "Left"),
                                                       (1.1, [0.9, 0.9], "Left"),
                                                       (1.1, [0.1, 0.1], "Right")])
def test_track_changes_reset_smoothing(new_time, new_wrist, new_hand):
    smoother = PredictionSmoother()
    smoother.update(probability(0), [0.1, 0.1], "Left", now=1)
    result = smoother.update(probability(1), new_wrist, new_hand, now=new_time)
    np.testing.assert_array_equal(result, probability(1))


def test_overlay_does_not_mutate_input_and_handles_all_statuses():
    original = np.full((240, 320, 3), 90, dtype=np.uint8)
    for status in ["no_hand", "ambiguous_multiple_hands", "invalid_geometry", "usable"]:
        detection = Detection(status, 0, [], None)
        output = render_overlay(original, detection, probability(4) if status == "usable" else None, 0.6, mirror=True)
        assert output.shape == (672, 650, 3)
        assert np.all(original == 90)


def test_camera_is_released_if_detection_raises(monkeypatch, tmp_path):
    class Capture:
        released = False
        def release(self):
            self.released = True
    class Detector:
        closed = False
        def detect(self, image):
            raise RuntimeError("Detector failed")
        def close(self):
            self.closed = True
    capture, detector = Capture(), Detector()
    monkeypatch.setattr(demo, "open_camera", lambda *args: (capture, np.zeros((20, 20, 3), dtype=np.uint8)))
    monkeypatch.setattr(demo, "HandDetector", lambda *args, **kwargs: detector)
    with pytest.raises(RuntimeError, match="Detector failed"):
        demo.camera_loop(tmp_path, None, None, headless=True, max_frames=1)
    assert capture.released and detector.closed


def test_headless_camera_test_requires_a_bound_before_opening_camera(monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail("Camera should not be opened")
    monkeypatch.setattr(demo, "open_camera", forbidden)
    with pytest.raises(ValueError, match="positive"):
        demo.camera_loop(tmp_path, None, None, headless=True, max_frames=0)


def test_camera_backend_failure_releases_capture(monkeypatch):
    captures = []
    class FailedCapture:
        def __init__(self, *args):
            self.released = False
            captures.append(self)
        def isOpened(self):
            return False
        def release(self):
            self.released = True
    monkeypatch.setattr(demo.cv2, "VideoCapture", FailedCapture)
    with pytest.raises(RuntimeError, match="Cannot read camera"):
        demo.open_camera(0)
    assert len(captures) == 2
    assert all(capture.released for capture in captures)


def test_gallery_uses_test_images_without_selecting_by_prediction_correctness():
    rows = []
    for label in LABELS:
        rows.extend([
            {"class_name": label, "image_path": f"{label}/0.jpg", "split": "train", "status": "usable"},
            {"class_name": label, "image_path": f"{label}/2.jpg", "split": "test", "status": "usable"},
            {"class_name": label, "image_path": f"{label}/1.jpg", "split": "test", "status": "usable"},
        ])
    rows.append({"class_name": "call", "image_path": "failure.jpg", "split": "test", "status": "no_hand"})
    selected = select_examples(rows)
    assert len(selected) == 19
    assert all(row["split"] == "test" for row in selected)
    assert [row["image_path"] for row in selected[:18]] == [f"{label}/1.jpg" for label in LABELS]
    assert selected[-1]["status"] == "no_hand"


def test_gallery_reports_missing_class_instead_of_silently_omitting_it():
    with pytest.raises(ValueError, match="No usable test example"):
        select_examples([])
