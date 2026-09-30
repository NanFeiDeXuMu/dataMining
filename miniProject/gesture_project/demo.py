"""OpenCV desktop demo, offline image inference, and bounded camera diagnostics."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .audit import sha256
from .config import DetectorConfig
from .detector import HandDetector
from .features import LABELS
from .predictor import Predictor
from .presentation import PredictionSmoother, render_overlay
from .storage import PROJECT, active_run, contained_run, read_json
from .validate import verify_artifact_hashes


def load_run(run=None):
    run = active_run() if run is None else contained_run(run)
    verify_artifact_hashes(run)
    if not read_json(run / "validation.json")["passed"]:
        raise ValueError("Model run has not passed validation")
    torch.set_num_threads(4)
    predictor = Predictor(run / "classifier.pt")
    detector_config = DetectorConfig(**read_json(run / "config.json")["detector"])
    return run, predictor, detector_config


def open_camera(index, backend="auto"):
    choices = {"dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF], "auto": [cv2.CAP_DSHOW, cv2.CAP_MSMF]}
    if index < 0 or backend not in choices:
        raise ValueError("Invalid camera index or backend")
    for api in choices[backend]:
        capture = cv2.VideoCapture(index, api)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = capture.read()
            if ok and frame is not None:
                return capture, frame
        capture.release()
    raise RuntimeError(f"Cannot read camera {index}. Check Windows camera permissions, camera connection, and other camera apps; try --camera 1 or --backend msmf. Offline mode: --image <path> --output <new-output.png>.")


def image_inference(run, predictor, config, path, output=None):
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    with HandDetector(run / "hand_landmarker.task", config=config) as detector:
        detection = detector.detect(rgb)
    probabilities = predictor.probabilities(detection.feature)[0] if detection.feature is not None else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.resolve() == path.resolve():
            raise FileExistsError("Choose a new output path; existing files are not overwritten")
        frame = render_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detection, probabilities, predictor.threshold,
                               subtitle=f"Image: {path.name}")
        if not cv2.imwrite(str(output), frame):
            raise OSError("Could not write annotated image")
    return {"status": detection.status, "predicted_label": LABELS[int(probabilities.argmax())] if probabilities is not None else None,
            "score": float(probabilities.max()) if probabilities is not None else None,
            "accepted": bool(probabilities is not None and probabilities.max() >= predictor.threshold)}


def camera_loop(run, predictor, config, camera=0, backend="auto", headless=False, max_frames=0, mirror=True):
    if headless and max_frames < 1:
        raise ValueError("Headless diagnostics require a positive --max-frames")
    capture = detector = None
    frame_count, usable_count, accepted_count = 0, 0, 0
    start = time.perf_counter()
    smoother = PredictionSmoother()
    try:
        capture, bgr = open_camera(camera, backend)
        detector = HandDetector(run / "hand_landmarker.task", video=True, config=config)
        while True:
            detection = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            probabilities = predictor.probabilities(detection.feature)[0] if detection.feature is not None else None
            probabilities = smoother.update(probabilities, detection.landmarks[0][0, :2] if detection.feature is not None else None,
                                              detection.handedness)
            frame_count += 1
            usable_count += int(probabilities is not None)
            accepted_count += int(probabilities is not None and probabilities.max() >= predictor.threshold)
            fps = frame_count / max(time.perf_counter() - start, 1e-6)
            if not headless:
                overlay = render_overlay(bgr, detection, probabilities, predictor.threshold, fps, mirror,
                                         "18 gestures | Hold one hand inside the frame")
                cv2.imshow("Gesture Lab", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord("q")]:
                    break
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("s"):
                    directory = PROJECT / "reports" / "screenshots"
                    directory.mkdir(parents=True, exist_ok=True)
                    output = directory / (datetime.now().strftime("gesture-%Y%m%d-%H%M%S-%f") + ".png")
                    if not cv2.imwrite(str(output), overlay):
                        raise OSError("Failed to save screenshot")
                    print(f"Screenshot saved: {output}", flush=True)
                if cv2.getWindowProperty("Gesture Lab", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if max_frames and frame_count >= max_frames:
                break
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError("Camera disconnected or stopped delivering frames")
        return {"frames": frame_count, "frames_with_usable_hand": usable_count, "accepted_frames": accepted_count,
                "processing_fps": frame_count / max(time.perf_counter() - start, 1e-6),
                "recording_saved": False, "note": "Frame availability is not labeled gesture accuracy"}
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            try:
                if not headless:
                    cv2.destroyAllWindows()
            finally:
                if detector is not None:
                    detector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--image", type=Path, help="Infer one local image instead of opening the webcam")
    parser.add_argument("--output", type=Path, help="New annotated output file for --image")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto")
    parser.add_argument("--headless", action="store_true", help="Bounded camera test without window or saving frames")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    if args.output and not args.image:
        parser.error("--output requires --image")
    if args.max_frames < 0:
        parser.error("--max-frames must not be negative")
    try:
        run, predictor, config = load_run(args.run_dir)
        result = image_inference(run, predictor, config, args.image, args.output) if args.image else camera_loop(
            run, predictor, config, args.camera, args.backend, args.headless, args.max_frames, not args.no_mirror)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Gesture Lab: {exc}\n")


if __name__ == "__main__":
    main()
