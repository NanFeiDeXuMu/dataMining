"""Exercise MediaPipe -> shared features -> PyTorch -> checkpoint round trip.

This uses a small sample only. Its validation loss is an engineering check,
not a final estimate of gesture recognition accuracy.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "matplotlib-cache"))

import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torch import nn

from .audit import resolve_image, sha256
from .features import FEATURE_VERSION, LABELS, normalize_landmarks
from .network import make_classifier


def run_smoke(project: Path, run: Path, samples_per_class: int = 10, epochs: int = 3):
    if samples_per_class < 3 or epochs < 1:
        raise ValueError("At least 3 images per class and one epoch are required")
    run = run.resolve()
    run.relative_to((project / "data" / "processed").resolve())
    if (run / "smoke_result.json").exists():
        raise FileExistsError("Completed runs are immutable; use a new audit run")
    audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Data audit must pass before smoke testing")
    if sha256(project / "gesture_9000" / "manifest.csv") != audit["manifest_sha256"]:
        raise ValueError("Manifest changed since the audit")
    model_path = project / "models" / "hand_landmarker.task"
    model_hashes = {key.replace("\\", "/"): value for key, value in audit["models"].items()}
    if sha256(model_path) != model_hashes["models/hand_landmarker.task"]:
        raise ValueError("Landmarker model changed since the audit")
    with (run / "image_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    rng = random.Random(42)
    selected = []
    for label in LABELS:
        candidates = sorted((r for r in inventory if r["class_name"] == label), key=lambda r: r["image_path"])
        selected.extend(rng.sample(candidates, min(samples_per_class, len(candidates))))

    features, targets, paths, statuses = [], [], [], []
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    started = time.perf_counter()
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(selected, 1):
            path = resolve_image(project, row["image_path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Image changed since the audit: {path}")
            with Image.open(path) as source:
                rgb = np.array(source.convert("RGB"))
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            before = time.perf_counter()
            result = detector.detect(frame)
            status = {"image_path": row["image_path"], "label": row["class_name"],
                      "detected_hands": len(result.hand_landmarks),
                      "inference_ms": (time.perf_counter() - before) * 1000}
            if len(result.hand_landmarks) != 1:
                status["status"] = "no_hand" if not result.hand_landmarks else "ambiguous_multiple_hands"
            else:
                points = [[p.x, p.y, p.z] for p in result.hand_landmarks[0]]
                try:
                    vector = normalize_landmarks(points, rgb.shape[1], rgb.shape[0])
                    features.append(vector)
                    targets.append(LABELS.index(row["class_name"]))
                    paths.append(row["image_path"])
                    status["status"] = "usable"
                except ValueError as exc:
                    status.update(status="invalid_geometry", error=str(exc))
            statuses.append(status)
            if index % 30 == 0:
                print(f"Landmarks: {index}/{len(selected)}, usable: {len(features)}", flush=True)
    extraction_seconds = time.perf_counter() - started
    columns = ["image_path", "label", "detected_hands", "inference_ms", "status", "error"]
    with (run / "smoke_detections.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(statuses)

    counts = Counter(targets)
    if any(counts[i] < 3 for i in range(len(LABELS))):
        raise ValueError("Too few detected hands in a class for a smoke split; inspect smoke_detections.csv")
    x, y = np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)
    generator = np.random.default_rng(42)
    train_indices, validation_indices = [], []
    for class_id in range(len(LABELS)):
        indices = np.flatnonzero(y == class_id)
        generator.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * 0.2)))
        validation_indices.extend(indices[:n_validation].tolist())
        train_indices.extend(indices[n_validation:].tolist())
    np.savez_compressed(run / "smoke_features.npz", features=x, labels=y,
                        image_paths=np.asarray(paths), train_indices=train_indices,
                        validation_indices=validation_indices, feature_version=FEATURE_VERSION)

    torch.manual_seed(42)
    torch.set_num_threads(4)
    classifier = make_classifier().cpu()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    train_x, train_y = torch.from_numpy(x[train_indices]), torch.from_numpy(y[train_indices])
    val_x, val_y = torch.from_numpy(x[validation_indices]), torch.from_numpy(y[validation_indices])
    before_weight = classifier[0].weight.detach().clone()
    history = []
    for epoch in range(1, epochs + 1):
        classifier.train()
        order = torch.randperm(len(train_x))
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = criterion(classifier(train_x[batch]), train_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.inference_mode():
            train_loss = float(criterion(classifier(train_x), train_y))
            val_loss = float(criterion(classifier(val_x), val_y))
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Non-finite evaluation loss")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Smoke epoch {epoch}/{epochs}: train={train_loss:.4f}, validation={val_loss:.4f}", flush=True)
    if torch.equal(before_weight, classifier[0].weight):
        raise ValueError("Training did not update weights")

    checkpoint = run / "smoke_classifier.pt"
    torch.save({"state_dict": classifier.state_dict(), "labels": list(LABELS),
                "feature_version": FEATURE_VERSION, "purpose": "smoke_test_only"}, checkpoint)
    restored = make_classifier().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(saved["state_dict"])
    with torch.inference_mode():
        if not torch.allclose(classifier(val_x), restored(val_x), atol=1e-7, rtol=1e-6):
            raise ValueError("Checkpoint round-trip predictions differ")
    with (run / "smoke_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], marker="o", label="Training")
    ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Smoke test only - not final training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "smoke_loss.png", dpi=160)
    plt.close(fig)

    summary = {
        "passed": True, "purpose": "engineering_smoke_test_not_final_evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": 42, "device": "cpu",
        "sampled_images": len(selected), "usable_images": len(features),
        "status_counts": dict(Counter(s["status"] for s in statuses)),
        "usable_per_class": {label: counts[i] for i, label in enumerate(LABELS)},
        "train_samples": len(train_indices), "validation_samples": len(validation_indices),
        "epochs": epochs, "history": history, "extraction_seconds": extraction_seconds,
        "checkpoint_roundtrip_passed": True, "feature_version": FEATURE_VERSION,
        "detection_config": {"num_hands": 2, "min_hand_detection_confidence": 0.5,
                             "min_hand_presence_confidence": 0.5, "min_tracking_confidence": 0.5},
        "versions": {name: importlib.metadata.version(name) for name in
                     ["mediapipe", "torch", "opencv-contrib-python", "numpy", "Pillow", "matplotlib", "scikit-learn"]},
        "not_yet_verified": ["full_dataset_landmark_yield", "final_classifier_accuracy", "webcam_demo"],
    }
    (run / "smoke_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    print(json.dumps(run_smoke(project, args.run_dir, args.samples_per_class, args.epochs),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
