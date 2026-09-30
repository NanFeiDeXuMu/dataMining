"""Split original images before detection, preserving all evaluation denominators."""

from collections import Counter
import csv
from dataclasses import asdict
import shutil
import time

import numpy as np
from PIL import Image

from .audit import audit_dataset, resolve_image, sha256
from .config import DETECTOR, TRAINING
from .features import FEATURE_VERSION, LABELS
from .storage import new_run, read_csv, status, write_csv, write_json


def assign_splits(rows, seed=42, train_fraction=0.7, validation_fraction=0.15):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Train, validation and test fractions must all be positive")
    paths = [row["image_path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths cannot be independently split")
    if set(row["class_name"] for row in rows) != set(LABELS):
        raise ValueError("All 18 labels are required")
    rng = np.random.default_rng(seed)
    output = []
    for label in LABELS:
        group = sorted((dict(row) for row in rows if row["class_name"] == label), key=lambda row: row["image_path"])
        order = rng.permutation(len(group))
        n_train, n_validation = int(len(group) * train_fraction), int(len(group) * validation_fraction)
        if min(n_train, n_validation, len(group) - n_train - n_validation) < 1:
            raise ValueError(f"Too few images in {label} for three splits")
        for position, index in enumerate(order):
            split = "train" if position < n_train else "validation" if position < n_train + n_validation else "test"
            output.append({**group[index], "split": split})
    return sorted(output, key=lambda row: row["image_path"])


def prepare_run(project, config=TRAINING):
    run = new_run(project)
    print(f"RUN_DIRECTORY={run}", flush=True)
    status(run, "auditing")
    try:
        report, rows = audit_dataset(project, run)
        model = project / "models" / "hand_landmarker.task"
        report["models"] = {"models/hand_landmarker.task": sha256(model)}
        write_json(run / "data_audit.json", report)
        if not report["passed"]:
            raise ValueError("Data audit failed; see data_audit.json")
        shutil.copyfile(model, run / "hand_landmarker.task")
        write_json(run / "config.json", {"training": asdict(config), "detector": asdict(DETECTOR),
                                         "feature_version": FEATURE_VERSION, "labels": list(LABELS)})
        split_rows = assign_splits(rows, config.seed, config.train_fraction, config.validation_fraction)
        write_csv(run / "split_manifest.csv", split_rows)
        status(run, "extracting", original_split_counts=dict(Counter(row["split"] for row in split_rows)))
        extract_features(project, run, split_rows)
        status(run, "features_ready")
        return run
    except Exception as exc:
        status(run, "failed", error=str(exc))
        raise


def extract_features(project, run, rows):
    from .detector import HandDetector
    n = len(rows)
    features = np.full((n, 63), np.nan, dtype=np.float32)
    usable = np.zeros(n, dtype=bool)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows])
    records = []
    started = time.perf_counter()
    columns = ["image_path", "class_name", "class_id", "split", "sha256", "status", "usable",
               "no_hand", "multiple_hands", "invalid_geometry", "hand_count", "handedness", "inference_ms"]
    with (run / "detections.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        with HandDetector(run / "hand_landmarker.task") as detector:
            for index, row in enumerate(rows):
                image = resolve_image(project, row["image_path"])
                if sha256(image) != row["sha256"]:
                    raise ValueError(f"Image changed after audit: {image}")
                with Image.open(image) as source:
                    rgb = np.asarray(source.convert("RGB"))
                detection = detector.detect(rgb)
                if detection.feature is not None:
                    features[index] = detection.feature
                    usable[index] = True
                record = {key: row[key] for key in ["image_path", "class_name", "class_id", "split", "sha256"]}
                record.update(status=detection.status, usable=bool(usable[index]), no_hand=detection.status == "no_hand",
                              multiple_hands=detection.status == "ambiguous_multiple_hands",
                              invalid_geometry=detection.status == "invalid_geometry", hand_count=detection.hand_count,
                              handedness=detection.handedness, inference_ms=detection.milliseconds)
                writer.writerow(record)
                records.append(record)
                if (index + 1) % 500 == 0:
                    stream.flush()
                    print(f"Extracted {index + 1}/{n}; usable={int(usable.sum())}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if not np.isfinite(features[usable]).all():
        raise ValueError("Non-finite features in usable rows")
    for split in ["train", "validation", "test"]:
        counts = np.bincount(labels[usable & (splits == split)], minlength=len(LABELS))
        if (counts < 2).any():
            raise ValueError(f"Insufficient usable images in {split}: {counts.tolist()}")
    np.savez_compressed(run / "features.npz", features=features, usable=usable, labels=labels, splits=splits,
                        image_paths=np.asarray([row["image_path"] for row in rows]), feature_version=FEATURE_VERSION)
    summary = {"original_images": n, "usable_images": int(usable.sum()),
               "status_counts": dict(Counter(record["status"] for record in records)),
               "elapsed_seconds": time.perf_counter() - started, "feature_version": FEATURE_VERSION,
               "splits": {split: {"original": int((splits == split).sum()),
                                   "usable": int((usable & (splits == split)).sum())}
                          for split in ["train", "validation", "test"]},
               "per_class": {label: {"original": int((labels == i).sum()),
                                     "usable": int((usable & (labels == i)).sum())}
                             for i, label in enumerate(LABELS)}}
    write_json(run / "extraction.json", summary)
    print(f"Feature extraction complete: {int(usable.sum())}/{n}", flush=True)
