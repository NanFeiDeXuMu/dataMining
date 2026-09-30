"""Recompute saved metrics and predictions before activating a model snapshot."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .audit import sha256
from .features import FEATURE_VERSION, LABELS
from .metrics import evaluate_predictions
from .predictor import Predictor
from .storage import PROJECT, contained_run, read_csv, read_json, status, write_json

REQUIRED_ARTIFACTS = {"classifier.pt", "baseline.joblib", "hand_landmarker.task", "config.json", "features.npz",
                      "split_manifest.csv", "detections.csv", "predictions.csv", "evaluation.json", "extraction.json",
                      "training_history.csv", "selection.json", "training_curves.png", "confusion_matrix.png", "REPORT.md",
                      "data_audit.json", "per_class_test.csv"}


def verify_artifact_hashes(run):
    hashes = read_json(run / "artifacts.json")["sha256"]
    if not REQUIRED_ARTIFACTS.issubset(hashes):
        raise ValueError("Artifact manifest is incomplete")
    for name, expected in hashes.items():
        path = (run / name).resolve()
        path.relative_to(run.resolve())
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or changed: {name}")
    return hashes


def validate_run(run, project=PROJECT):
    run = contained_run(run, project)
    hashes = verify_artifact_hashes(run)
    manifest, detections, predictions = (read_csv(run / name) for name in
                                       ["split_manifest.csv", "detections.csv", "predictions.csv"])
    config = read_json(run / "config.json")
    audit, evaluation = read_json(run / "data_audit.json"), read_json(run / "evaluation.json")
    if not audit["passed"] or config["labels"] != list(LABELS) or config["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible or failed source audit/configuration")
    paths = [row["image_path"] for row in manifest]
    if len(paths) != len(set(paths)) or len(paths) != audit["manifest_rows"]:
        raise ValueError("Source images are missing or duplicated")
    if len({row["sha256"] for row in manifest}) != len(paths):
        raise ValueError("Byte-identical images could leak across splits")
    if [row["image_path"] for row in detections] != paths:
        raise ValueError("Detections are not aligned with the source manifest")
    with np.load(run / "features.npz", allow_pickle=False) as archive:
        x, y, usable, splits, feature_paths = (archive[key] for key in ["features", "labels", "usable", "splits", "image_paths"])
        if str(archive["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Feature format mismatch")
    if x.shape != (len(paths), 63) or usable.dtype != np.bool_ or feature_paths.tolist() != paths:
        raise ValueError("Feature rows or boolean mask are invalid")
    if not np.isfinite(x[usable]).all() or not np.isnan(x[~usable]).all():
        raise ValueError("Invalid usable/failed feature values")
    if splits.tolist() != [row["split"] for row in manifest] or y.tolist() != [int(row["class_id"]) for row in manifest]:
        raise ValueError("Feature labels or splits disagree with the manifest")
    if not np.array_equal(usable, np.asarray([row["status"] == "usable" for row in detections])):
        raise ValueError("Feature availability disagrees with detector status")
    per_class = Counter(row["class_name"] for row in manifest)
    if dict(per_class) != audit["class_counts"] or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Class coverage or split membership changed")
    expected_paths = set(feature_paths[splits != "train"].tolist())
    if len(predictions) != len(expected_paths) or {row["image_path"] for row in predictions} != expected_paths:
        raise ValueError("Evaluation rows are incomplete, duplicated or contain training images")
    lookup = {path: i for i, path in enumerate(paths)}
    for row in predictions:
        i = lookup[row["image_path"]]
        if row["split"] != splits[i] or int(row["true_class_id"]) != y[i] or row["status"] != detections[i]["status"]:
            raise ValueError("Prediction source metadata mismatch")
        for name in ["mlp", "logistic_regression"]:
            if (int(row[f"{name}_class_id"]) >= 0) != usable[i]:
                raise ValueError("A detection failure has been counted as a classification")
    for split in ["validation", "test"]:
        rows = [row for row in predictions if row["split"] == split]
        for name in ["mlp", "logistic_regression"]:
            recomputed = evaluate_predictions([int(row["true_class_id"]) for row in rows],
                                             [int(row[f"{name}_class_id"]) for row in rows],
                                             [float(row[f"{name}_confidence"]) for row in rows],
                                             config["training"]["confidence_threshold"])
            if recomputed != evaluation[split][name]:
                raise ValueError(f"Saved metrics do not match prediction evidence: {split}/{name}")
    predictor = Predictor(run / "classifier.pt")
    good_rows = [row for row in predictions if row["status"] == "usable"]
    probabilities = predictor.probabilities(x[[lookup[row["image_path"]] for row in good_rows]])
    np.testing.assert_array_equal(probabilities.argmax(1), [int(row["mlp_class_id"]) for row in good_rows])
    np.testing.assert_allclose(probabilities.max(1), [float(row["mlp_confidence"]) for row in good_rows], atol=1e-5, rtol=1e-5)
    return {"passed": True, "original_images": len(paths), "usable_images": int(usable.sum()),
            "split_counts": dict(Counter(splits.tolist())), "evaluation_rows": len(predictions),
            "metrics_recomputed": True, "checkpoint_predictions_verified": True,
            "classifier_sha256": hashes["classifier.pt"]}


def publish_run(run, project=PROJECT):
    run = contained_run(run, project)
    # Nothing touches active.json until every check above has passed.
    result = validate_run(run, project)
    write_json(run / "validation.json", result)
    status(run, "complete", validation_passed=True)
    write_json(project / "data" / "processed" / "active.json",
               {"run_path": run.relative_to(project).as_posix(), "classifier_sha256": result["classifier_sha256"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    print(publish_run(args.run_dir) if args.activate else validate_run(args.run_dir))


if __name__ == "__main__":
    main()
