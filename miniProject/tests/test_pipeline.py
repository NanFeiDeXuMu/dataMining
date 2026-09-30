from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from gesture_project.audit import sha256
from gesture_project.config import TRAINING
from gesture_project.dataset import assign_splits
from gesture_project.features import FEATURE_VERSION, LABELS
from gesture_project.metrics import evaluate_predictions
from gesture_project.network import make_classifier
from gesture_project.predictor import Predictor, fit_standardizer
from gesture_project.storage import read_json, write_json
from gesture_project.train import fit_mlp
from gesture_project.validate import REQUIRED_ARTIFACTS, publish_run, verify_artifact_hashes


def sample_rows(n=20):
    return [{"image_path": f"gesture_9000/{label}/{i}.jpg", "class_name": label, "class_id": index}
            for index, label in enumerate(LABELS) for i in range(n)]


def test_split_is_disjoint_complete_balanced_and_order_independent():
    rows = sample_rows()
    split = assign_splits(rows)
    assert split == assign_splits(list(reversed(rows)))
    assert len({row["image_path"] for row in split}) == len(rows)
    for label in LABELS:
        assert Counter(row["split"] for row in split if row["class_name"] == label) == {
            "train": 14, "validation": 3, "test": 3}
    assert split != assign_splits(rows, seed=43)


def test_split_rejects_duplicate_paths_and_missing_classes():
    rows = sample_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        assign_splits([*rows, rows[0]])
    with pytest.raises(ValueError, match="18 labels"):
        assign_splits([row for row in rows if row["class_name"] != "call"])


def test_detector_failure_stays_in_end_to_end_denominator():
    metrics = evaluate_predictions([0, 1, 1, 2], [0, 1, -1, 0], [0.9, 0.8, 0.0, 0.4])
    assert metrics["original_images"] == 4
    assert metrics["usable_images"] == 3
    assert metrics["accuracy_on_usable"] == pytest.approx(2 / 3)
    assert metrics["end_to_end_accuracy"] == 0.5
    assert metrics["accepted_predictions"] == 2
    assert metrics["accuracy_on_accepted"] == 1.0
    assert metrics["confusion_matrix_with_detection_failure"][1][-1] == 1


def test_all_detection_failures_do_not_create_nan_or_fake_accuracy():
    metrics = evaluate_predictions([0, 1], [-1, -1], [0, 0])
    assert metrics["end_to_end_accuracy"] == 0
    assert metrics["accuracy_on_usable"] is None
    assert metrics["accuracy_on_accepted"] is None
    assert metrics["macro_f1_on_usable"] is None


@pytest.mark.parametrize("pred,score", [([18], [0.5]), ([0], [np.nan]), ([0], [1.2])])
def test_invalid_predictions_are_rejected(pred, score):
    with pytest.raises(ValueError):
        evaluate_predictions([0], pred, score)


def test_scaler_uses_only_passed_training_data_and_handles_constant_columns():
    train = np.tile(np.arange(63, dtype=np.float32), (3, 1))
    train[:, 10] = [1, 2, 3]
    mean, scale = fit_standardizer(train)
    assert mean[10] == 2
    assert scale[0] == 1
    assert np.isfinite((train - mean) / scale).all()
    validation = train + 10000
    assert ((validation - mean) / scale).mean() > 1000
    np.testing.assert_array_equal(mean, fit_standardizer(train)[0])


def save_checkpoint(path, **overrides):
    torch.manual_seed(42)
    checkpoint = {"purpose": "trained_gesture_classifier", "state_dict": make_classifier().state_dict(),
                  "labels": list(LABELS), "feature_version": FEATURE_VERSION,
                  "mean": torch.zeros(63), "scale": torch.ones(63), "confidence_threshold": 0.6}
    checkpoint.update(overrides)
    torch.save(checkpoint, path)


@pytest.mark.parametrize("overrides", [{"purpose": "smoke_test_only"}, {"labels": list(reversed(LABELS))},
                                      {"feature_version": "other"}, {"scale": torch.zeros(63)}])
def test_predictor_rejects_incompatible_checkpoints(tmp_path, overrides):
    path = tmp_path / "model.pt"
    save_checkpoint(path, **overrides)
    with pytest.raises(ValueError):
        Predictor(path)


def test_checkpoint_predictions_are_batch_independent_and_deterministic(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path)
    predictor = Predictor(path)
    x = np.random.default_rng(42).normal(size=(10, 63)).astype(np.float32)
    batch = predictor.probabilities(x)
    one = np.vstack([predictor.probabilities(row) for row in x])
    np.testing.assert_allclose(batch, one, atol=1e-6)
    np.testing.assert_allclose(batch.sum(1), 1, atol=1e-6)
    np.testing.assert_array_equal(batch, predictor.probabilities(x))


def test_fit_returns_minimum_validation_loss_checkpoint():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(72, 63)).astype(np.float32)
    y = np.tile(np.arange(18), 4).astype(np.int64)
    config = replace(TRAINING, max_epochs=4, patience=3)
    model, history, selection = fit_mlp(x[:54], y[:54], x[54:], y[54:], config)
    with torch.inference_mode():
        restored_loss = float(torch.nn.functional.cross_entropy(model(torch.from_numpy(x[54:])), torch.from_numpy(y[54:])))
    assert restored_loss == pytest.approx(min(row["validation_loss"] for row in history))
    assert selection["selected_epoch"] == min(history, key=lambda row: row["validation_loss"])["epoch"]


def fake_artifacts(tmp_path):
    run = tmp_path / "data" / "processed" / "2026-09" / "run-test"
    run.mkdir(parents=True)
    for name in REQUIRED_ARTIFACTS:
        (run / name).write_bytes(name.encode())
    write_json(run / "artifacts.json", {"sha256": {name: sha256(run / name) for name in REQUIRED_ARTIFACTS}})
    return run


def test_failed_validation_never_changes_existing_active_pointer(tmp_path):
    run = fake_artifacts(tmp_path)
    active = tmp_path / "data" / "processed" / "active.json"
    write_json(active, {"run_path": "old-valid-run"})
    before = active.read_bytes()
    (run / "classifier.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        publish_run(run, tmp_path)
    assert active.read_bytes() == before


def test_artifact_manifest_cannot_read_outside_run(tmp_path):
    run = fake_artifacts(tmp_path)
    manifest = read_json(run / "artifacts.json")
    manifest["sha256"]["../outside"] = "anything"
    write_json(run / "artifacts.json", manifest)
    with pytest.raises(ValueError):
        verify_artifact_hashes(run)
