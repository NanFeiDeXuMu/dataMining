import csv
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from gesture_project.audit import audit_dataset, resolve_image
from gesture_project.features import LABELS, normalize_landmarks
from gesture_project.prepare import verify_archive


def example_points():
    return np.random.default_rng(42).uniform(0.1, 0.7, (21, 3))


def test_translation_and_scale_do_not_change_features():
    points = example_points()
    baseline = normalize_landmarks(points, 640, 480)
    shifted = points * 0.4 + [0.1, 0.2, 0.3]
    np.testing.assert_allclose(normalize_landmarks(shifted, 640, 480), baseline, atol=1e-6)
    assert baseline.shape == (63,)
    assert baseline.dtype == np.float32
    np.testing.assert_array_equal(baseline[:3], [0, 0, 0])


def test_aspect_ratio_conversion_preserves_same_physical_geometry():
    points = example_points()
    taller_image_points = points.copy()
    taller_image_points[:, 1] /= 2
    np.testing.assert_allclose(normalize_landmarks(points, 640, 480),
                               normalize_landmarks(taller_image_points, 640, 960), atol=1e-6)


def test_direction_and_reflection_remain_distinguishable():
    points = example_points()
    mirrored = points.copy()
    mirrored[:, 0] = 1 - mirrored[:, 0]
    upside_down = points.copy()
    upside_down[:, :2] = 1 - upside_down[:, :2]
    original = normalize_landmarks(points, 640, 480)
    assert not np.allclose(original, normalize_landmarks(mirrored, 640, 480))
    assert not np.allclose(original, normalize_landmarks(upside_down, 640, 480))


@pytest.mark.parametrize("points,width,height", [
    (np.zeros((21, 3)), 640, 480),
    (np.zeros((20, 3)), 640, 480),
    (np.full((21, 3), np.nan), 640, 480),
    (example_points(), 0, 480),
])
def test_invalid_geometry_is_rejected(points, width, height):
    with pytest.raises(ValueError):
        normalize_landmarks(points, width, height)


def create_dataset(project: Path):
    rows = []
    for index, label in enumerate(LABELS):
        image = project / "gesture_9000" / label / "example.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (24, 24), (index * 10, 90, 20)).save(image)
        rows.append({"image_path": image.relative_to(project).as_posix(), "class_name": label,
                     "class_id": index, "image_bytes": image.stat().st_size,
                     "source_parquet": "source.parquet", "source_row_index_0based": index,
                     "original_image_path": f"original-{index}.jpeg"})
    manifest = project / "gesture_9000" / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = project / "output"
    output.mkdir()
    return output


def test_audit_detects_complete_and_corrupt_dataset(tmp_path):
    output = create_dataset(tmp_path)
    report, _ = audit_dataset(tmp_path, output, expected_per_class=1)
    assert report["passed"]
    assert report["decoded_images"] == 18
    (tmp_path / "gesture_9000" / "call" / "example.jpg").write_bytes(b"broken")
    report, _ = audit_dataset(tmp_path, output, expected_per_class=1)
    assert not report["passed"]
    assert report["decoded_images"] == 17
    assert report["errors"][0]["image_path"].endswith("call/example.jpg")


def test_audit_does_not_ignore_unlisted_images(tmp_path):
    output = create_dataset(tmp_path)
    Image.new("RGB", (24, 24)).save(tmp_path / "gesture_9000" / "extra.jpg")
    report, _ = audit_dataset(tmp_path, output, expected_per_class=1)
    assert not report["passed"]
    assert report["unlisted_images"] == ["gesture_9000/extra.jpg"]


@pytest.mark.parametrize("relative", ["../secret.jpg", "gesture_9000/../../secret.jpg", "C:/secret.jpg"])
def test_manifest_cannot_reference_outside_dataset(tmp_path, relative):
    with pytest.raises(ValueError):
        resolve_image(tmp_path, relative)


def test_archive_check_detects_changed_extraction(tmp_path):
    original = tmp_path / "original.txt"
    original.write_bytes(b"original content")
    archive = tmp_path / "resource.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(original, "original.txt")
    report = verify_archive(tmp_path, archive)
    assert report["verified_files"] == 1
    assert not report["errors"]
    original.write_bytes(b"changed content")
    report = verify_archive(tmp_path, archive)
    assert report["errors"] == [{"extracted_file_mismatch": "original.txt"}]


def test_archive_check_rejects_path_traversal(tmp_path):
    archive = tmp_path / "resource.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside.txt", b"irrelevant")
    with pytest.raises(ValueError):
        verify_archive(tmp_path, archive)
