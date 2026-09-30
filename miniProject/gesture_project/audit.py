"""Read-only inspection of the supplied images and source manifest."""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .features import LABELS


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_image(project: Path, relative: str) -> Path:
    """Accept only relative manifest paths within the immutable image folder."""
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or ":" in relative:
        raise ValueError(f"Unsafe image path: {relative}")
    path = (project / Path(*posix.parts)).resolve()
    path.relative_to((project / "gesture_9000").resolve())
    return path


def audit_dataset(project: Path, output: Path, expected_per_class: int = 500):
    dataset = project / "gesture_9000"
    manifest = dataset / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "class_name", "class_id", "image_bytes",
                    "source_parquet", "source_row_index_0based", "original_image_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest is missing required columns")
        rows = list(reader)

    errors, inventory = [], []
    content_groups, source_groups, original_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    seen_paths, dimensions = set(), Counter()
    for index, row in enumerate(rows, 1):
        record = dict(row)
        try:
            path = resolve_image(project, row["image_path"])
            relative = path.relative_to(project).as_posix()
            if relative in seen_paths:
                raise ValueError("Duplicate manifest path")
            seen_paths.add(relative)
            label = row["class_name"]
            if label not in LABELS or int(row["class_id"]) != LABELS.index(label):
                raise ValueError("Class name and class ID do not match the fixed label mapping")
            if path.parent.name != label:
                raise ValueError("Directory label differs from manifest")
            encoded = path.read_bytes()
            if len(encoded) != int(row["image_bytes"]):
                raise ValueError("Byte size differs from source manifest")
            with Image.open(BytesIO(encoded)) as image:
                image.load()  # Decode all pixels, not just JPEG headers.
                if image.format != "JPEG":
                    raise ValueError("Image is not JPEG")
                width, height = image.size
                record.update(width=width, height=height, mode=image.mode)
                dimensions[(width, height)] += 1
            digest = hashlib.sha256(encoded).hexdigest()
            record.update(sha256=digest, decodable=True, error="")
            content_groups[digest].append(relative)
            source_groups[(row["source_parquet"], row["source_row_index_0based"])].append(relative)
            original_groups[row["original_image_path"]].append(relative)
        except (OSError, ValueError, KeyError) as exc:
            record.update(decodable=False, error=str(exc))
            errors.append({"image_path": row.get("image_path"), "error": str(exc)})
        inventory.append(record)
        if index % 1500 == 0:
            print(f"Decoded {index}/{len(rows)} images", flush=True)

    disk_paths = {p.relative_to(project).as_posix() for p in dataset.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    counts = Counter(r["class_name"] for r in rows)
    duplicates = [paths for paths in content_groups.values() if len(paths) > 1]
    duplicate_sources = [paths for paths in source_groups.values() if len(paths) > 1]
    duplicate_originals = [paths for paths in original_groups.values() if len(paths) > 1]
    unlisted, missing = sorted(disk_paths - seen_paths), sorted(seen_paths - disk_paths)
    expected_counts = {label: expected_per_class for label in LABELS}
    passed = (not errors and not unlisted and not missing and not duplicates
              and not duplicate_sources and not duplicate_originals and dict(counts) == expected_counts)
    fieldnames = list(dict.fromkeys(key for record in inventory for key in record))
    with (output / "image_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    summary = {
        "passed": passed, "manifest_sha256": sha256(manifest),
        "manifest_rows": len(rows), "disk_image_count": len(disk_paths),
        "class_counts": dict(counts), "decoded_images": sum(bool(r["decodable"]) for r in inventory),
        "errors": errors, "missing_images": missing, "unlisted_images": unlisted,
        "duplicate_byte_groups": duplicates, "duplicate_source_groups": duplicate_sources,
        "duplicate_original_path_groups": duplicate_originals,
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in sorted(dimensions.items())},
        "person_id_available": False,
        "split_limitation": "Manifest has no person ID; person-independent evaluation cannot be established from these columns.",
    }
    return summary, inventory
