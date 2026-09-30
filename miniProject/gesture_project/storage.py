"""Append-only run directories and atomically published active references."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def new_run(project=PROJECT):
    now = datetime.now()
    run = project / "data" / "processed" / now.strftime("%Y-%m") / (
        "run-" + now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    return run


def status(run, stage, **details):
    write_json(run / "status.json", {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details})


def contained_run(run, project=PROJECT):
    run = Path(run).resolve()
    run.relative_to((project / "data" / "processed").resolve())
    if not run.is_dir() or not run.name.startswith("run-"):
        raise ValueError("Expected an existing run directory under data/processed")
    return run


def active_run(project=PROJECT):
    pointer = project / "data" / "processed" / "active.json"
    if not pointer.is_file():
        raise FileNotFoundError("No trained model is active. Run: python -m gesture_project.build")
    return contained_run(project / read_json(pointer)["run_path"], project)
