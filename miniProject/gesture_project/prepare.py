"""Create a fresh, auditable readiness run without changing original images."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
import zipfile
import zlib

from .audit import audit_dataset, sha256


def verify_archive(project: Path, archive: Path):
    errors, verified = [], 0
    with zipfile.ZipFile(archive) as zipped:
        corrupt = zipped.testzip()
        if corrupt:
            errors.append({"archive_crc_error": corrupt})
        for info in zipped.infolist():
            if info.is_dir():
                continue
            local = (project / info.filename).resolve()
            local.relative_to(project)
            if not local.is_file() or zlib.crc32(local.read_bytes()) != info.CRC:
                errors.append({"extracted_file_mismatch": info.filename})
            else:
                verified += 1
    return {"path": str(archive), "sha256": sha256(archive), "verified_files": verified, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional original downloaded ZIP for CRC verification")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    now = datetime.now()
    run = project / "data" / "processed" / now.strftime("%Y-%m") / (
        "run-" + now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    report, _ = audit_dataset(project, run)
    model_paths = [project / "hand_landmarker.task", project / "models" / "hand_landmarker.task"]
    report["models"] = {p.relative_to(project).as_posix(): sha256(p) for p in model_paths if p.is_file()}
    model_ok = "models/hand_landmarker.task" in report["models"] and len(set(report["models"].values())) == 1
    report["passed"] = report["passed"] and model_ok
    if args.archive:
        report["archive"] = verify_archive(project, args.archive)
        report["passed"] = report["passed"] and not report["archive"]["errors"]
    (run / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RUN_DIRECTORY={run}", flush=True)
    print(f"Data audit passed: {report['passed']}; decoded images: {report['decoded_images']}", flush=True)
    if not report["passed"]:
        raise SystemExit("Audit failed. Inspect data_audit.json; original inputs remain unchanged.")
    if not args.audit_only:
        from .smoke import run_smoke
        smoke = run_smoke(project, run, args.samples_per_class, args.epochs)
        print(f"Smoke test passed: {smoke['passed']}; usable sample: {smoke['usable_images']}", flush=True)
    # Preparation is not a full trained-model release: do not write active.json.


if __name__ == "__main__":
    main()
