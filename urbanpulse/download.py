"""Download official public inputs; never overwrite an existing raw snapshot."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
from urllib.request import Request, urlopen

from .config import DICTIONARY_URL, SOURCE_PAGE, ZONE_URL, month_bounds, trip_url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Generated metadata only; atomic replacement avoids half-written pointers."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"保留已有原始文件: {destination.name}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    req = Request(url, headers={"User-Agent": "UrbanPulse-course-project/0.1"})
    print(f"下载: {url}", flush=True)
    with urlopen(req, timeout=60) as response, temporary.open("wb") as stream:
        length = response.headers.get("Content-Length")
        total, last_report = 0, 0
        while block := response.read(1024 * 1024):
            stream.write(block)
            total += len(block)
            if total - last_report >= 16 * 1024 * 1024:
                print(f"  已接收 {total / 1024**2:.1f} MiB", flush=True)
                last_report = total
        if not total or (length and total != int(length)):
            raise ValueError("下载不完整；未替换原始文件，请重新运行。")
    os.replace(temporary, destination)


def download(month: str, root: Path) -> Path:
    month_bounds(month)
    folder = root / "raw" / month
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, url in ((f"yellow_tripdata_{month}.parquet", trip_url(month)), ("taxi_zone_lookup.csv", ZONE_URL)):
        path = folder / name
        fetch(url, path)
        digest = sha256(path)
        entries.append({"file": name, "url": url, "bytes": path.stat().st_size, "sha256": digest})
    manifest_path = folder / "source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("files") != entries:
            raise ValueError("原始文件与已有来源清单不一致。请检查文件，不自动覆盖快照。")
    else:
        write_json(manifest_path, {
            "month": month, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_page": SOURCE_PAGE, "dictionary_url": DICTIONARY_URL, "files": entries,
        })
    return folder
