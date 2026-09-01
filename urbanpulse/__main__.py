"""Run with python -m urbanpulse prepare --month 2025-01."""

import argparse
from pathlib import Path
import sys

from .config import data_root, month_bounds


def main():
    parser = argparse.ArgumentParser(description="UrbanPulse v0.1 数据管道")
    parser.add_argument("command", choices=["download", "build", "prepare"])
    parser.add_argument("--month", default="2025-01", help="固定月份 YYYY-MM")
    parser.add_argument("--data-dir", type=Path, default=data_root())
    args = parser.parse_args()
    try:
        month_bounds(args.month)
        if args.command in ("download", "prepare"):
            from .download import download
            download(args.month, args.data_dir.resolve())
        if args.command in ("build", "prepare"):
            from .pipeline import build
            build(args.month, args.data_dir.resolve())
    except (OSError, ValueError, KeyError) as exc:
        print(f"无法完成：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
