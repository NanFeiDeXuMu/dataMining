"""Build the complete project artifacts, or stop after full feature extraction."""

import argparse

from .dataset import prepare_run
from .storage import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    run = prepare_run(PROJECT)
    if not args.extract_only:
        from .train import train_run
        train_run(run)


if __name__ == "__main__":
    main()
