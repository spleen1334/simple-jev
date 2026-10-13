"""Run with `uv run python scripts/download_model.py [--repo-id ... --filename ...]`."""

import sys

from simple_jev.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
