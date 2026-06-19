"""Optional utility to download Google Drive PDF files into data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.file_utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download Google Drive files with gdown.")
    parser.add_argument("url", nargs="+", help="Google Drive URLs.")
    parser.add_argument("--output-dir", default="data/raw", help="Destination directory.")
    return parser.parse_args()


def main() -> None:
    """Download files using gdown."""
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown: pip install gdown") from exc

    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    for index, url in enumerate(args.url, start=1):
        output_path = Path(output_dir) / f"file_{index}.pdf"
        gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True)
        print(f"Downloaded {output_path}")


if __name__ == "__main__":
    main()
