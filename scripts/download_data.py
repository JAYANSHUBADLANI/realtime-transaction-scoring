"""Download the PaySim dataset via kagglehub and place it where the pipeline
expects it (``data/paysim_transactions.csv``, per ``config.yaml``).

Usage:
    pip install kagglehub
    python scripts/download_data.py

No Kaggle credentials are required for this dataset on kagglehub; it pulls
from a public mirror. If kagglehub ever does prompt for a token, see
data/README.md for the manual Kaggle CLI / browser download path instead.
"""

from __future__ import annotations

import os
import shutil

import kagglehub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(PROJECT_ROOT, "data", "paysim_transactions.csv")
SOURCE_FILENAME = "PS_20174392719_1491204439457_log.csv"


def main() -> None:
    print("Downloading ealaxi/paysim1 via kagglehub...")
    path = kagglehub.dataset_download("ealaxi/paysim1")
    source = os.path.join(path, SOURCE_FILENAME)
    if not os.path.exists(source):
        found = os.listdir(path)
        raise FileNotFoundError(
            f"Expected {SOURCE_FILENAME} in {path}, found: {found}. "
            "The Kaggle dataset may have been repackaged; update "
            "SOURCE_FILENAME in this script to match."
        )

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.copyfile(source, DEST)
    size_mb = os.path.getsize(DEST) / (1024 * 1024)
    print(f"Copied to {DEST} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
