"""Verify that data/raw/ contains the expected datasets, splits, and classes.

Usage:
    python scripts/verify_data.py

Exits non-zero if anything is missing so it can be used as a preflight check
before training. Expected counts are what the datasets contained when this
pipeline was built (July 2026); a small drift after an upstream update is
reported as a warning, an empty or missing folder is an error.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# dataset -> split -> class -> expected image count
EXPECTED_CLASSIFICATION = {
    "skin_type": {
        "train": {"dry": 652, "normal": 1104, "oily": 1000},
        "valid": {"dry": 71, "normal": 111, "oily": 80},
        "test": {"dry": 35, "normal": 59, "oily": 40},
    },
    "acne_type": {
        "train": {"Whiteheads": 193, "Blackheads": 735, "Papules": 621, "Pustules": 584, "Cyst": 645},
        "valid": {"Whiteheads": 49, "Blackheads": 240, "Papules": 209, "Pustules": 217, "Cyst": 206},
        "test": {"Whiteheads": 57, "Blackheads": 265, "Papules": 202, "Pustules": 205, "Cyst": 189},
    },
}

# split -> expected (images, labels) counts for the YOLO detection set
EXPECTED_YOLO = {"train": 823, "valid": 56, "test": 48}


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return -1
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    errors = 0
    warnings = 0

    for dataset, splits in EXPECTED_CLASSIFICATION.items():
        for split, classes in splits.items():
            for cls, expected in classes.items():
                folder = RAW_DIR / dataset / split / cls
                n = count_images(folder)
                if n < 0:
                    print(f"[ERROR] missing: {folder.relative_to(REPO_ROOT)}")
                    errors += 1
                elif n == 0:
                    print(f"[ERROR] empty: {folder.relative_to(REPO_ROOT)}")
                    errors += 1
                elif n != expected:
                    print(f"[warn] {dataset}/{split}/{cls}: {n} images (expected {expected})")
                    warnings += 1

    for split, expected in EXPECTED_YOLO.items():
        images = count_images(RAW_DIR / "acne_yolo" / split / "images")
        labels_dir = RAW_DIR / "acne_yolo" / split / "labels"
        labels = sum(1 for f in labels_dir.iterdir() if f.suffix == ".txt") if labels_dir.is_dir() else -1
        if images <= 0 or labels <= 0:
            print(f"[ERROR] acne_yolo/{split}: images={images}, labels={labels}")
            errors += 1
        elif images != labels:
            print(f"[ERROR] acne_yolo/{split}: {images} images but {labels} label files")
            errors += 1
        elif images != expected:
            print(f"[warn] acne_yolo/{split}: {images} pairs (expected {expected})")
            warnings += 1

    if errors:
        print(f"\nFAILED: {errors} error(s), {warnings} warning(s). Run scripts/download_data.py first.")
        return 1
    print(f"\nOK: all datasets present ({warnings} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
