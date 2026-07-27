"""Audit the classifier datasets before we trust any accuracy number.

Checks three things that would each explain a bad test score for a different reason:

1. Duplicate / leaked images across train, valid and test (inflates scores, or means
   the splits are not really independent).
2. Baselines: what accuracy does "always guess the biggest class" get? A model below
   that line has learned nothing useful.
3. Image properties per split (size, aspect, brightness) so we can see whether the
   test split simply looks different from the training split.

Run:
    .venv/bin/python scripts/audit_data.py
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "raw"
RESULTS = REPO_ROOT / "results"
SPLITS = ["train", "valid", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(split_dir: Path):
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_dir.name, path


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def pixel_hash(path: Path) -> str:
    """Hash the decoded pixels at low resolution.

    Catches near-duplicates that survived a re-encode, which byte hashing misses.
    """
    with Image.open(path) as img:
        small = img.convert("L").resize((16, 16), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
    # binary gradient signature, robust to global brightness shifts
    sig = (arr > arr.mean()).astype(np.uint8).tobytes()
    return hashlib.md5(sig).hexdigest()


def audit(name: str) -> dict:
    root = DATA / name
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    counts: dict[str, Counter] = {}
    byte_hashes: dict[str, dict[str, Path]] = {}
    pix_hashes: dict[str, dict[str, Path]] = {}
    stats: dict[str, dict] = {}

    for split in SPLITS:
        split_dir = root / split
        counts[split] = Counter()
        byte_hashes[split] = {}
        pix_hashes[split] = {}
        widths, heights, brightness = [], [], []

        for class_name, path in iter_images(split_dir):
            counts[split][class_name] += 1
            byte_hashes[split][file_hash(path)] = path
            pix_hashes[split][pixel_hash(path)] = path
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
                if len(brightness) < 400:  # sample, decoding everything is slow
                    brightness.append(float(np.asarray(img.convert("L")).mean()))

        total = sum(counts[split].values())
        majority = max(counts[split].values()) / total if total else 0.0
        stats[split] = {
            "n": total,
            "classes": dict(counts[split]),
            "majority_baseline": round(majority, 4),
            "median_width": int(np.median(widths)) if widths else 0,
            "median_height": int(np.median(heights)) if heights else 0,
            "mean_brightness": round(float(np.mean(brightness)), 1) if brightness else 0,
        }
        print(f"\n[{split}] {total} images across {len(counts[split])} classes")
        for cls, n in sorted(counts[split].items()):
            print(f"    {cls:<12} {n:>5}  ({n / total:.1%})")
        print(f"    majority-class baseline: {majority:.1%}")
        print(f"    median size {stats[split]['median_width']}x{stats[split]['median_height']}"
              f", mean brightness {stats[split]['mean_brightness']}")

    # cross-split overlap
    print("\n--- overlap between splits ---")
    overlaps = {}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        exact = set(byte_hashes[a]) & set(byte_hashes[b])
        near = set(pix_hashes[a]) & set(pix_hashes[b])
        overlaps[f"{a}-{b}"] = {"exact": len(exact), "near": len(near)}
        flag = "  <-- LEAKAGE" if exact or near else ""
        print(f"    {a:>5} vs {b:<5}  exact {len(exact):>4}   near {len(near):>4}{flag}")

    # duplicates inside train
    print("\n--- duplicates within train ---")
    within = defaultdict(list)
    for class_name, path in iter_images(root / "train"):
        within[pixel_hash(path)].append(class_name)
    dup_groups = {h: c for h, c in within.items() if len(c) > 1}
    conflicting = {h: c for h, c in dup_groups.items() if len(set(c)) > 1}
    print(f"    {len(dup_groups)} near-duplicate groups")
    print(f"    {len(conflicting)} of them span DIFFERENT classes (contradictory labels)")

    return {
        "splits": stats,
        "overlaps": overlaps,
        "train_duplicate_groups": len(dup_groups),
        "train_contradictory_groups": len(conflicting),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {name: audit(name) for name in ["skin_type", "acne_type"]}
    out = RESULTS / "data_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
