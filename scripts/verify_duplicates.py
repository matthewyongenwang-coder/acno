"""Confirm that the near-duplicate hits from audit_data.py are real duplicates.

The audit uses a 16x16 average-hash signature, which is fast but can in principle
collide on two genuinely different images. Before we let that number change how we
report accuracy, re-check every hit the expensive way: decode both images at 64x64
and measure the actual pixel difference.

Writes results/duplicate_pairs.json listing every confirmed train<->test duplicate,
which train_classifier.py then uses to build a leak-free evaluation split.

Run:
    .venv/bin/python scripts/verify_duplicates.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "raw"
RESULTS = REPO_ROOT / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# mean absolute difference (0-255 scale) below which we call two images duplicates
DUP_MAD_THRESHOLD = 6.0


def iter_images(split_dir: Path):
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_dir.name, path


def signature(path: Path) -> str:
    with Image.open(path) as img:
        small = img.convert("L").resize((16, 16), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
    return hashlib.md5((arr > arr.mean()).astype(np.uint8).tobytes()).hexdigest()


def thumb(path: Path, cache: dict) -> np.ndarray:
    if path not in cache:
        with Image.open(path) as img:
            small = img.convert("RGB").resize((64, 64), Image.BILINEAR)
        cache[path] = np.asarray(small, dtype=np.float32)
    return cache[path]


def verify(name: str) -> dict:
    root = DATA / name
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    # signature -> list of (split, class, path)
    index: dict[str, list[tuple[str, str, Path]]] = {}
    for split in ["train", "valid", "test"]:
        for class_name, path in iter_images(root / split):
            index.setdefault(signature(path), []).append((split, class_name, path))

    cache: dict = {}
    confirmed = {"train-valid": [], "train-test": [], "valid-test": []}
    rejected = 0

    for entries in index.values():
        splits_present = {e[0] for e in entries}
        if len(splits_present) < 2:
            continue
        for i, (split_a, cls_a, path_a) in enumerate(entries):
            for split_b, cls_b, path_b in entries[i + 1:]:
                if split_a == split_b:
                    continue
                pair_key = "-".join(sorted([split_a, split_b], key=
                                           ["train", "valid", "test"].index))
                mad = float(np.abs(thumb(path_a, cache) - thumb(path_b, cache)).mean())
                if mad <= DUP_MAD_THRESHOLD:
                    confirmed[pair_key].append({
                        "a": str(path_a.relative_to(REPO_ROOT)),
                        "b": str(path_b.relative_to(REPO_ROOT)),
                        "a_class": cls_a,
                        "b_class": cls_b,
                        "mad": round(mad, 2),
                        "same_class": cls_a == cls_b,
                    })
                else:
                    rejected += 1

    summary = {}
    for pair, hits in confirmed.items():
        cross_label = sum(1 for h in hits if not h["same_class"])
        summary[pair] = {"confirmed": len(hits), "different_class": cross_label}
        print(f"    {pair:<12} {len(hits):>4} confirmed duplicates"
              f"   ({cross_label} of them labelled differently in each split)")
    print(f"    {rejected} signature hits rejected as false positives")

    # how much of the test split is compromised
    test_paths = {str(p.relative_to(REPO_ROOT)) for _, p in iter_images(root / "test")}
    leaked = set()
    for h in confirmed["train-test"] + confirmed["valid-test"]:
        for side in (h["a"], h["b"]):
            if side in test_paths:
                leaked.add(side)
    pct = len(leaked) / len(test_paths) if test_paths else 0
    print(f"\n    {len(leaked)} of {len(test_paths)} test images ({pct:.1%}) "
          f"also appear in train or valid")

    return {
        "summary": summary,
        "rejected_false_positives": rejected,
        "test_total": len(test_paths),
        "test_leaked": sorted(leaked),
        "test_leaked_fraction": round(pct, 4),
        "pairs": confirmed,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {name: verify(name) for name in ["skin_type", "acne_type"]}
    out = RESULTS / "duplicate_pairs.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
