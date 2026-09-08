"""Find leaked test images the hash check cannot see.

scripts/verify_duplicates.py compares images by a downscaled pixel signature. That
catches re-encodes and exact copies, but the acne_type dataset was augmented by its
author *before* being split, so the same source photo appears as rotated, cropped and
mosaicked variants spread across train, valid and test. A 30-degree rotation of an
image does not match its original under any pixel hash, yet a model that memorised
one will recognise the other.

So we compare in feature space instead. Embed every image with a frozen ImageNet
backbone, L2-normalise, and for each test image find its nearest training neighbour by
cosine similarity. Near-duplicates sit far above the similarity of genuinely unrelated
images, so the histogram separates and we can pick a threshold and check it by eye.

Writes results/leak_scan.json, which acno_data.load_splits() uses to build the strict
evaluation split.

Run:
    .venv/bin/python scripts/leak_scan.py --dataset acne_type
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

import acno_data
from train_classifier import BACKBONES, IMG_SIZE, set_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
# Contact sheets go here. This used to be a hardcoded absolute path inside one
# Claude Code session's scratch directory, which no longer exists: the sheets
# were written somewhere nobody would ever look, silently defeating the
# "always eyeball the contact sheets before trusting a number" rule that both
# the leak scan and the fairness review depend on. Keep it in the repo, under
# the already-gitignored results/ tree, so it survives across sessions.
SCRATCH = RESULTS / "qa"


def embed(backbone: str, x: np.ndarray, batch: int = 48) -> np.ndarray:
    constructor, preprocess = BACKBONES[backbone]
    base = constructor((IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    y = preprocess()(inputs)
    y = base(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    model = tf.keras.Model(inputs, y)

    resized = tf.image.resize(x.astype(np.float32), (IMG_SIZE, IMG_SIZE)).numpy()
    feats = model.predict(resized, batch_size=batch, verbose=0)
    tf.keras.backend.clear_session()
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    return feats


def contact_sheet(splits, pairs, out_path, n=12):
    """Save test/neighbour image pairs so a human can confirm the threshold."""
    rows = []
    for test_i, train_i, sim in pairs[:n]:
        a = splits["test"]["x"][test_i]
        b = splits["train"]["x"][train_i]
        gap = np.full((a.shape[0], 8, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([a, gap, b], axis=1))
    if not rows:
        return
    sheet = np.concatenate(rows[: n // 2], axis=0)
    if len(rows) > n // 2:
        right = np.concatenate(rows[n // 2:], axis=0)
        h = min(sheet.shape[0], right.shape[0])
        sheet = np.concatenate([sheet[:h], right[:h]], axis=1)
    Image.fromarray(sheet).save(out_path)
    print(f"    contact sheet -> {out_path}")


def scan(dataset: str, backbone: str, threshold: float) -> dict:
    splits = acno_data.load_splits(dataset)
    print(f"\n{'=' * 78}\n{dataset}: nearest-neighbour leak scan ({backbone})\n{'=' * 78}")

    ref = np.concatenate([splits["train"]["x"], splits["valid"]["x"]])
    ref_feats = embed(backbone, ref)
    test_feats = embed(backbone, splits["test"]["x"])

    sims = test_feats @ ref_feats.T
    best = sims.max(axis=1)
    best_idx = sims.argmax(axis=1)

    for q in [50, 75, 90, 95, 99]:
        print(f"    similarity p{q}: {np.percentile(best, q):.3f}")
    for t in [0.80, 0.85, 0.90, 0.95, 0.98]:
        print(f"    would flag at >{t:.2f}: {(best > t).sum():>4} of {len(best)} "
              f"({(best > t).mean():.1%})")

    flagged = np.where(best > threshold)[0]
    order = flagged[np.argsort(-best[flagged])]
    n_train = len(splits["train"]["x"])
    pairs = [(int(i), int(best_idx[i]), float(best[i]))
             for i in order if best_idx[i] < n_train]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    contact_sheet(splits, pairs, SCRATCH / f"{dataset}_leak_pairs.png")

    leaked_paths = sorted({splits["test"]["paths"][i] for i in flagged})
    print(f"\n    threshold {threshold}: {len(leaked_paths)} of {len(best)} test images "
          f"flagged ({len(leaked_paths) / len(best):.1%})")
    print(f"    strict test split would keep {len(best) - len(leaked_paths)} images")

    return {
        "dataset": dataset,
        "backbone": backbone,
        "threshold": threshold,
        "test_total": int(len(best)),
        "flagged": len(leaked_paths),
        "strict_test_paths": [p for p in splits["test"]["paths"]
                              if p not in set(leaked_paths)],
        "leaked_paths": leaked_paths,
        "similarity_percentiles": {str(q): float(np.percentile(best, q))
                                   for q in [50, 75, 90, 95, 99]},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--backbone", default="efficientnetv2b1")
    p.add_argument("--threshold", type=float, default=0.92)
    args = p.parse_args()

    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = ["skin_type", "acne_type"] if args.dataset == "all" else [args.dataset]

    out_file = RESULTS / "leak_scan.json"
    report = json.loads(out_file.read_text()) if out_file.exists() else {}
    for name in datasets:
        report[name] = scan(name, args.backbone, args.threshold)
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
