"""Cheap survey of which ImageNet backbone actually separates our skin classes.

Full fine-tuning every candidate would take hours on a laptop CPU. Instead we run
each frozen backbone over the data once, keep the pooled feature vectors, and fit a
logistic-regression head on them in seconds. That is not the final accuracy, but it
ranks the backbones reliably and costs one forward pass each.

The linear-probe score is also a useful sanity floor: if no backbone can separate the
classes even a little, the problem is the labels, not the training recipe.

Run:
    .venv/bin/python scripts/probe_backbones.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import acno_data
from train_classifier import BACKBONES, IMG_SIZE, set_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

CANDIDATES = [
    "mobilenetv2",
    "mobilenetv3large",
    "efficientnetv2b0",
    "efficientnetv2b1",
    "resnet50v2",
    "densenet121",
]


def embed(backbone: str, x: np.ndarray, batch: int = 64) -> np.ndarray:
    constructor, preprocess = BACKBONES[backbone]
    base = constructor((IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    y = preprocess()(inputs)
    y = base(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    model = tf.keras.Model(inputs, y)

    # centre crop 256 -> 224, matching evaluation in train_classifier.py
    cropped = x[:, 16:16 + IMG_SIZE, 16:16 + IMG_SIZE, :].astype(np.float32)
    feats = model.predict(cropped, batch_size=batch, verbose=0)
    tf.keras.backend.clear_session()
    return feats


def baseline(y: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes)
    return float(counts.max() / counts.sum())


def probe(dataset: str, backbones: list[str]) -> list[dict]:
    splits = acno_data.load_splits(dataset)
    classes = splits["classes"]
    rows = []

    # strict_test only exists once scripts/leak_scan.py has run. It is the split
    # that matters most here: clean_test still contains feature-space duplicates,
    # so a high probe score there proves nothing. A high score on strict_test,
    # where the near-identical copies are gone, means the frozen features are
    # reading something other than the lesions.
    evaluated = ["valid", "test", "clean_test"]
    if "strict_test" in splits:
        evaluated.append("strict_test")

    print(f"\n{'=' * 88}\n{dataset}  ({len(classes)} classes: {', '.join(classes)})\n{'=' * 88}")
    for split in evaluated:
        print(f"{split:<12} n={len(splits[split]['y']):<6} "
              f"majority baseline {baseline(splits[split]['y'], len(classes)):.1%}")
    print()
    header = f"{'backbone':<18}{'dim':>6}{'valid':>9}{'test':>9}{'clean':>9}"
    if "strict_test" in splits:
        header += f"{'strict':>9}"
    print(header)
    print("-" * 88)

    for name in backbones:
        feats = {s: embed(name, splits[s]["x"]) for s in ["train"] + evaluated}
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(feats["train"], splits["train"]["y"])

        scores = {}
        for split in evaluated:
            pred = clf.predict(feats[split])
            scores[split] = {
                "accuracy": float((pred == splits[split]["y"]).mean()),
                "macro_f1": float(f1_score(splits[split]["y"], pred,
                                           average="macro", zero_division=0)),
                "n": int(len(splits[split]["y"])),
                "majority_baseline": baseline(splits[split]["y"], len(classes)),
            }
        rows.append({"dataset": dataset, "backbone": name,
                     "feature_dim": int(feats["train"].shape[1]), "scores": scores})
        line = (f"{name:<18}{feats['train'].shape[1]:>6}"
                f"{scores['valid']['accuracy']:>9.3f}"
                f"{scores['test']['accuracy']:>9.3f}"
                f"{scores['clean_test']['accuracy']:>9.3f}")
        if "strict_test" in scores:
            line += f"{scores['strict_test']['accuracy']:>9.3f}"
        print(line)

    if "strict_test" in evaluated:
        best = max(rows, key=lambda r: r["scores"]["strict_test"]["accuracy"])
        base = best["scores"]["strict_test"]["majority_baseline"]
        acc = best["scores"]["strict_test"]["accuracy"]
        print(f"\nbest frozen probe on strict_test: {best['backbone']} at {acc:.1%} "
              f"against a {base:.1%} majority baseline "
              f"({acc - base:+.1%} over baseline)")
        print("A frozen ImageNet probe should sit near baseline on a task that needs")
        print("lesion morphology. Well above it means a non-lesion shortcut survives")
        print("the strict split, and the fine-tuned number is suspect.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["skin_type", "acne_type"], action="append",
                        help="repeatable; defaults to both")
    parser.add_argument("--backbones", nargs="+", default=CANDIDATES,
                        help=f"subset of {CANDIDATES}")
    parser.add_argument("--out", default="backbone_probe.json",
                        help="filename under results/")
    args = parser.parse_args()

    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset in args.dataset or ["skin_type", "acne_type"]:
        all_rows.extend(probe(dataset, args.backbones))
    out = RESULTS / args.out
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
