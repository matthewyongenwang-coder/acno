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


def probe(dataset: str) -> list[dict]:
    splits = acno_data.load_splits(dataset)
    classes = splits["classes"]
    rows = []

    print(f"\n{'=' * 78}\n{dataset}  ({len(classes)} classes: {', '.join(classes)})\n{'=' * 78}")
    counts = np.bincount(splits["clean_test"]["y"], minlength=len(classes))
    print(f"clean-test majority baseline: {counts.max() / counts.sum():.1%}\n")
    print(f"{'backbone':<18}{'dim':>6}{'valid':>9}{'test':>9}{'clean':>9}{'cleanF1':>10}")
    print("-" * 78)

    for name in CANDIDATES:
        feats = {s: embed(name, splits[s]["x"]) for s in
                 ["train", "valid", "test", "clean_test"]}
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(feats["train"], splits["train"]["y"])

        scores = {}
        for split in ["valid", "test", "clean_test"]:
            pred = clf.predict(feats[split])
            scores[split] = {
                "accuracy": float((pred == splits[split]["y"]).mean()),
                "macro_f1": float(f1_score(splits[split]["y"], pred,
                                           average="macro", zero_division=0)),
            }
        rows.append({"dataset": dataset, "backbone": name,
                     "feature_dim": int(feats["train"].shape[1]), "scores": scores})
        print(f"{name:<18}{feats['train'].shape[1]:>6}"
              f"{scores['valid']['accuracy']:>9.3f}"
              f"{scores['test']['accuracy']:>9.3f}"
              f"{scores['clean_test']['accuracy']:>9.3f}"
              f"{scores['clean_test']['macro_f1']:>10.3f}")

    best = max(rows, key=lambda r: r["scores"]["clean_test"]["accuracy"])
    print(f"\nbest linear probe: {best['backbone']} "
          f"at {best['scores']['clean_test']['accuracy']:.1%} clean-test accuracy")
    return rows


def main() -> None:
    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset in ["skin_type", "acne_type"]:
        all_rows.extend(probe(dataset))
    out = RESULTS / "backbone_probe.json"
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
