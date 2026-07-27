"""Work out why skin_type barely beats guessing, before spending hours training it.

The backbone probe showed every ImageNet backbone stuck near the majority-class
baseline on this dataset. That is unusual, and it means one of three things:

  (a) the labels are noisy, so no model can do much better
  (b) the useful signal is fine skin texture that 224px pooling throws away
  (c) the task is separable but only for some classes

This script tests all three cheaply, using cached embeddings so each experiment
takes seconds instead of an epoch:

  1. contradictions: identical images that carry different labels
  2. resolution: does embedding at a larger input size help?
  3. task shape: is any one-vs-rest split easier than the full 3-way problem?
  4. shortcut check: how well can the model do from the image border alone, where
     there is no skin? Anything well above baseline means it is reading layout,
     watermarks or backgrounds rather than skin.

Run:
    .venv/bin/python scripts/diagnose_skin_type.py
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import acno_data
from train_classifier import BACKBONES, set_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CACHE = REPO_ROOT / "data" / "cache"
DATASET = "skin_type"


def contradictions() -> dict:
    """Identical images inside train that carry different class labels."""
    print(f"\n{'=' * 78}\n1. label contradictions\n{'=' * 78}")
    groups = defaultdict(list)
    for class_name, path in acno_data.iter_images(acno_data.DATA / DATASET / "train"):
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L").resize((16, 16), Image.BILINEAR),
                             dtype=np.float32)
        sig = hashlib.md5((arr > arr.mean()).astype(np.uint8).tobytes()).hexdigest()
        groups[sig].append((class_name, str(path.relative_to(REPO_ROOT))))

    conflict = {s: g for s, g in groups.items() if len({c for c, _ in g}) > 1}
    print(f"    {len(conflict)} groups of identical images carry more than one label")
    for sig, g in list(conflict.items())[:5]:
        print(f"      - {' vs '.join(sorted({c for c, _ in g}))}")
        for cls, p in g[:3]:
            print(f"          [{cls}] {p}")
    return {"contradictory_groups": len(conflict)}


def embed_at(backbone: str, size: int, x_cache: np.ndarray, batch: int = 32) -> np.ndarray:
    """Embed images at `size`.

    x_cache must already be decoded at that resolution or higher. Upsampling a small
    cache would add no detail and would make the resolution comparison meaningless.
    """
    constructor, preprocess = BACKBONES[backbone]
    base = constructor((size, size, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3))
    y = preprocess()(inputs)
    y = base(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    model = tf.keras.Model(inputs, y)

    if x_cache.shape[1] != size:
        x_cache = tf.image.resize(x_cache.astype(np.float32), (size, size)).numpy()
    feats = model.predict(x_cache.astype(np.float32), batch_size=batch, verbose=0)
    tf.keras.backend.clear_session()
    return feats


def fit_score(train_x, train_y, evals: dict) -> dict:
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(train_x, train_y)
    out = {}
    for name, (ex, ey) in evals.items():
        pred = clf.predict(ex)
        out[name] = {
            "accuracy": float((pred == ey).mean()),
            "macro_f1": float(f1_score(ey, pred, average="macro", zero_division=0)),
            "baseline": float(np.bincount(ey).max() / len(ey)),
        }
    return out


def resolution_sweep(splits) -> dict:
    """Skin texture is high-frequency detail. If 224px is throwing it away, decoding
    the originals at a larger size should show a clear gain."""
    print(f"\n{'=' * 78}\n2. does more resolution help?\n{'=' * 78}")
    print(f"{'backbone':<18}{'size':>6}{'clean acc':>12}{'baseline':>11}{'macro F1':>11}")
    print("-" * 78)

    leaked = acno_data.leaked_test_paths(DATASET)
    _, _, test_paths = acno_data.build_cache(DATASET, "test")
    keep = np.array([p not in leaked for p in test_paths])

    rows = []
    for size in [224, 320, 384]:
        train_x, train_y = acno_data.build_cache_at(DATASET, "train", size)
        test_x, test_y = acno_data.build_cache_at(DATASET, "test", size)
        test_x, test_y = test_x[keep], test_y[keep]
        for backbone in ["efficientnetv2b1", "resnet50v2"]:
            feats_tr = embed_at(backbone, size, train_x)
            feats_te = embed_at(backbone, size, test_x)
            c = fit_score(feats_tr, train_y, {"e": (feats_te, test_y)})["e"]
            rows.append({"backbone": backbone, "size": size, **c})
            print(f"{backbone:<18}{size:>6}{c['accuracy']:>12.3f}"
                  f"{c['baseline']:>11.3f}{c['macro_f1']:>11.3f}")
    return {"resolution": rows}


def task_shape(splits, backbone="efficientnetv2b1", size=320) -> dict:
    print(f"\n{'=' * 78}\n3. is any sub-task easier than the full 3-way problem?\n{'=' * 78}")
    feats = {s: embed_at(backbone, size, splits[s]["x"])
             for s in ["train", "clean_test"]}
    classes = splits["classes"]
    rows = []

    tasks = {"3-way (dry/normal/oily)": None}
    for i, name in enumerate(classes):
        tasks[f"{name} vs rest"] = i

    print(f"{'task':<26}{'clean acc':>12}{'baseline':>11}{'macro F1':>11}{'lift':>9}")
    print("-" * 78)
    for label, positive in tasks.items():
        if positive is None:
            ty, ey = splits["train"]["y"], splits["clean_test"]["y"]
        else:
            ty = (splits["train"]["y"] == positive).astype(int)
            ey = (splits["clean_test"]["y"] == positive).astype(int)
        scores = fit_score(feats["train"], ty, {"e": (feats["clean_test"], ey)})["e"]
        lift = scores["accuracy"] - scores["baseline"]
        rows.append({"task": label, **scores, "lift": lift})
        print(f"{label:<26}{scores['accuracy']:>12.3f}{scores['baseline']:>11.3f}"
              f"{scores['macro_f1']:>11.3f}{lift:>+9.3f}")
    return {"task_shape": rows}


def shortcut_check(splits, backbone="efficientnetv2b1") -> dict:
    """Train on the image border only. Skin is mostly central; the border is
    background, watermarks and padding. Accuracy here is accuracy from shortcuts."""
    print(f"\n{'=' * 78}\n4. how much can be predicted WITHOUT looking at skin?\n{'=' * 78}")

    def border_only(x):
        out = x.copy()
        h, w = out.shape[1:3]
        out[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4, :] = 0  # blank the centre
        return out

    def centre_only(x):
        out = np.zeros_like(x)
        h, w = x.shape[1:3]
        out[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4, :] = \
            x[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4, :]
        return out

    rows = []
    for label, fn in [("full image", lambda x: x),
                      ("centre only (skin)", centre_only),
                      ("border only (no skin)", border_only)]:
        feats = {s: embed_at(backbone, 224, fn(splits[s]["x"]))
                 for s in ["train", "clean_test"]}
        scores = fit_score(feats["train"], splits["train"]["y"],
                           {"e": (feats["clean_test"], splits["clean_test"]["y"])})["e"]
        rows.append({"region": label, **scores})
        print(f"    {label:<24} clean acc {scores['accuracy']:.3f}  "
              f"(baseline {scores['baseline']:.3f})")
    print("\n    If 'border only' scores well above baseline, the dataset leaks the")
    print("    answer through backgrounds, watermarks and framing rather than skin.")
    return {"shortcut": rows}


def main() -> None:
    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    splits = acno_data.load_splits(DATASET)

    report = {}
    report.update(contradictions())
    report.update(resolution_sweep(splits))
    report.update(task_shape(splits))
    report.update(shortcut_check(splits))

    out = RESULTS / "skin_type_diagnosis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
