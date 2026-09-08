"""Find out WHAT the frozen probe is reading on acne_type.

scripts/probe_backbones.py established that a frozen ImageNet backbone scores
73.5% on acne_type's strict split against a 26.8% baseline, while scoring below
baseline on skin_type. Something is predictable without fine-tuning. That result
says a shortcut exists; it does not say what the shortcut is.

This tests the mechanism directly, which is cheaper and far more decisive than
rebuilding the split. Each condition destroys one kind of information and keeps
the rest, then re-fits the same frozen probe:

  original     everything
  blur         Gaussian blur that destroys lesion texture, keeps colour,
               framing and background
  tiny         downsample to 16x16 and back: only gross colour and layout left
  edges        keep only high-frequency structure, drop colour entirely
  border       the outer frame only, lesion region blanked out

If accuracy holds up under `blur`, `tiny` or `border`, the probe is not reading
lesion morphology, and acne_type's headline number is measuring something else.
If accuracy collapses, the signal really is in the lesions and the shortcut story
is wrong.

Run:
    .venv/bin/python scripts/probe_shortcut.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

import acno_data
import tensorflow as tf
from probe_backbones import baseline
from train_classifier import BACKBONES, IMG_SIZE, set_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# Embedding the whole train split at once needs about 1.7GB of float32, which is
# fine on an idle machine and fails on a busy one. Streaming in batches keeps
# peak memory at a few tens of MB and costs nothing in accuracy: the frozen
# backbone has no cross-sample state.
BATCH = 32


def degrade(images: np.ndarray, mode: str) -> np.ndarray:
    """Apply one destructive transform to a stack of uint8 images."""
    if mode == "original":
        return images
    out = np.empty_like(images)
    for i, img in enumerate(images):
        u8 = img.astype(np.uint8)
        if mode == "blur":
            # kernel large enough that individual lesions become smudges
            out[i] = cv2.GaussianBlur(u8, (0, 0), sigmaX=8)
        elif mode == "tiny":
            small = cv2.resize(u8, (16, 16), interpolation=cv2.INTER_AREA)
            out[i] = cv2.resize(small, u8.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        elif mode == "edges":
            grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
            norm = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
            out[i] = np.dstack([norm, norm, norm])
        elif mode == "border":
            # blank the middle half, where any lesion the label refers to must be
            blanked = u8.copy()
            h, w = blanked.shape[:2]
            blanked[h // 4:3 * h // 4, w // 4:3 * w // 4] = 0
            out[i] = blanked
        else:
            raise ValueError(mode)
    return out


def build_embedder(backbone: str):
    """Frozen backbone as a feature extractor, built once and reused."""
    constructor, preprocess = BACKBONES[backbone]
    base = constructor((IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    y = preprocess()(inputs)
    y = base(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    return tf.keras.Model(inputs, y)


def features(model, images: np.ndarray, mode: str) -> np.ndarray:
    """Degrade, centre-crop and embed in batches, holding little memory at once."""
    out = []
    for start in range(0, len(images), BATCH):
        chunk = degrade(images[start:start + BATCH], mode)
        # centre crop 256 -> 224, matching evaluation in train_classifier.py
        cropped = chunk[:, 16:16 + IMG_SIZE, 16:16 + IMG_SIZE, :].astype(np.float32)
        # Call the model directly rather than through predict(). predict() builds
        # a fresh tf.data pipeline with a prefetch thread on every call, and
        # calling it once per batch in a loop deadlocks: the main thread parks in
        # PrefetchDatasetOp::GetNextInternal waiting for a worker that never
        # runs, with no error and no CPU use. Eager calls have no such pipeline.
        out.append(np.asarray(model(cropped, training=False)))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="acne_type")
    parser.add_argument("--backbone", default="mobilenetv3large")
    parser.add_argument("--modes", nargs="+",
                        default=["original", "blur", "tiny", "edges", "border"])
    args = parser.parse_args()

    set_seeds()
    splits = acno_data.load_splits(args.dataset)
    target = "strict_test" if "strict_test" in splits else "clean_test"
    base = baseline(splits[target]["y"], len(splits["classes"]))

    print(f"\n{args.dataset} / {args.backbone} / evaluated on {target}")
    print(f"majority baseline {base:.1%}  n={len(splits[target]['y'])}\n")
    print(f"{'condition':<12}{'accuracy':>10}{'over baseline':>16}")
    print("-" * 40)

    rows = []
    for mode in args.modes:
        # Embed everything first, then tear TensorFlow down before handing the
        # features to scikit-learn. Holding a live Keras model across a
        # LogisticRegression fit deadlocks on macOS, where TensorFlow and
        # scikit-learn each bring their own OpenMP runtime: the process simply
        # stops with no CPU use and no error. probe_backbones.py avoids this the
        # same way, by calling clear_session() before it fits.
        model = build_embedder(args.backbone)
        train_features = features(model, splits["train"]["x"], mode)
        test_features = features(model, splits[target]["x"], mode)
        del model
        tf.keras.backend.clear_session()

        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(train_features, splits["train"]["y"])
        pred = clf.predict(test_features)
        acc = float((pred == splits[target]["y"]).mean())
        rows.append({"mode": mode, "accuracy": acc, "over_baseline": acc - base})
        print(f"{mode:<12}{acc:>9.1%}{acc - base:>+15.1%}")

    out = RESULTS / f"probe_shortcut_{args.dataset}.json"
    out.write_text(json.dumps(
        {"dataset": args.dataset, "backbone": args.backbone, "split": target,
         "majority_baseline": base, "conditions": rows}, indent=2))
    print(f"\nwrote {out}")
    print("\nReading it: a condition that stays far above baseline is information")
    print("the probe can use WITHOUT the detail that condition destroyed.")


if __name__ == "__main__":
    main()
