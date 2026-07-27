"""Evaluate a saved classifier, with and without flip test-time augmentation.

Test-time augmentation runs the image twice, once normally and once mirrored, and
averages the two predictions. Faces are roughly symmetric, so a mirrored face is still
a valid face, and averaging two views cancels some of the model's noise. It costs one
extra forward pass and needs no retraining.

It can also be baked into the exported model with --export-tta, which builds a single
graph containing both branches. The browser then gets the benefit without any change
to web/lib/analyze.ts, since the input and output contract stays identical.

Run:
    .venv/bin/python scripts/evaluate_model.py --dataset skin_type --tag face_mnv2_light
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

import acno_data
from train_classifier import IMG_SIZE, evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy estimate.

    The clean skin_type test split is only 119 images. At that size a single accuracy
    figure carries roughly nine points of uncertainty in each direction, which is wide
    enough to change what conclusion you draw, so we always print the interval.
    """
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def build_tta(model: tf.keras.Model) -> tf.keras.Model:
    """Average the model's prediction on the image and its mirror."""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    flipped = tf.keras.layers.Lambda(
        lambda t: tf.image.flip_left_right(t), name="mirror")(inputs)
    averaged = tf.keras.layers.Average(name="probs")([model(inputs), model(flipped)])
    return tf.keras.Model(inputs, averaged, name=f"{model.name}_tta")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["skin_type", "acne_type"])
    p.add_argument("--tag", required=True, help="run tag, or 'current' for the live model")
    p.add_argument("--variant", default="whole", choices=["whole", "face"])
    p.add_argument("--export-tta", action="store_true",
                   help="save the TTA version over models/<dataset>.keras")
    args = p.parse_args()

    path = (MODELS / f"{args.dataset}.keras" if args.tag == "current"
            else MODELS / f"{args.dataset}__{args.tag}.keras")
    if not path.exists():
        raise SystemExit(f"no weights at {path}")

    model = tf.keras.models.load_model(path)
    tta = build_tta(model)
    splits = acno_data.load_splits(args.dataset, variant=args.variant)
    classes = splits["classes"]
    eval_splits = [s for s in ["valid", "test", "clean_test", "strict_test"] if s in splits]

    print(f"\n{path.name}   variant={args.variant}")
    print(f"{'split':<13}{'n':>6}{'plain':>8}{'TTA':>8}{'TTA 95% CI':>18}"
          f"{'baseline':>10}{'TTA F1':>9}")
    print("-" * 72)

    out = {}
    for split in eval_splits:
        x, y = splits[split]["x"], splits[split]["y"]
        plain = evaluate(model, x, y, classes)
        with_tta = evaluate(tta, x, y, classes)
        n = len(y)
        lo, hi = wilson_interval(int(round(with_tta["accuracy"] * n)), n)
        baseline = float(np.bincount(y, minlength=len(classes)).max() / n)
        out[split] = {
            "n": n,
            "majority_baseline": baseline,
            "plain": {k: v for k, v in plain.items() if k != "predictions"},
            "tta": {k: v for k, v in with_tta.items() if k != "predictions"},
            "tta_accuracy_ci95": [lo, hi],
        }
        print(f"{split:<13}{n:>6}{plain['accuracy']:>8.3f}{with_tta['accuracy']:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{baseline:>10.3f}"
              f"{with_tta['macro_f1']:>9.3f}")

    best = "clean_test" if "clean_test" in out else eval_splits[-1]
    print(f"\nper-class report on {best} (with TTA):")
    x, y = splits[best]["x"], splits[best]["y"]
    pred = np.array(evaluate(tta, x, y, classes)["predictions"])
    print(classification_report(y, pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"eval_{args.dataset}_{args.tag}.json").write_text(json.dumps(out, indent=2))

    if args.export_tta:
        dest = MODELS / f"{args.dataset}.keras"
        tta.save(dest)
        print(f"[promoted] TTA model -> {dest}")


if __name__ == "__main__":
    main()
