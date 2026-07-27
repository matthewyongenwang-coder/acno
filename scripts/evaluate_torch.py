"""Evaluate a saved PyTorch checkpoint on every split, including the strict one.

train_torch.py evaluates whatever splits exist at the time it runs. If the
feature-space leak scan is produced afterwards, earlier checkpoints never got measured
against the strict split. This re-scores them without retraining.

Run:
    .venv/bin/python scripts/evaluate_torch.py --dataset acne_type --tag t_acne_mobilenet_v3_large
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

import acno_data
from train_torch import AcnoNet, device, predict, score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["skin_type", "acne_type"])
    p.add_argument("--tag", required=True)
    p.add_argument("--variant", default="whole", choices=["whole", "face"])
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    path = MODELS / f"{args.dataset}__{args.tag}.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    dev = device()
    model = AcnoNet(ckpt["backbone"], len(ckpt["classes"]), ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    splits = acno_data.load_splits(args.dataset, variant=args.variant)
    classes = splits["classes"]
    names = [s for s in ["valid", "test", "clean_test", "strict_test"] if s in splits]

    print(f"\n{path.name}")
    print(f"{'split':<13}{'n':>6}{'acc':>8}{'+TTA':>8}{'TTA 95% CI':>18}"
          f"{'baseline':>10}{'macroF1':>9}")
    print("-" * 72)

    out = {}
    for split in names:
        x, y = splits[split]["x"], splits[split]["y"]
        plain = score(predict(model, x, y, args.batch, dev), y, classes)
        tta = score(predict(model, x, y, args.batch, dev, tta=True), y, classes)
        n = len(y)
        lo, hi = wilson(int(round(tta["accuracy"] * n)), n)
        base = float(np.bincount(y, minlength=len(classes)).max() / n)
        out[split] = {"n": n, "baseline": base,
                      "plain": {k: v for k, v in plain.items() if k != "predictions"},
                      "tta": {k: v for k, v in tta.items() if k != "predictions"},
                      "tta_ci95": [lo, hi]}
        print(f"{split:<13}{n:>6}{plain['accuracy']:>8.3f}{tta['accuracy']:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{base:>10.3f}{tta['macro_f1']:>9.3f}")

    ref = "strict_test" if "strict_test" in out else "clean_test"
    x, y = splits[ref]["x"], splits[ref]["y"]
    pred = np.array(score(predict(model, x, y, args.batch, dev, tta=True),
                          y, classes)["predictions"])
    print(f"\nper-class report on {ref} (with TTA):")
    print(classification_report(y, pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"eval_torch_{args.dataset}_{args.tag}.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
