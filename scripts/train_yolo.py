"""Retrain the lesion detector and report metrics on the held-out test split.

Two things differ from the notebook version:

1. It trains on Apple Silicon's GPU (MPS) when available, so a longer schedule is
   affordable. The notebook used 40 epochs because that was what fit in a Colab
   session; ultralytics will early-stop here if the model stops improving.
2. It evaluates on the *test* split, not the validation split. Ultralytics reports
   validation mAP at the end of training by default, and validation is what the
   early-stopping decision was made on, so quoting it overstates the model.

Run:
    .venv/bin/python scripts/train_yolo.py --epochs 150
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "data" / "raw" / "acne_yolo" / "data.yaml"
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--tag", default="v2")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    device = pick_device()
    print(f"training {args.model} on {device} for up to {args.epochs} epochs")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=77,
        device=device,
        project=str(RESULTS / "yolo"),
        name=args.tag,
        exist_ok=True,
        verbose=True,
    )

    best = Path(model.trainer.best)
    if not best.exists():
        best = Path(model.trainer.last)

    # the number we quote must come from the split that took no part in training
    # or in early stopping
    trained = YOLO(str(best))
    metrics = trained.val(data=str(DATA_YAML.resolve()), split="test",
                          imgsz=args.imgsz, device=device, verbose=False)

    record = {
        "tag": args.tag,
        "model": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "device": device,
        "test": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "yolo_experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    print("\n=== lesion detector, held-out test split ===")
    for k, v in record["test"].items():
        print(f"  {k:<10} {v:.4f}")

    if args.save:
        MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, MODELS / "acne_yolo.pt")
        print(f"\n[saved] {MODELS / 'acne_yolo.pt'}")


if __name__ == "__main__":
    main()
