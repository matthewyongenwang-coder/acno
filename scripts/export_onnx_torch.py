"""Export a PyTorch-trained Acno classifier to ONNX for the browser.

The exported graph must match what web/lib/analyze.ts already sends and expects:

    input   "image"  float32  (1, 224, 224, 3)  NHWC, values 0-255
    output  "probs"  float32  (1, num_classes)  softmax, alphabetical class order

Normalisation and softmax are inside the graph, so nothing in the web app changes.

With --tta the graph runs the image and its mirror and averages the two, which is one
extra forward pass in exchange for a small accuracy gain. Everything else stays the
same from the browser's point of view.

Run:
    .venv/bin/python scripts/export_onnx_torch.py --dataset skin_type --tag t_face_resnet18_med
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import acno_data
from train_torch import IMG_SIZE, AcnoNet

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
WEB_MODELS = REPO_ROOT / "web" / "public" / "models"
SIZE_BUDGET_MB = 40.0


class MirrorAverage(nn.Module):
    """Average the model's output on the image and its left-right mirror."""

    def __init__(self, model: AcnoNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (self.model(x) + self.model(torch.flip(x, dims=[2]))) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["skin_type", "acne_type"])
    p.add_argument("--tag", required=True)
    p.add_argument("--variant", default="whole", choices=["whole", "face"])
    p.add_argument("--tta", action="store_true", help="bake mirror averaging into the graph")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt_path = MODELS / f"{args.dataset}__{args.tag}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = AcnoNet(ckpt["backbone"], len(ckpt["classes"]), ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.apply_softmax = True   # the browser expects probabilities
    model.eval()

    exportable = MirrorAverage(model) if args.tta else model
    exportable.eval()

    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    out = WEB_MODELS / f"{args.dataset}.onnx"
    dummy = torch.zeros(1, IMG_SIZE, IMG_SIZE, 3)

    torch.onnx.export(
        exportable, dummy, str(out),
        input_names=["image"], output_names=["probs"],
        opset_version=args.opset, dynamo=False,
    )

    # verify against real images, not noise: a wrong normalisation can still look
    # right on random input while being badly wrong on photographs
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    splits = acno_data.load_splits(args.dataset, variant=args.variant)
    sample = splits["test"]["x"][:24, 16:16 + IMG_SIZE, 16:16 + IMG_SIZE, :].astype(np.float32)

    worst, disagree = 0.0, 0
    with torch.no_grad():
        for image in sample:
            batch = image[None, ...]
            torch_out = exportable(torch.from_numpy(batch)).numpy()
            onnx_out = sess.run(None, {"image": batch})[0]
            worst = max(worst, float(np.abs(torch_out - onnx_out).max()))
            disagree += int(torch_out.argmax() != onnx_out.argmax())

    sums = sess.run(None, {"image": sample[:1]})[0].sum()
    size_mb = out.stat().st_size / 1e6

    assert worst < 1e-4, f"onnx diverges from torch (max diff {worst})"
    assert disagree == 0, f"{disagree} of {len(sample)} predictions disagree"
    assert abs(sums - 1.0) < 1e-3, f"output is not a probability distribution (sums to {sums})"

    print(f"[ok] {out.name}")
    print(f"     classes  {ckpt['classes']}")
    print(f"     size     {size_mb:.1f} MB"
          f"{'  <-- OVER BUDGET' if size_mb > SIZE_BUDGET_MB else ''}")
    print(f"     max diff {worst:.2e} over {len(sample)} real images, all agree")
    print(f"     tta      {args.tta}")

    total = sum(q.stat().st_size for q in WEB_MODELS.glob("*.onnx")) / 1e6
    print(f"\ntotal browser download now {total:.1f} MB")

    meta = WEB_MODELS / "models.json"
    record = json.loads(meta.read_text()) if meta.exists() else {}
    record[args.dataset] = {"classes": ckpt["classes"], "backbone": ckpt["backbone"],
                            "tag": args.tag, "variant": args.variant, "tta": args.tta}
    meta.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
