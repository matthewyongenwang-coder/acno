"""Train an Acno classifier on the Mac GPU (MPS).

Same data, same splits and same reporting as train_classifier.py, but in PyTorch so it
runs on Apple Silicon's GPU instead of the CPU. On an M3 Pro a full fine-tune is about
five times faster, which is what makes a real hyperparameter search practical.

The exported model keeps the exact contract web/lib/analyze.ts already depends on:

    input   "image"  float32  (1, 224, 224, 3)  NHWC, values 0-255
    output  "probs"  float32  (1, num_classes)  softmax, alphabetical class order

Normalisation lives inside the graph, so the browser still sends raw pixels and needs
no change.

Run:
    .venv/bin/python scripts/train_torch.py --dataset skin_type --backbone mobilenet_v3_large
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import models as tvm
from torchvision.transforms import v2

import acno_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"
LEDGER = RESULTS / "experiments.jsonl"

SEED = 77
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# name -> (constructor, weights enum, attribute holding the classifier head)
BACKBONES = {
    "mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.IMAGENET1K_V1, "classifier"),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large,
                           tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier"),
    "efficientnet_b0": (tvm.efficientnet_b0,
                        tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "shufflenet_v2_x1_0": (tvm.shufflenet_v2_x1_0,
                           tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, "fc"),
}


class AcnoNet(nn.Module):
    """Backbone plus head, with preprocessing and softmax inside the graph."""

    def __init__(self, backbone_name: str, num_classes: int, dropout: float):
        super().__init__()
        ctor, weights, head_attr = BACKBONES[backbone_name]
        net = ctor(weights=weights)

        # Replace only the final Linear. MobileNetV3's classifier is
        # Linear(960->1280) -> Hardswish -> Dropout -> Linear(1280->1000), so swapping
        # the whole block would drop a layer the backbone's output shape depends on.
        head = getattr(net, head_attr)
        if isinstance(head, nn.Sequential):
            last_linear = max(i for i, m in enumerate(head) if isinstance(m, nn.Linear))
            head[last_linear] = nn.Linear(head[last_linear].in_features, num_classes)
            for module in head:
                if isinstance(module, nn.Dropout):
                    module.p = dropout
        else:
            setattr(net, head_attr, nn.Sequential(
                nn.Dropout(dropout), nn.Linear(head.in_features, num_classes)))

        self.net = net
        self.head_attr = head_attr
        # Training runs on raw logits so the loss can use log_softmax internally,
        # which is numerically stable. Softmax is switched on only for export, so the
        # ONNX graph still ends in probabilities the way the browser expects.
        self.apply_softmax = False
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def features_parameters(self):
        head = getattr(self.net, self.head_attr)
        head_ids = {id(p) for p in head.parameters()}
        return [p for p in self.net.parameters() if id(p) not in head_ids]

    def head_parameters(self):
        return list(getattr(self.net, self.head_attr).parameters())

    def forward(self, x):
        # x arrives as NHWC in 0-255, exactly what the browser sends.
        # .contiguous() matters: permute leaves the tensor with strides that some
        # backbones' internal .view() calls cannot handle, and it only surfaces in
        # backward once the backbone is unfrozen.
        x = (x.permute(0, 3, 1, 2) / 255.0).contiguous()
        x = (x - self.mean) / self.std
        out = self.net(x)
        return F.softmax(out, dim=1) if self.apply_softmax else out


def build_augment(strength: str):
    if strength == "none":
        return None
    if strength == "light":
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    if strength == "medium":
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])
    if strength == "strong":
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=25, translate=(0.12, 0.12), scale=(0.85, 1.2)),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    if strength == "tone":
        # Geometry as in "medium", but with colour jitter aimed specifically at skin
        # tone. Our datasets are overwhelmingly light-skinned, so the model can learn
        # that a particular skin colour is normal. Shifting hue, saturation and
        # brightness across the range that separates lighter from darker skin pushes it
        # to rely on lesion shape and texture instead of on the colour of the person.
        # Hue stays small: past about 0.05 the skin stops looking like skin at all.
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.35, contrast=0.2, saturation=0.35, hue=0.04),
        ])
    raise ValueError(strength)


def batches(x_u8, y, batch, shuffle, rng, train, augment, dev):
    """Yield batches, cropping 256 -> 224 and augmenting on the GPU."""
    order = rng.permutation(len(y)) if shuffle else np.arange(len(y))
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        chunk = x_u8[idx]

        if train:
            # random crop out of the cached 256px image
            oy, ox = rng.integers(0, chunk.shape[1] - IMG_SIZE + 1, size=2)
        else:
            oy = ox = (chunk.shape[1] - IMG_SIZE) // 2
        chunk = chunk[:, oy:oy + IMG_SIZE, ox:ox + IMG_SIZE, :]

        images = torch.from_numpy(np.ascontiguousarray(chunk)).to(dev)
        if train and augment is not None:
            # Augment while still uint8. torchvision v2 treats a *float* image as
            # being in [0, 1], so passing 0-255 floats makes ColorJitter clamp every
            # pixel to white and the model learns nothing at all.
            images = augment(images.permute(0, 3, 1, 2))
            images = images.permute(0, 2, 3, 1)
        images = images.contiguous().float()
        labels = torch.from_numpy(y[idx]).to(dev).long()
        yield images, labels


@torch.no_grad()
def predict(model, x_u8, y, batch, dev, tta=False):
    model.eval()
    rng = np.random.default_rng(0)
    out = []
    for images, _ in batches(x_u8, y, batch, False, rng, False, None, dev):
        probs = F.softmax(model(images), dim=1)
        if tta:
            mirrored = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
            probs = (probs + mirrored) / 2
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def score(probs, y, classes):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean([
            (pred[y == c] == c).mean() for c in range(len(classes)) if (y == c).any()])),
        "per_class_recall": {
            classes[c]: float((pred[y == c] == c).mean()) if (y == c).any() else None
            for c in range(len(classes))},
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(classes)))).tolist(),
        "predictions": pred.tolist(),
    }


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = device()

    splits = acno_data.load_splits(args.dataset, variant=args.variant)
    classes = splits["classes"]
    n_classes = len(classes)
    xtr, ytr = splits["train"]["x"], splits["train"]["y"]
    xva, yva = splits["valid"]["x"], splits["valid"]["y"]

    if args.dedup_train:
        keep = acno_data.dedup_train_indices(args.dataset, splits["train"]["paths"])
        xtr, ytr = xtr[keep], ytr[keep]

    model = AcnoNet(args.backbone, n_classes, args.dropout).to(dev)
    augment = build_augment(args.augment)

    if args.class_weights:
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1
        w = torch.tensor(len(ytr) / (n_classes * counts), dtype=torch.float32, device=dev)
    else:
        w = None
    # cross_entropy takes logits and does log_softmax internally, and supports both
    # class weights and label smoothing
    def loss_fn(logits, target):
        return F.cross_entropy(logits, target, weight=w,
                               label_smoothing=args.label_smoothing)

    # the head is randomly initialised and needs a much larger step than the
    # pretrained backbone, which only needs nudging
    opt = torch.optim.AdamW([
        {"params": model.features_parameters(), "lr": args.backbone_lr},
        {"params": model.head_parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(ytr) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch * args.warmup_epochs

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best_acc, best_state, best_epoch, since_best = -1.0, None, -1, 0
    started = time.time()
    history = []

    for epoch in range(args.epochs):
        model.train()
        # BatchNorm running statistics come from ImageNet, estimated on far more data
        # than we have. Left in train mode they drift on a few thousand images and the
        # model collapses to predicting one class. Keeping them in eval mode is what
        # the working TensorFlow recipe did, and porting without it was a real bug.
        if args.freeze_bn:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()

        # freeze the backbone for the first few epochs so the random head does not
        # push large gradients through good pretrained weights
        frozen = epoch < args.freeze_epochs
        for p in model.features_parameters():
            p.requires_grad_(not frozen)

        running, seen = 0.0, 0
        for images, labels in batches(xtr, ytr, args.batch, True, rng, True, augment, dev):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach()) * len(labels)
            seen += len(labels)

        val_probs = predict(model, xva, yva, args.batch, dev)
        val_acc = float((val_probs.argmax(axis=1) == yva).mean())
        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        "val_accuracy": val_acc, "frozen": frozen})

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch, since_best = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            since_best += 1

        if args.verbose:
            print(f"  epoch {epoch + 1:>3}/{args.epochs}  loss {running / max(1, seen):.4f}"
                  f"  val_acc {val_acc:.4f}{marker}")

        if since_best >= args.patience:
            print(f"  early stop: no improvement for {args.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - started

    eval_splits = [s for s in ["valid", "test", "clean_test", "strict_test"] if s in splits]
    metrics, metrics_tta = {}, {}
    for split in eval_splits:
        x, y = splits[split]["x"], splits[split]["y"]
        metrics[split] = score(predict(model, x, y, args.batch, dev), y, classes)
        metrics_tta[split] = score(
            predict(model, x, y, args.batch, dev, tta=True), y, classes)

    record = {
        "tag": args.tag,
        "framework": "pytorch",
        "device": str(dev),
        "dataset": args.dataset,
        "variant": args.variant,
        "backbone": args.backbone,
        "img_size": IMG_SIZE,
        "batch": args.batch,
        "augment": args.augment,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "class_weights": bool(args.class_weights),
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "freeze_bn": bool(args.freeze_bn),
        "dedup_train": bool(args.dedup_train),
        "n_train": int(len(ytr)),
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "classes": classes,
        "train_seconds": round(elapsed, 1),
        "params": int(sum(p.numel() for p in model.parameters())),
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                    for k, v in metrics.items()},
        "metrics_tta": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                        for k, v in metrics_tta.items()},
        "history": history,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    MODELS.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS / f"{args.dataset}__{args.tag}.pt"
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                "classes": classes, "dropout": args.dropout}, weights_path)

    print(f"\n=== {args.dataset} / {args.variant} / {args.backbone} / {args.tag} ===")
    print(f"{'split':<13}{'n':>6}{'acc':>8}{'+TTA':>8}{'macroF1':>9}{'baseline':>10}")
    for split in eval_splits:
        y = splits[split]["y"]
        base = float(np.bincount(y, minlength=n_classes).max() / len(y))
        print(f"{split:<13}{len(y):>6}{metrics[split]['accuracy']:>8.3f}"
              f"{metrics_tta[split]['accuracy']:>8.3f}"
              f"{metrics_tta[split]['macro_f1']:>9.3f}{base:>10.3f}")
    print(f"  best epoch {best_epoch + 1}, trained in {elapsed / 60:.1f} min on {dev}")

    ref = "clean_test" if "clean_test" in metrics_tta else eval_splits[-1]
    print(f"\nclean test report (with TTA):")
    print(classification_report(
        splits[ref]["y"], np.array(metrics_tta[ref]["predictions"]),
        labels=list(range(n_classes)), target_names=classes, zero_division=0))

    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["skin_type", "acne_type"])
    p.add_argument("--variant", default="whole", choices=["whole", "face"])
    p.add_argument("--backbone", default="mobilenet_v3_large", choices=sorted(BACKBONES))
    p.add_argument("--tag", default="torch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--augment", default="light",
                   choices=["none", "light", "medium", "strong", "tone"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weights", type=int, default=1)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--freeze-bn", type=int, default=1)
    p.add_argument("--dedup-train", type=int, default=0,
                   help="drop duplicate and contradictory training images")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
