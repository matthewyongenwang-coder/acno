"""Train one Acno classifier, properly.

This replaces the training recipe that was inlined in notebooks/Acno.ipynb. The
notebook version had a specific defect: it ran a fixed number of epochs and saved
whatever the model looked like at the end, so an overfitting run overwrote its own
best weights. skin_type came out of that at 42.5% test accuracy, which is below the
44% you get by ignoring the image and always answering "normal".

What is different here:

- best-weights checkpointing and early stopping, so an overfitting tail cannot
  destroy a good model
- a real fine-tuning stage: more of the backbone unfrozen, at a learning rate high
  enough to actually move it, on a cosine schedule
- stronger augmentation (random crop, flip, rotation, zoom, translation, brightness,
  contrast) applied in the data pipeline rather than inside the model, which also
  keeps the exported ONNX graph clean
- label smoothing, so the model stops being confidently wrong on a noisy dataset
- evaluation on the leak-free "clean" test split as well as the official one
- every run appended to results/experiments.jsonl so the search is reproducible

Only preprocessing is baked into the saved model. Its input contract stays exactly
what the browser app already sends: float32 [0,255], shape (1, 224, 224, 3), NHWC.

Run:
    .venv/bin/python scripts/train_classifier.py --dataset skin_type --backbone mobilenetv2
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import acno_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"
LEDGER = RESULTS / "experiments.jsonl"

SEED = 77
IMG_SIZE = 224


# backbone name -> (keras constructor, how to scale [0,255] before it)
BACKBONES = {
    "mobilenetv2": (
        lambda shape: tf.keras.applications.MobileNetV2(
            input_shape=shape, include_top=False, weights="imagenet"),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
    "mobilenetv3large": (
        lambda shape: tf.keras.applications.MobileNetV3Large(
            input_shape=shape, include_top=False, weights="imagenet",
            include_preprocessing=False),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
    "efficientnetv2b0": (
        lambda shape: tf.keras.applications.EfficientNetV2B0(
            input_shape=shape, include_top=False, weights="imagenet",
            include_preprocessing=False),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
    "efficientnetv2b1": (
        lambda shape: tf.keras.applications.EfficientNetV2B1(
            input_shape=shape, include_top=False, weights="imagenet",
            include_preprocessing=False),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
    "resnet50v2": (
        lambda shape: tf.keras.applications.ResNet50V2(
            input_shape=shape, include_top=False, weights="imagenet"),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
    "densenet121": (
        lambda shape: tf.keras.applications.DenseNet121(
            input_shape=shape, include_top=False, weights="imagenet"),
        lambda: tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess"),
    ),
}


def set_seeds(seed: int = SEED) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_augmenter(strength: str):
    """Augmentation used only while training, applied in the data pipeline.

    Returns None for "none": an empty Sequential cannot be built by Keras, so the
    pipeline skips the augmentation step entirely instead.
    """
    if strength == "none":
        return None
    if strength == "light":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
        ], name="augment")
    if strength == "medium":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.15, 0.15),
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.25),
        ], name="augment")
    if strength == "strong":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.20),
            tf.keras.layers.RandomZoom(0.25, 0.25),
            tf.keras.layers.RandomTranslation(0.15, 0.15),
            tf.keras.layers.RandomBrightness(0.35, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.35),
        ], name="augment")
    raise ValueError(f"unknown augmentation strength: {strength}")


def make_dataset(x, y, num_classes, batch, training, augmenter, label_smoothing):
    """Build a tf.data pipeline from cached uint8 arrays."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)

    def prep(image, label):
        image = tf.cast(image, tf.float32)
        if training:
            # random crop out of the 256px cache, which is free extra augmentation
            image = tf.image.random_crop(image, (IMG_SIZE, IMG_SIZE, 3))
        else:
            image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
        onehot = tf.one_hot(label, num_classes)
        if training and label_smoothing > 0:
            onehot = onehot * (1 - label_smoothing) + label_smoothing / num_classes
        return image, onehot

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    if training and augmenter is not None:
        ds = ds.map(lambda i, l: (augmenter(i, training=True), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(backbone: str, num_classes: int, dropout: float):
    shape = (IMG_SIZE, IMG_SIZE, 3)
    constructor, preprocess = BACKBONES[backbone]
    base = constructor(shape)
    base.trainable = False

    inputs = tf.keras.Input(shape=shape, name="image")
    x = preprocess()(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name=f"acno_{backbone}"), base


def class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    return {i: float(len(y) / (num_classes * counts[i])) for i in range(num_classes)}


def evaluate(model, x, y, classes, batch=64):
    ds = make_dataset(x, y, len(classes), batch, False, None, 0.0)
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean([
            (pred[y == c] == c).mean() for c in range(len(classes)) if (y == c).any()
        ])),
        "per_class_recall": {
            classes[c]: float((pred[y == c] == c).mean()) if (y == c).any() else None
            for c in range(len(classes))
        },
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(classes)))).tolist(),
        "predictions": pred.tolist(),
    }


def train(args) -> dict:
    set_seeds(args.seed)
    splits = acno_data.load_splits(args.dataset, variant=args.variant)
    classes = splits["classes"]
    num_classes = len(classes)
    augmenter = build_augmenter(args.augment)

    train_ds = make_dataset(splits["train"]["x"], splits["train"]["y"], num_classes,
                            args.batch, True, augmenter, args.label_smoothing)
    val_ds = make_dataset(splits["valid"]["x"], splits["valid"]["y"], num_classes,
                          args.batch, False, None, 0.0)

    model, base = build_model(args.backbone, num_classes, args.dropout)
    weights = class_weights(splits["train"]["y"], num_classes) if args.class_weights else None
    loss = tf.keras.losses.CategoricalCrossentropy()

    ckpt = MODELS / f"_ckpt_{args.dataset}_{args.tag}.keras"
    MODELS.mkdir(parents=True, exist_ok=True)

    def callbacks(patience):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                str(ckpt), monitor="val_accuracy", mode="max",
                save_best_only=True, verbose=0),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=patience,
                restore_best_weights=True, verbose=1),
        ]

    started = time.time()
    history = {}

    # stage 1: train only the new head, backbone frozen
    model.compile(optimizer=tf.keras.optimizers.Adam(args.head_lr),
                  loss=loss, metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                   class_weight=weights, callbacks=callbacks(args.patience),
                   verbose=args.verbose)
    history.update({f"stage1_{k}": v for k, v in h1.history.items()})

    # stage 2: unfreeze the top of the backbone and fine-tune on a cosine schedule
    if args.finetune_epochs > 0:
        base.trainable = True
        n_freeze = int(len(base.layers) * args.freeze_fraction)
        for layer in base.layers[:n_freeze]:
            layer.trainable = False
        # BatchNorm statistics are estimated from ImageNet on far more data than we
        # have; letting them drift on a few thousand images destabilises training
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        steps = max(1, len(splits["train"]["y"]) // args.batch)
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.finetune_lr,
            decay_steps=steps * args.finetune_epochs,
            alpha=0.02,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(schedule),
                      loss=loss, metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                       class_weight=weights, callbacks=callbacks(args.patience),
                       verbose=args.verbose)
        history.update({f"stage2_{k}": v for k, v in h2.history.items()})

    elapsed = time.time() - started

    eval_splits = [s for s in ["valid", "test", "clean_test", "strict_test"]
                   if s in splits]
    metrics = {
        split: evaluate(model, splits[split]["x"], splits[split]["y"], classes)
        for split in eval_splits
    }

    record = {
        "tag": args.tag,
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
        "head_epochs": args.head_epochs,
        "finetune_lr": args.finetune_lr,
        "finetune_epochs": args.finetune_epochs,
        "freeze_fraction": args.freeze_fraction,
        "seed": args.seed,
        "classes": classes,
        "train_seconds": round(elapsed, 1),
        "params": int(model.count_params()),
        "machine": platform.processor() or platform.machine(),
        "metrics": {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in metrics.items()
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    # always keep the run's own weights under its tag, so a good configuration found
    # mid-search never has to be retrained just to be kept
    tagged = MODELS / f"{args.dataset}__{args.tag}.keras"
    model.save(tagged)
    record["weights"] = str(tagged.relative_to(REPO_ROOT))
    print(f"[saved] {tagged}")

    if args.save:
        dest = MODELS / f"{args.dataset}.keras"
        model.save(dest)
        print(f"[promoted] {dest}")
    ckpt.unlink(missing_ok=True)

    print(f"\n=== {args.dataset} / {args.variant} / {args.backbone} / {args.tag} ===")
    for split in eval_splits:
        m = metrics[split]
        print(f"  {split:<11} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  "
              f"balanced_acc={m['balanced_accuracy']:.3f}")
    print(f"  trained in {elapsed / 60:.1f} min")
    print("\nclean test classification report:")
    print(classification_report(
        splits["clean_test"]["y"], np.array(metrics["clean_test"]["predictions"]),
        labels=list(range(num_classes)), target_names=classes, zero_division=0))

    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["skin_type", "acne_type"])
    p.add_argument("--variant", default="whole", choices=["whole", "face"],
                   help="'face' trains on Haar-cascade face crops, matching inference")
    p.add_argument("--backbone", default="mobilenetv2", choices=sorted(BACKBONES))
    p.add_argument("--tag", default="run")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--augment", default="medium",
                   choices=["none", "light", "medium", "strong"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weights", type=int, default=1)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--head-epochs", type=int, default=12)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--freeze-fraction", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--save", action="store_true", help="overwrite models/<dataset>.keras")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
