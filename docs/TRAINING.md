# How the Acno models are trained

This replaces the training recipe that used to live only inside `notebooks/Acno.ipynb`.
The notebook is still the version the team runs on Colab; these scripts are the same
recipe, runnable locally and reproducible run for run.

## Why the original recipe was rewritten

The first version of Acno reported 42.5% test accuracy on skin_type and 59.3% on
acne_type. Investigating those numbers turned up four separate problems, three of them
in the training code rather than the data.

**1. The best model was being thrown away.** The notebook trained for a fixed 8 + 6
epochs and saved whatever the weights looked like after the final epoch. There was no
checkpointing on validation accuracy and no early stopping. When a run overfitted, and
on datasets this small they do, the overfitted tail overwrote the good model. This is
the single biggest cause of the bad skin_type number.

**2. The fine-tuning stage barely fine-tuned.** Stage two unfroze the last 40 layers at
a learning rate of 1e-5 for 6 epochs. That is small enough and short enough that the
backbone hardly moved, so the model was effectively a linear probe with extra steps.

**3. Augmentation was fighting the task.** The recipe applied brightness and contrast
jitter to every image. For acne_type, the difference between a blackhead and a
whitehead is largely whether the plug is dark or light, so aggressive photometric
augmentation attacks the exact signal the model needs. Augmentation strength is now a
tuned parameter rather than a fixed choice, and every setting we tried is recorded.

**4. The two inference paths disagree with each other.** `src/pipeline.py` detects the
largest face with a Haar cascade and classifies a padded crop of it.
`web/lib/analyze.ts`, which is what users actually run, stretch-resizes the whole
photo to 224x224 and classifies that. CLAUDE.md describes the Python file as mirroring
the browser app, but on this point it does not. Since the browser app is the product,
the shipped models are trained on whole images to match it.

`scripts/build_face_crops.py` builds a face-cropped variant of the training data
anyway, selectable with `--variant face`. It is worth measuring for two reasons: it
tells us how much of the model's score comes from backgrounds and framing rather than
skin, and it tells us whether adding face detection to the browser app would be worth
the work. Adopting it for real would mean shipping a face detector in the browser, so
the bar is a clear accuracy win, not a marginal one.

Separately, the datasets themselves have problems that made the reported numbers
optimistic. Those are documented in [DATA_QUALITY.md](DATA_QUALITY.md).

## The current recipe

`scripts/train_classifier.py` does the following:

- **Caching.** Every image is decoded once to a 256px uint8 array in `data/cache/`.
  Training then reads arrays instead of JPEGs, which is what makes a full sweep
  affordable on a laptop.
- **Random crop.** Images are cached at 256px and randomly cropped to 224px during
  training, centre-cropped at evaluation. Free augmentation with no distortion.
- **Two stages.** Stage one trains only the new classification head with the backbone
  frozen. Stage two unfreezes the upper part of the backbone (`--freeze-fraction`
  controls how much) and fine-tunes on a cosine-decayed schedule.
- **Frozen BatchNorm during fine-tuning.** BatchNorm statistics come from ImageNet,
  estimated on far more data than we have. Letting them drift on a few thousand images
  destabilises training, so they stay frozen.
- **Best-weight checkpointing and early stopping** on validation accuracy, with
  `restore_best_weights`. An overfitting tail can no longer destroy the run.
- **Class weights and label smoothing**, both optional flags, because Whiteheads has
  about a quarter the images of the other acne classes.
- **Evaluation on four splits**: validation, the official test split, the leak-free
  clean split, and the stricter feature-space-deduplicated split.

Only preprocessing is baked into the saved model, as a `Rescaling` layer. Augmentation
lives in the data pipeline. This keeps the exported ONNX graph clean and preserves the
input contract the browser already sends: float32 in the range 0 to 255, shape
`(1, 224, 224, 3)`, NHWC.

## Running it

```bash
# one-time: audit the data and build the caches
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/verify_duplicates.py
.venv/bin/python scripts/leak_scan.py
.venv/bin/python scripts/build_face_crops.py --dataset skin_type

# survey backbones cheaply before committing to a full fine-tune
.venv/bin/python scripts/probe_backbones.py

# train one configuration
.venv/bin/python scripts/train_classifier.py \
    --dataset acne_type --backbone mobilenetv2 --augment medium --tag my_run

# see every run so far, best first
.venv/bin/python scripts/leaderboard.py

# lesion detector (uses the Mac GPU through MPS when available)
.venv/bin/python scripts/train_yolo.py --epochs 150
```

Add `--save` to overwrite `models/<dataset>.keras`. Without it, a run is measured and
recorded but changes nothing, which is what you want while searching.

Every run appends one line to `results/experiments.jsonl` with its full configuration
and all metrics. The leaderboard reads that file, so the search history is the record.
Failed and mediocre configurations stay in it on purpose.

## Choosing a backbone

`scripts/probe_backbones.py` runs each candidate backbone frozen, once, over the whole
dataset and fits a logistic-regression head on the resulting features. That takes
seconds per backbone instead of an hour, and it ranks candidates well enough to decide
which ones deserve a full fine-tune.

It is also a useful floor. If no backbone can separate the classes even a little when
frozen, the problem is the labels rather than the training recipe. That is exactly what
it told us about skin_type.

## Reporting rules for this project

- Always quote the majority-class baseline next to an accuracy figure. On acne_type,
  29% accuracy is not a weak model, it is no model at all.
- Always quote the clean or strict split, not just the official one. Roughly half the
  official acne_type test set is contaminated.
- Always report macro F1 and per-class recall alongside accuracy, because Whiteheads is
  small enough to be ignored by a model with a healthy-looking overall score.
- The lesion detector is measured on the test split, not the validation split.
  Validation is what early stopping optimised, so quoting it overstates the model.
