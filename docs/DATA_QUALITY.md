# Data quality audit

Written July 2026, before retraining the Acno models. Everything here is reproducible
with two scripts in this repo:

```bash
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/verify_duplicates.py
```

The short version: our three datasets are public Kaggle datasets of varying quality,
and two of them have problems that made our original accuracy numbers misleading. We
found them, measured them, and changed how we report results. This document is the
record of that, because a project that tells teenagers things about their skin should
be honest about how well its models actually work.

## 1. The headline problem: test images that are also training images

A test set only means something if the model has never seen those images. We checked
by hashing every image and comparing across splits, then confirmed each hit by
decoding both images and measuring the actual pixel difference (so we are not fooled
by hash collisions).

| Dataset | Test images | Also in train or valid | Share of test |
|---|---|---|---|
| skin_type | 134 | 15 | 11.2% |
| acne_type | 918 | 448 | 48.8% |

Nearly half of the acne_type test set was sitting in the training data. Any accuracy
measured on that split is partly a memory test, not a generalisation test.

### Why acne_type is so badly affected

The acne_type dataset was exported from Roboflow, and looking at the images makes the
cause obvious: many are rotated with black corner padding, and some are mosaics of
several crops stitched together. Those are augmentations. The dataset author applied
augmentation to the whole collection **and then split it** into train, valid and test,
so augmented siblings of the same original photo ended up on both sides of the split.

This also means our duplicate count is a floor, not a ceiling. A photo rotated by 30
degrees does not match its original under any pixel hash, but a model that memorised
one will still recognise the other. `scripts/leak_scan.py` re-checks the same question
in feature space, where rotated and re-cropped copies stay close together.

The result is much worse than the pixel check suggested:

| Dataset | Test images | Flagged by feature-space scan | Share of test |
|---|---|---|---|
| skin_type | 134 | 20 | 14.9% |
| acne_type | 918 | 597 | 65.0% |

For acne_type the *median* test image has a training neighbour at 0.979 cosine
similarity. We checked the flagged pairs by eye rather than trusting the threshold, and
they are unambiguous: the same stray hairs, the same moles, the same mosaic collages
tile for tile. These are the same photographs, not merely similar skin.

So two thirds of the official acne_type test split is unusable for measuring
generalisation. The **strict split** (321 images) is what we report.

### What we do about it

We keep the official split so our numbers stay comparable with what we reported
before, and we add two stricter ones:

- **clean test**: official test, minus every image confirmed to appear in train or valid
- **strict test**: also minus every image whose nearest training neighbour in feature
  space is near-identical

The strict number is the one we treat as real. Reporting only the official number would
have flattered us by a wide margin.

### A second consequence: the evaluation sets are now small

Removing leaked images leaves skin_type with 114 usable test images and acne_type with
321. At 114 images, a measured accuracy of 50% carries a 95% confidence interval about
nine points wide in each direction. That is wide enough that a sweep's best run is
mostly luck, which we confirmed by retraining one configuration with three different
random seeds and getting 50.9%, 38.6% and 43.9%. See RESULTS.md.

The practical rule: for skin_type, do not treat a difference of under about ten points
as real.

## 2. skin_type is weakly labelled

The skin_type dataset (dry / normal / oily) is scraped from the web, and the labels
appear to come from what the original post said rather than from what the skin looks
like. Sampling the training images shows:

- stock photos with the word "OILY" printed across them, and creator watermarks
- makeup tutorials, with brushes and foundation mid-application
- people holding a skincare product, where the product is the reason for the label
- before/after collages, two panels in one image
- images rotated with black padding, and extreme close-ups of a single eye

Two measurements confirm the impression:

- **Only 62.6% of skin_type images contain a detectable frontal face** (OpenCV Haar
  cascade, the same detector the app itself uses). The rest are collages, product
  shots or crops too tight to register as a face.
- **9 groups of identical images inside the training set carry different labels.** The
  same photo is filed as dry in one place and oily in another.

This is why every ImageNet backbone we tried stalls near the majority-class baseline
on this dataset. The signal is genuinely weak, not merely hard to reach.

| Backbone (frozen, linear probe) | skin_type clean-test accuracy |
|---|---|
| majority-class baseline | 37.8% |
| mobilenetv2 | 38.7% |
| mobilenetv3large | 32.8% |
| efficientnetv2b0 | 38.7% |
| efficientnetv2b1 | 39.5% |
| resnet50v2 | 41.2% |
| densenet121 | 38.7% |

For contrast, the same probe on acne_type reaches 81.9%. The method is fine. The
skin_type labels are the problem.

## 3. Baselines every number should be read against

"Accuracy" means nothing without the score for ignoring the image entirely and always
answering the most common class:

| Dataset | Split | Majority-class baseline |
|---|---|---|
| skin_type | official test | 44.0% |
| skin_type | clean test | 37.8% |
| acne_type | official test | 28.9% |
| acne_type | clean test | 29.1% |

Our previously reported skin_type accuracy was **42.5% against a 44.0% baseline**. The
model was worse than a constant answer of "normal". That is the single most important
number in this document, and it is why the retraining work started.

## 4. Class balance

| Dataset | Train split |
|---|---|
| skin_type | normal 1104, oily 1000, dry 652 |
| acne_type | Blackheads 735, Cyst 645, Papules 621, Pustules 584, Whiteheads 193 |

Whiteheads is roughly a quarter the size of the other acne classes, so we report
per-class recall and macro F1 alongside accuracy. An overall accuracy score can look
healthy while a small class is being missed almost every time.

## 5. What this changes about the app

- skin_type is the weakest part of Acno and the app must keep saying so. It is
  presented as a suggestion, never a finding.
- acne_type and the lesion detector are on much firmer ground.
- Fairness testing across skin tones (the funded project's first milestone) has to be
  read with this in mind: on a dataset this weakly labelled, a per-tone accuracy gap
  may reflect who gets photographed for skincare posts rather than how the model
  behaves.

## Reproducing

```bash
.venv/bin/python scripts/audit_data.py          # counts, baselines, overlap
.venv/bin/python scripts/verify_duplicates.py   # confirms duplicates pixel by pixel
.venv/bin/python scripts/leak_scan.py           # feature-space leak scan
.venv/bin/python scripts/build_face_crops.py    # face detection rate
.venv/bin/python scripts/diagnose_skin_type.py  # why skin_type is hard
```

Outputs land in `results/`, which is gitignored because the artefacts can contain
dataset faces. The numbers in this document are the shareable summary.
