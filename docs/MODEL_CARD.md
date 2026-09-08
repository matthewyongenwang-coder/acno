# Acno model card

Current accuracy figures live in [RESULTS.md](RESULTS.md), which is regenerated from
the experiment ledger rather than written by hand. This document covers what the models
are, what they should not be used for, and where they fail.

## What Acno is

A free web app that gives a teenager a private first answer about their skin. The user
takes or uploads a photo; an input gate checks the photo actually shows skin, then three
models run entirely in their browser and report skin type, acne type and a lesion count
that maps to a severity band. A fourth step sends the resulting numbers, never the photo,
to Gemini to write a personalised guide.

Photos never leave the device. There is no upload, no account and no storage.

## The input gate

None of the three models has a "none of the above" class, so argmax always names a skin
type, an acne type and a severity, even for a photo of a wall. Before September 2026 the
app printed that as a real report. Now a photo has to pass one of two checks first:

| Signal | What it is | Fires on |
|---|---|---|
| face | YuNet (232KB ONNX), same model in the browser and in `src/pipeline.py` | 92% of face photos, 4.3% of acne close-ups |
| skin | Fraction of the frame passing the YCrCb + CIELAB skin test in `src/skin.py`, threshold 0.10 | close-ups, where no face is visible |

A photo passes if **either** fires. A face alone cannot be required, because most
legitimate close-ups of a cheek contain no detectable face. Calibrated on 300 random
images per dataset in `scripts/calibrate_face_gate.py`: the combined gate passes 100% of
skin_type photos and 98.7% of acne_type close-ups, while every non-skin negative tested
(solid colours, plot images, random noise) scores 0.069 or below.

The gate is deliberately soft. A photo that fails is not analysed, but the person can
choose to see the result anyway, in which case the report is labelled unverified and the
AI write-up is withheld. It is a gate, not a crop: the classifiers are trained on whole
images, so cropping to the detected face would break train/serve parity.

## The three models

| Model | Task | Architecture | Output |
|---|---|---|---|
| skin_type | dry / normal / oily | MobileNetV3-Large, ImageNet pretrained, fine-tuned | softmax over 3 classes |
| acne_type | blackheads / whiteheads / papules / pustules / cysts | MobileNetV3-Large, ImageNet pretrained, fine-tuned, mirror TTA in the graph | softmax over 5 classes |
| acne_yolo | locate every lesion | YOLOv8-nano | boxes, counted into a severity band |

All three are exported to ONNX and run through onnxruntime-web. The classifiers take
float32 pixels in the range 0 to 255, shape `(1, 224, 224, 3)`, NHWC, and return softmax
probabilities. Normalisation and softmax live inside the graph, so the browser sends raw
pixel values and needs no knowledge of the model. Classes are in alphabetical order,
matching what `web/lib/analyze.ts` expects.

The acne_type graph additionally runs each image twice, once mirrored, and averages the
two predictions. That is one extra forward pass and no change at all from the browser's
point of view.

Severity comes from the lesion count, following the shape of the Hayashi grading scale:
0 to 1 clear, 2 to 10 mild, 11 to 25 moderate, more than 25 severe.

## Training data

Three public Kaggle datasets, documented in [DATASETS.md](DATASETS.md) with licences.
None of the images are in this repository: they are photographs of real people and are
licensed for research, not redistribution. `scripts/download_data.py` fetches them.

Read [DATA_QUALITY.md](DATA_QUALITY.md) before trusting any number. In short: both
classifier datasets had substantial overlap between their training and test splits, so
every accuracy figure published for them before this audit, including our own earlier
numbers, was optimistic.

**acne_type's 98.8% is an upper bound, not field accuracy.** A frozen ImageNet backbone,
trained on nothing, scores 73.5% on the same strict split against a 26.8% baseline, while
the same probe sits below baseline on skin_type. Blank out the middle half of every image,
where the lesion has to be, and the probe still scores 63.9%, so nearly 80% of its
advantage survives deleting the thing it is supposed to be classifying. The class is
largely predictable from colour and background outside the lesion. The strict split cannot
remove that, because it is shared across a whole class rather than tied to duplicate
images. Do not present 98.8% as how well Acno reads a stranger's face.

## Intended use

Educational guidance for a general teenage audience, as a first step before deciding
whether to see someone. Acne affects most people aged 12 to 24, and Acno exists because
real guidance is expensive, slow to book, and for many teens too embarrassing to seek
in person.

## Out of scope

- **Not a diagnosis.** Nothing the app says is a medical finding. Every report carries a
  disclaimer, and anything severe or cystic is routed to "please see a dermatologist".
- **Not for skin cancer, moles or rashes.** The models have only ever seen acne. They
  will confidently give an acne answer for something that is not acne. If a mark is
  changing, bleeding or new and unexplained, that is a doctor's job.
- **Not for tracking treatment.** Lesion counts move with lighting and camera angle by
  more than they move with real change over a few days.
- **Not for adults with a different skin concern**, and not validated on skin of any
  particular tone (see below).

## Known limitations

**skin_type is the weak one, and the app says so.** Its dataset is scraped from the web
and weakly labelled: labels follow what the original post said rather than what the skin
looks like. Only 62.6% of its images even contain a detectable face; the rest are
collages, product shots and close-ups. Nine groups of identical training images carry
contradictory labels. The model is presented as a suggestion, never a finding, and the
app must keep it that way.

**The strict skin_type test split is only 114 images, and its results are unstable.**
Retraining the same configuration with three different random seeds gave 50.9%, 38.6%
and 43.9%. Validation accuracy did not predict which run would do well, so we cannot
even select the good one honestly. Treat skin_type as roughly 44% with several points
of slack, against a 37.7% baseline, and do not treat small differences as real.

This is a dataset limit rather than a tuning limit. Bigger backbones did worse, not
better: EfficientNetV2B1 scored below MobileNetV2. Extra capacity memorises noisy
labels. Replacing the dataset is the only change likely to move this number much.

**Whiteheads is the smallest acne class** at roughly a quarter the size of the others,
so per-class recall is reported alongside accuracy. An overall score can look healthy
while a small class is missed nearly every time.

**Fairness across skin tones is now measured, and only partly answerable.** See
[FAIRNESS.md](FAIRNESS.md). acne_type shows no meaningful difference across tone groups,
on a test with enough power to have detected a gap larger than about 7 points. skin_type
and the lesion detector show no difference either, but their test sets are so small that
a 40-point gap could hide in them, so those are not findings.

The limiting factor is representation: these datasets contain almost no dark skin. The
skin_type test split has zero images in the darkest tone band. **We do not claim Acno
works equally well on dark skin, because we have barely been able to test it.** Training
with tone-aware augmentation raised every acne_type tone group and narrowed the spread
between best and worst from 4.4 to 2.4 points, and that is the version shipping.

**The two inference paths disagree.** `src/pipeline.py` crops to a detected face before
classifying; `web/lib/analyze.ts`, which is what users actually run, classifies the whole
photo. The shipped models are trained to match the browser. Training on face crops does
measurably better (see RESULTS.md), but adopting it means shipping a face detector in
the browser, which has not been done.

## Ethical considerations

- The subject matter is one teenagers are already self-conscious about. The app's tone
  is deliberately plain and non-judgemental, and it never uses words like "bad skin".
- Photographs of faces are the most sensitive input a teenager could give an app, which
  is why the entire pipeline runs on-device and nothing is ever uploaded.
- A confident wrong answer is worse than an uncertain one here. Where the models are
  weak, the interface should show that rather than hide it.

## Maintenance

The shipped classifiers are trained in PyTorch on the Mac GPU, which is roughly five
times faster than the TensorFlow path and is what made a real hyperparameter search
possible. The TensorFlow scripts and the Colab notebook still work and produce
comparable models; both are documented in [TRAINING.md](TRAINING.md).

- Retrain: `scripts/train_torch.py` (MPS/CUDA) or `scripts/train_classifier.py`
  (TensorFlow, CPU).
- Re-score an existing checkpoint, including on the strict split:
  `scripts/evaluate_torch.py`.
- Regenerate results: `scripts/write_report.py`. Never edit RESULTS.md by hand.
- Re-export to ONNX: `scripts/export_onnx_torch.py` (PyTorch) or
  `scripts/convert_models.py` (TensorFlow). Both check the ONNX output against the
  original model on real dataset images, confirm the predicted class never changes, and
  warn if the download exceeds budget.
- `web/public/models/models.json` records which run produced each shipped model.

## Download size

The browser downloads all three models once and caches them.

| Model | Size |
|---|---|
| skin_type.onnx | 16.8 MB |
| acne_type.onnx | 16.9 MB |
| acne_yolo.onnx | 12.3 MB |
| **total** | **46.0 MB** |

This is up from about 30 MB, because both classifiers moved from MobileNetV2 to
MobileNetV3-Large. The acne_type gain justifies it comfortably. If the download becomes
a problem for users on mobile data, quantising the classifiers to float16 would roughly
halve them, but that needs testing against onnxruntime-web's wasm backend before it
ships.
