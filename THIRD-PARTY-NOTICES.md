# Third-party notices

The MIT license in [LICENSE](LICENSE) covers the code, documentation, and advice
content the Acno team wrote. Some files in this repository come from, or are derived
from, other people's work and carry their own terms. Those terms win where they apply.

Full source list, including research papers, is in [CITATIONS.md](CITATIONS.md).

## Lesion detector weights (AGPL-3.0, not MIT)

- Files: `models/acne_yolo.pt`, `models/acne_yolo.onnx`, `web/public/models/acne_yolo.onnx`
- Trained with Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics),
  which is licensed AGPL-3.0.
- Ultralytics treats models trained with its software as covered by AGPL-3.0 unless
  you hold an Ultralytics Enterprise License. We do not, so we treat these weight
  files as AGPL-3.0, and this repository is public and open source, which is what
  AGPL-3.0 asks for.
- If you want the rest of Acno under plain MIT terms with no AGPL obligations,
  omit these three files and swap in a detector of your own.

Note that the browser app itself does not ship Ultralytics code: inference runs on
onnxruntime-web (MIT).

## Classifier weights

- Files: `models/skin_type.*`, `models/acne_type.*`, `web/public/models/skin_type.onnx`,
  `web/public/models/acne_type.onnx`
- Fine-tuned from MobileNetV2 as shipped in Keras/TensorFlow (Apache-2.0), pretrained
  on ImageNet. ImageNet's own terms allow non-commercial research use, so treat these
  weights as educational and non-commercial.

## Face detector weights (MIT)

- File: `web/public/models/face_detector.onnx`
- YuNet, `face_detection_yunet_2023mar.onnx`, taken unmodified from the OpenCV Zoo
  (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
  which is MIT licensed. SHA-256
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Authored by Wei Wu, Yuantao Feng and Shiqi Yu. Cited in CITATIONS.md.
- Used only to decide whether a photo contains a face before analysing it. It is
  never used to identify anyone, and like every other model here it runs in the
  browser: no photo is uploaded.

## Training datasets

Dataset images are never committed to this repository; `scripts/download_data.py`
fetches them from Kaggle at build time. They are photos of real people's faces.

- Acne lesion detection dataset (Osman Kagan Kurnaz, derived from ACNE04): CC BY 4.0.
  Attribution is given in CITATIONS.md section 1.3.
- Skin type dataset (Shakya Dissanayake) and acne type dataset (Tiswan): used under
  the terms shown on their Kaggle pages. Redistribution of the images is not granted
  by the MIT license here.

## Libraries

Python dependencies (`requirements.txt`) and npm dependencies (`web/package.json`)
stay under their own licenses, held by their own authors. Notably `ultralytics`
(AGPL-3.0) is a training-time dependency only.

## Medical disclaimer

Acno is an educational student project. It does not diagnose, treat, or give medical
advice, and the MIT license's "as is, no warranty" terms apply to everything here.
For severe or persistent acne, see a dermatologist.
