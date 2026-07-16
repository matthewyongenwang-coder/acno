# Citations and Sources

This document lists every external source Acno relies on: the datasets our models
were trained on, the research the methods come from, the software libraries and
pretrained models we build on, the AI service that writes personalized guidance, and
the basis for the skincare advice shown to users.

Acno is an educational student project (Inspirit AI). Nothing here is a medical
diagnosis. The app always recommends seeing a dermatologist for severe or cystic acne.

Last updated: 2026-07-16.

---

## 1. Training datasets

All three datasets are public datasets on Kaggle. They are downloaded (not committed to
this repo) by `scripts/download_data.py`. The images are photos of real people's faces,
licensed for research use, so we never redistribute them. See `docs/DATASETS.md` for
class counts, splits, and known risks.

### 1.1 Skin type classification (dry / normal / oily)
- **Title:** Oily, Dry and Normal Skin Types Dataset
- **Author:** Shakya Dissanayake
- **Source:** https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
- **Used for:** training the skin-type classifier (3 classes)

### 1.2 Acne type classification (whiteheads / blackheads / papules / pustules / cyst)
- **Title:** Acne Dataset Image
- **Author:** Tiswan (Kaggle user tiswan14)
- **Source:** https://www.kaggle.com/datasets/tiswan14/acne-dataset-image
- **Used for:** training the acne-type classifier (5 classes)

### 1.3 Acne lesion detection (severity by lesion count)
- **Title:** Acne Dataset in YOLOv8 Format
- **Author:** Osman Kagan Kurnaz
- **Source:** https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format
- **License:** CC BY 4.0 (exported via the Roboflow project `skin-detection-uvj1f`, v8)
- **Derived from:** the ACNE04 research dataset (see 2.1)
- **Used for:** training the YOLOv8 lesion detector; lesion count feeds our severity score

---

## 2. Research and methods

### 2.1 ACNE04 dataset and acne grading
- Wu, X., Wen, N., Liang, J., Lai, Y-K., She, D., Cheng, M-M., Yang, J.
  "Joint Acne Image Grading and Counting via Label Distribution Learning."
  *IEEE International Conference on Computer Vision (ICCV)*, 2019.
- The lesion-detection dataset (1.3) is derived from ACNE04. Our severity idea (count
  the lesions, then bin the count) follows this line of work.

### 2.2 Lesion-count severity grading
- Hayashi, N., Akamatsu, H., Kawashima, M.
  "Establishment of grading criteria for acne severity."
  *The Journal of Dermatology*, 35(5), 255-260, 2008.
- Our count-to-severity bins (clear / mild / moderate / severe) follow the clinical
  practice of grading acne severity by lesion count.

### 2.3 MobileNetV2 (backbone for the two classifiers)
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L-C.
  "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
  *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.
- We use MobileNetV2 pretrained on ImageNet and fine-tune it (transfer learning).

### 2.4 ImageNet (pretraining data for the backbone)
- Deng, J., Dong, W., Socher, R., Li, L-J., Li, K., Fei-Fei, L.
  "ImageNet: A Large-Scale Hierarchical Image Database."
  *CVPR*, 2009.

### 2.5 YOLOv8 (lesion detector architecture)
- Jocher, G., Chaurasia, A., Qiu, J. "Ultralytics YOLOv8." 2023.
  https://github.com/ultralytics/ultralytics

### 2.6 Grad-CAM (model explainability visualization)
- Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.
  "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."
  *ICCV*, 2017.
- Used in the training notebook to show where the skin-type model looks.

### 2.7 Haar cascade face detection
- Viola, P., Jones, M. "Rapid Object Detection using a Boosted Cascade of Simple
  Features." *CVPR*, 2001.
- We use OpenCV's `haarcascade_frontalface_default.xml` to crop the face before analysis.

### 2.8 Fitzpatrick17k (recommended for fairness auditing)
- Groh, M., Harris, C., Soenksen, L., et al. "Evaluating Deep Neural Networks Trained
  on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset."
  *CVPR Workshops*, 2021. https://github.com/mattgroh/fitzpatrick17k
- Not used for training. Listed as the intended dataset for auditing performance across
  skin tones, an acknowledged weakness of dermatology AI.

---

## 3. Software, libraries, and frameworks

### Machine learning / data (Python)
- **TensorFlow / Keras** - training the two image classifiers. https://www.tensorflow.org
- **Ultralytics YOLOv8** - training and running the lesion detector. https://github.com/ultralytics/ultralytics
- **OpenCV** (`opencv-python-headless`) - face detection and image processing. https://opencv.org
- **scikit-learn** - evaluation metrics and confusion matrices. https://scikit-learn.org
- **matplotlib** - result and training charts. https://matplotlib.org
- **kagglehub** - anonymous dataset downloads. https://github.com/Kaggle/kagglehub
- **tf2onnx / ONNX** - converting the trained models to ONNX. https://github.com/onnx/tensorflow-onnx

### Web app (browser)
- **Next.js** - the web application framework. https://nextjs.org
- **React** - user interface library. https://react.dev
- **onnxruntime-web** - runs all three models in the browser so the photo never leaves
  the device. https://onnxruntime.ai
- **TypeScript** - https://www.typescriptlang.org

Full pinned versions live in `requirements.txt` (Python) and `web/package.json` (web).

---

## 4. AI service (personalized guidance)

- **Anthropic Claude** (model `claude-opus-4-8`), via the `@anthropic-ai/sdk` package.
  https://www.anthropic.com
- Used in `web/app/api/advice/route.ts` to turn the scan results into a personalized
  written analysis and over-the-counter product suggestions.
- **Important:** only the scan results (skin type, acne types, lesion count, severity)
  are sent to the API. The photo itself is never uploaded and never leaves the device.

---

## 5. Skincare and dermatology guidance (advice content)

The rules-based routines in `data/guide_rules.csv` and the acne-type explanations in
`data/acne_type_info.csv` were written by the team as general, widely published skincare
guidance for teens (for example: benzoyl peroxide 2.5%, salicylic acid, adapalene 0.1%,
non-comedogenic moisturizers, and the "give a change 6 to 8 weeks" rule). This guidance
reflects publicly available consumer information from established dermatology and health
authorities. For readers who want to verify or go deeper, these are the appropriate
authoritative references:

- **American Academy of Dermatology (AAD)** - Acne resource center.
  https://www.aad.org/public/diseases/acne
- **Mayo Clinic** - Acne: diagnosis and treatment.
  https://www.mayoclinic.org/diseases-conditions/acne
- **NHS (UK)** - Acne.
  https://www.nhs.uk/conditions/acne
- **U.S. FDA** - Over-the-counter acne products (benzoyl peroxide, salicylic acid, adapalene).
  https://www.fda.gov

This content is educational only and is not a substitute for professional medical advice.

---

## How to cite Acno

> Acno: an educational acne-analysis web app. Inspirit AI student project, 2026.
> https://github.com/matthewyongenwang-coder/acno

If you reuse this project, please also credit the dataset authors listed in Section 1
and the research listed in Section 2.
