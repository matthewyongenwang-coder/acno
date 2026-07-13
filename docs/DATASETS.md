# Acno Datasets

All datasets are public Kaggle datasets fetched by `scripts/download_data.py` into
`data/raw/` (gitignored). Run `scripts/verify_data.py` after downloading.

## Why the images are not committed to this repo

1. Privacy: these are photos of real people's faces, collected for research. We do not
   redistribute them, and we never commit our own or users' photos either.
2. Licensing: research datasets generally allow use, not re-publication.
3. Size: GitHub blocks files over 100 MB and repos become unusable with thousands of
   binary images in history.

Anyone on the team gets identical data by running one script, which is safer and more
reproducible than copies living in git.

## 1. skin_type - Oily, Dry and Normal Skin Types

- Source: [shakyadissanayake/oily-dry-and-normal-skin-types-dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset)
- Task: 3-class image classification for the skin type model
- Layout: `data/raw/skin_type/{train,valid,test}/{dry,normal,oily}/`

| split | dry | normal | oily | total |
|-------|-----|--------|------|-------|
| train | 652 | 1104   | 1000 | 2756  |
| valid | 71  | 111    | 80   | 262   |
| test  | 35  | 59     | 40   | 134   |

Note the class imbalance (dry is under-represented): use class weights or oversampling.

## 2. acne_type - Acne Type Classification

- Source: [tiswan14/acne-dataset-image](https://www.kaggle.com/datasets/tiswan14/acne-dataset-image)
- Task: 5-class image classification for the acne type model
- Layout: `data/raw/acne_type/{train,valid,test}/{Whiteheads,Blackheads,Papules,Pustules,Cyst}/`

| split | Whiteheads | Blackheads | Papules | Pustules | Cyst | total |
|-------|-----------|------------|---------|----------|------|-------|
| train | 193       | 735        | 621     | 584      | 645  | 2778  |
| valid | 49        | 240        | 209     | 217      | 206  | 921   |
| test  | 57        | 265        | 202     | 205      | 189  | 918   |

Whiteheads is heavily under-represented (about 4x fewer than other classes): weight it.

## 3. acne_yolo - Acne Lesion Detection (ACNE04-derived)

- Source: [osmankagankurnaz/acne-dataset-in-yolov8-format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format)
- License: CC BY 4.0 (via Roboflow project skin-detection-uvj1f v8)
- Task: single-class ("Acne") lesion detection. Lesion count per face is our severity
  score input (more lesions = more severe, aligned with the Hayashi grading idea from ACNE04).
- Layout: `data/raw/acne_yolo/{train,valid,test}/{images,labels}/` plus `data.yaml`
- Counts: train 823, valid 56, test 48 image/label pairs. Labels are YOLO format
  (class x_center y_center width height, normalized).

## Known gaps and risks (read before training)

- Skin tone coverage: none of these datasets document skin tone distribution. Dermatology
  models are known to underperform on darker skin. Before shipping anything, evaluate on
  diverse examples; consider adding [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
  for a fairness audit.
- Combination/sensitive skin types are not in the skin_type dataset (only dry/normal/oily).
  The app copy must not pretend to detect classes the model was never trained on.
- Labels come from public datasets, not our own dermatologist review. Treat model output
  as guidance, never diagnosis. The app must always include a see-a-dermatologist path.
