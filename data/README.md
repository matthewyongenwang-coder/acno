# data/

This folder is gitignored. Nothing in here should ever be committed: the datasets
contain photos of real people's faces and are licensed for research use, not
redistribution.

To populate it:

```bash
python scripts/download_data.py
python scripts/verify_data.py
```

Expected layout after download:

```
data/raw/
  skin_type/{train,valid,test}/{dry,normal,oily}/
  acne_type/{train,valid,test}/{Whiteheads,Blackheads,Papules,Pustules,Cyst}/
  acne_yolo/{train,valid,test}/{images,labels}/ + data.yaml
```

See docs/DATASETS.md for sources, class counts, licenses, and known risks.
