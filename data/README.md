# Data

The images are not included in this repository. The dataset is licensed CC BY-NC-SA 4.0, so it is linked here rather than redistributed.

**Dataset:** Maize_in_Field_Dataset, Kaggle
https://www.kaggle.com/datasets/hamishcrazeai/maize-in-field-dataset

**Paper:** Craze, H.A., Pillay, N., Joubert, F. and Berger, D.K. (2022) "Deep learning diagnostics of gray leaf spot in maize under mixed disease field conditions", *Plants*, 11(15), 1942. https://doi.org/10.3390/plants11151942

## Download

Either download the zip from the Kaggle page above, or use the Kaggle CLI (needs a Kaggle account and API token):

```bash
kaggle datasets download -d hamishcrazeai/maize-in-field-dataset
```

The unzipped images are large (about 11 GB on my machine), so check your free disk space first.

## Where to put it

The notebook reads from `../data` relative to `notebooks/`, so the files must end up like this:

```
data/
├── README.md          (this file)
├── Database.csv       (one row per image, 2355 rows)
└── leaf_images/       (2355 .JPG/.jpg files, no subfolders)
```

If the Kaggle zip unpacks into a different folder structure, move `Database.csv` and the image files so they match the layout above. The notebook looks up each image by the filename in the `filePath` column of `Database.csv`, directly inside `data/leaf_images/`.

Everything in `data/` except this README is ignored by git.
