# Backup Manifest

This repository keeps the source code and the small runtime model needed for normal image-processing inference.

## Safe In GitHub

These are included or intended to be included in Git:

- `codes/`
- `src/`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `label_map.json`
- `labels.txt`
- `unknown_threshold.json`
- `codes/train_config.json`
- `codes/herbaura_database/*.json`
- `herbaura_best_float.tflite`

## Back Up Separately

These files and folders are needed if you want to fully retrain or recover every experiment, but they should be backed up outside normal Git:

- `dataset/`
- `test_images/`
- `*.keras`
- `*.h5`
- Extra `.tflite` experiments other than `herbaura_best_float.tflite`
- `models/` model checkpoints

Recommended places:

- Google Drive or OneDrive for `dataset/`
- GitHub Releases or Git LFS for large model artifacts
- An external hard drive for a full project backup

## Minimum Restore For Inference

After cloning from GitHub, inference needs:

- `herbaura_best_float.tflite`
- `label_map.json`
- `labels.txt`
- `unknown_threshold.json`

Install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then run:

```bash
python codes/model_input.py
```

## Minimum Restore For Training

After cloning from GitHub, training needs the dataset restored to:

```text
dataset/raw/
```

Then run:

```bash
python codes/train_cnn.py
```
