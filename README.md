# Herbaura Image Processing

Python scripts for training, evaluating, converting, and testing Herbaura plant image classification models.

## What is included in Git

- Source code in `codes/` and `src/`
- Project dependencies in `requirements.txt`
- Small label/config/database JSON files
- The deployment TFLite model `herbaura_best_float.tflite`
- Runtime metadata such as `label_map.json`, `labels.txt`, and `unknown_threshold.json`

## What is intentionally not included

- `dataset/` image folders
- `venv/` virtual environment
- Saved training model files such as `.keras` and `.h5`
- Generated reports such as confusion matrices and audit CSV files

These files are large or generated locally, so they should not be committed directly to GitHub. Put large model files in GitHub Releases, Google Drive, or Git LFS if you need to share them.

## Setup

For Windows teammates, double-click:

```text
setup_windows.bat
```

Then put the Kaggle dataset class folders inside:

```text
dataset/raw/
```

To train, double-click:

```text
run_training_windows.bat
```

To quickly check the included TFLite model, double-click:

```text
run_tflite_check_windows.bat
```

Manual setup:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Common scripts

Final training script:

```bash
python codes/train_cnn.py
```

Convert a Keras model to TFLite:

```bash
python codes/convert_to_tflite.py --model herbaura_best.keras --output herbaura_best_float.tflite
```

Predict with the trained Keras model:

```bash
python codes/test_model.py --model herbaura_best.keras --labels labels.txt --dataset_dir dataset/raw
```

Evaluate a TFLite model:

```bash
python codes/tflite_evaluate.py
```

## Restore After Downloading From GitHub

To run inference after cloning this repository, keep these files in the project root:

- `herbaura_best_float.tflite`
- `label_map.json`
- `labels.txt`
- `unknown_threshold.json`

To train again after cloning, restore the image dataset separately into:

```text
dataset/raw/
```

The training config uses this relative path, so it can run from any folder after the dataset is restored.

If the dataset is hosted on Kaggle, download and unzip it so the class folders are directly inside `dataset/raw/`.
