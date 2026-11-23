<div align="center">
  <img src="docs/project-banner.svg" alt="COVID-19 Chest X-ray Classification" width="100%" />

  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![Task](https://img.shields.io/badge/Task-4--class%20image%20classification-168A75)](#project-overview)
  [![Status](https://img.shields.io/badge/Status-Research%20prototype-64748b)](#limitations-and-responsible-use)
</div>

# COVID-19 Chest X-ray Classification with CNNs

A TensorFlow/Keras computer-vision project exploring four-class classification of chest X-ray images into **COVID**, **Lung Opacity**, **Normal**, and **Viral Pneumonia**. The repository consolidates a set of earlier CNN experiments into a reproducible training, validation, evaluation, and inference pipeline.

> **Medical-use limitation:** This is an educational/research prototype. It is not clinically validated, must not be used to diagnose or rule out COVID-19 or any other condition, and must not guide treatment or triage decisions.

## Dataset snapshot

The supplied archive contained 4,032 PNG files across four folders, with 1,008 files per class. An exact SHA-256 audit found 17 duplicate copies; the cleaned scanner retains one copy of each identical image, leaving 4,015 unique files.

| Class folder | Files received | Exact copies excluded | Unique files |
|---|---:|---:|---:|
| `COVID` | 1,008 | 12 | 996 |
| `Lung_Opacity` | 1,008 | 0 | 1,008 |
| `Normal` | 1,008 | 0 | 1,008 |
| `Viral Pneumonia` | 1,008 | 5 | 1,003 |
| **Total** | **4,032** | **17** | **4,015** |

The images inspected in the archive were 299×299 pixels; most were grayscale. All images are converted to three-channel tensors consistently by the input pipeline.

**Dataset files are not included in this repository package.** Check the source provider's terms and obtain the data independently. See [`data/README.md`](data/README.md).

## Workflow

<img src="docs/architecture.svg" alt="Chest X-ray classification workflow" width="100%" />

1. Scan the class-folder dataset and compute SHA-256 hashes.
2. Remove exact duplicate files and stop on identical content assigned to conflicting labels.
3. Create deterministic, stratified train/validation/test file splits and save a manifest.
4. Decode images, resize, convert to three channels, and normalize to `[0, 1]`.
5. Train a compact CNN with training-only class weights, checkpointing, early stopping, and learning-rate reduction.
6. Evaluate once on the held-out test split and save per-class metrics and a confusion matrix.

## Class labels

Class names are discovered from the sorted directory names at runtime and saved in `class_names.json`. For the supplied archive, the alphabetical label order is:

| Label | Directory name |
|---:|---|
| 0 | `COVID` |
| 1 | `Lung_Opacity` |
| 2 | `Normal` |
| 3 | `Viral Pneumonia` |

Do not change folder names between training and inference without retraining or updating the saved class mapping.

## Repository structure

```text
covid-xray-classification/
├── data/                         # Dataset placement instructions; raw images ignored
├── docs/
│   ├── architecture.svg
│   ├── original-code-review.md
│   └── project-banner.svg
├── experiments/
│   └── original_scripts/         # Original uploaded Python experiments
├── src/covid_xray/
│   ├── data.py                   # Scan, deduplicate, split, tf.data pipeline
│   ├── model.py                  # CNN architecture
│   ├── train.py                  # Training and checkpointing
│   ├── evaluate.py               # Held-out test evaluation
│   └── predict.py                # Single-image experimental inference
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Python 3.10–3.12 is recommended. A CUDA-enabled GPU can accelerate training, but a compatible TensorFlow/CUDA setup is hardware- and platform-dependent.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Prepare the data

Place the four class folders beneath a root directory, for example:

```text
data/raw/
├── COVID/
├── Lung_Opacity/
├── Normal/
└── Viral Pneumonia/
```

The loader accepts PNG, JPG/JPEG, BMP, and TIFF files. It will identify exact duplicate content before creating splits. If identical image content appears under different labels, the scan stops and asks you to resolve the conflicting labels.

## Train

From the repository root:

```bash
python -m covid_xray.train \
  --data-dir data/raw \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 16 \
  --image-size 256 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 42
```

Windows PowerShell can use the same command on one line or replace the continuation characters with backticks.

Training writes:

- `artifacts/split_manifest.csv`
- `artifacts/dataset_audit.json`
- `artifacts/class_names.json`
- `artifacts/best_model.keras` and `artifacts/final_model.keras`
- `artifacts/training_history.csv`
- `artifacts/training_config.json`

## Evaluate

```bash
python -m covid_xray.evaluate --artifacts-dir artifacts --batch-size 16
```

This evaluates only rows marked `test` in the saved manifest. The script writes `test_metrics.json`, `classification_report.json`, and `confusion_matrix.csv`.

## Predict one image

```bash
python -m covid_xray.predict path/to/chest-xray.png --artifacts-dir artifacts
```

The output is the model's class-score ranking, not a clinical result. Do not use it to make medical decisions.

## Reproducibility and interpretation

- The file-level split is stratified and deterministic for a fixed seed.
- Exact byte-identical duplicates are removed before splitting.
- Patient/study identifiers were not included in the supplied archive. Therefore, the split cannot guarantee patient-level independence; multiple images from one patient may still cross splits if such images exist.
- All reported test metrics must be generated by running the supplied evaluation command. The uploaded archive contained no trained checkpoints, saved training history, or verified evaluation report, so this README makes no accuracy claims.
- The four folders describe dataset labels, not a complete clinical differential diagnosis.

## Tests

The data scanning and splitting logic is testable without installing TensorFlow:

```bash
pip install pytest
pytest -q
```

## Limitations and responsible use

- This work is not a medical device and has not undergone clinical validation.
- Chest X-ray images can contain acquisition-site, scanner, demographic, and dataset-specific shortcuts. High internal test performance alone does not establish external validity.
- A patient-level or institution-level external test set is required for meaningful generalization analysis.
- Do not upload protected health information or redistribute images without confirming the dataset licence and privacy conditions.
- Predictions must not be used for diagnosis, screening, triage, treatment, or reassurance.

## Original experiments

The supplied scripts are retained under `experiments/original_scripts/`. Their issues and the reasoning behind the cleaned implementation are documented in [`docs/original-code-review.md`](docs/original-code-review.md).

## Author

**Rahul Sinha** — AI Engineer | Computer Vision & Applied Machine Learning

GitHub: [Rahul-DSAI](https://github.com/Rahul-DSAI)
