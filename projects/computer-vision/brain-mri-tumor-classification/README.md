# Brain MRI Tumour Classification Using a CNN

A TensorFlow/Keras image-classification project based on the original MRI notebook and two Python experiments. The model classifies MRI images into the four folder labels supplied by the dataset.

> **Important:** This is an educational machine-learning project, not a medical device. Its predictions must not be used to diagnose, rule out, or make treatment decisions about a medical condition.

## Dataset classes

The supplied notebook loads four folders from the Brain Tumor MRI Dataset. The code discovers the folder names at runtime and sorts them alphabetically, rather than hardcoding a potentially incorrect numeric label mapping. For the commonly used dataset layout, these folders are `glioma`, `meningioma`, `notumor`, and `pituitary`.

## Project structure

```text
brain-mri-tumor-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── notebooks/
│   └── MRI_original.ipynb
├── original_files/
│   ├── MRI_original.py
│   └── MRImul_original.py
├── artifacts/      # generated model files; do not commit large weights by default
└── results/        # generated evaluation reports and plots
```

## What is included

- `src/model.py`: 128 × 128 CNN architecture adapted from the supplied notebook (Conv2D, Batch Normalization, Max Pooling, Dropout and Dense layers).
- `src/data.py`: image loading, normalization, class discovery, and deterministic stratified train/validation splitting.
- `src/train.py`: training, best-validation-loss checkpointing, early stopping, learning-rate reduction, CSV logging and learning-curve plots.
- `src/evaluate.py`: evaluation on the separate `Testing` directory, classification metrics, confusion matrix and one-vs-rest ROC curves.
- `src/predict.py`: single-image inference using the saved model and class-name metadata.
- `notebooks/MRI_original.ipynb` and `original_files/`: the three uploaded files retained as original experimental material.

The clean implementation uses the main notebook's 128 × 128 architecture and 60-epoch maximum. Its default batch size is 16 rather than the notebook's 128 to reduce memory requirements. The earlier `MRImul.py` experiment used 64 × 64 images and 20 epochs; `MRI.py` used 128 × 128 images and 60 epochs.

## Dataset setup

The dataset is not included in this repository. Obtain it from the original Kaggle dataset page:

<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

After accepting any applicable Kaggle terms and configuring the Kaggle CLI credentials on your own machine, download and extract it from the repository root:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

The expected directory layout is:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

If Kaggle extracts the two folders into a nested directory, pass that directory to `--dataset-dir` when running the commands below. **Never commit `kaggle.json`, API keys, or the dataset images to GitHub.** Check the dataset's licence and usage terms before redistribution.

## Installation

Python 3.10–3.12 is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS / WSL
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation support can vary by operating system, Python version and hardware. Refer to the TensorFlow installation documentation if the installation command fails: <https://www.tensorflow.org/install>.

## Train

Run from the repository root:

```bash
python -m src.train
```

To override defaults:

```bash
python -m src.train \
  --dataset-dir data \
  --image-size 128 \
  --batch-size 16 \
  --epochs 60 \
  --validation-split 0.10 \
  --seed 42 \
  --output-dir artifacts
```

On Windows PowerShell, enter the command on one line or replace the Unix `\\` line continuations with PowerShell backticks.

The training code splits only the `Training` folder into training and validation partitions. It uses stratification so each class is represented proportionally, with a fixed random seed. The official `Testing` folder is kept separate for final evaluation.

Generated files include:

- `artifacts/best_model.keras` — checkpoint with the lowest validation loss.
- `artifacts/final_model.keras` — model state at the end of training (after early stopping, if triggered).
- `artifacts/metadata.json` — class-name mapping and preprocessing metadata.
- `artifacts/training_log.csv` and `artifacts/training_history.json` — training history.
- `artifacts/training_history.png` — training and validation curves.

## Evaluate

After training:

```bash
python -m src.evaluate
```

Optional arguments:

```bash
python -m src.evaluate \
  --dataset-dir data \
  --model-path artifacts/best_model.keras \
  --metadata-path artifacts/metadata.json \
  --output-dir results
```

Evaluation reports accuracy, weighted precision, weighted recall, weighted F1, macro F1, loss, per-class precision/recall/F1, a confusion matrix and one-vs-rest ROC curves. Results are written under `results/`.

## Predict one image

```bash
python -m src.predict path/to/mri_image.jpg
```

You can also specify another compatible model and metadata file:

```bash
python -m src.predict path/to/mri_image.jpg \
  --model-path artifacts/best_model.keras \
  --metadata-path artifacts/metadata.json
```

The image is resized to the model's configured input dimensions and normalized to `[0, 1]`, matching training preprocessing.

## Original notebook results

The uploaded notebook contains a saved evaluation output reporting approximately:

| Metric | Notebook-reported value |
|---|---:|
| Test images | 1,311 |
| Test accuracy | 98.25% |
| Weighted precision | 98.26% |
| Weighted recall | 98.25% |
| Weighted F1-score | 98.25% |
| Test loss | 0.0938 |
| Overall one-vs-rest ROC AUC | 1.00 (rounded to two decimal places) |

These values are historical outputs from the supplied notebook. They have **not** been reproduced by this cleaned implementation. Re-run training and evaluation before presenting them as results of this code version. The rounded AUC should not be interpreted as an exact value of 1.0.

## Methodology and limitations

- The main notebook used 128 × 128 RGB images, a four-class softmax output, Adam optimization and sparse categorical cross-entropy.
- The cleaned pipeline makes the training/validation split reproducible and stratified. This is a correction to the original scripts' dataset-pipeline split, which could produce inconsistent subsets when reshuffled.
- Folder names determine class labels. The mapping is saved to `artifacts/metadata.json` and must be preserved at inference time.
- Performance on this dataset does not establish generalization to other hospitals, scanners, populations or image-acquisition protocols.
- Dataset leakage, duplicates, class imbalance, external validation and clinically meaningful sensitivity/specificity should be examined before any research or deployment claims.
- A high score on a public image dataset does not establish clinical utility.

## Citation and dataset use

The dataset was downloaded in the original notebook from `masoudnickparvar/brain-tumor-mri-dataset` on Kaggle. Please cite and follow the current Kaggle dataset page's attribution, licence and usage conditions when using it. The dataset itself is intentionally not redistributed here.

## Author

Rahul Sinha  
GitHub: <https://github.com/Rahul-DSAI>  
LinkedIn: <https://www.linkedin.com/in/rahul-sinha99/>
