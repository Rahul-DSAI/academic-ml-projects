 
# BART-Based Automated Short-Answer Scoring

<p align="center">
  <strong>Automated Assessment of Student Responses Using Transformer-Based Language Models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Model-BART-orange" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/Task-Text%20Classification-purple" />
  <img src="https://img.shields.io/badge/Status-Academic%20Project-lightgrey" />
</p>

An academic NLP project developed in 2024 exploring automated short-answer scoring using a fine-tuned BART model, PyTorch and Hugging Face Transformers.

The project investigates how pretrained Transformer architectures can be adapted to evaluate student-written responses and predict ordinal score categories.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-project-objectives)
- [Model Architecture](#-model-architecture)
- [Technical Stack](#-technical-stack)
- [Original Coursework Implementation](#-original-coursework-implementation)
- [Cleaned Implementation](#-cleaned-implementation)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Model Availability](#-model-availability)
- [Evaluation Metrics](#-evaluation-metrics)
- [Experiments and Results](#-experiments-and-results)
- [Limitations](#-limitations)
- [Responsible Use](#-responsible-use)
- [Future Improvements](#-future-improvements)
- [Tests](#-tests)
- [Author](#-author)

---

## 🎯 Project Overview

Automated short-answer scoring is an NLP task that involves evaluating student responses and predicting scores based on their content.

Traditional assessment requires human evaluation, which can be time-consuming and difficult to scale.

This project explores a Transformer-based approach using BART for four-class student-response scoring.

The implementation uses the ASAP-SAS dataset and trains a sequence-classification model to predict the first human-assigned score (`Score1`).

### Key Features

- BART-based sequence classification.
- Fine-tuning of a pretrained language model.
- Four-class score prediction.
- Text preprocessing and tokenisation.
- GPU-accelerated training.
- Model evaluation using classification and ordinal metrics.
- Prediction on new student responses.
- Reproducible training and evaluation pipeline.

---

## 🎯 Project Objectives

1. Explore Transformer-based architectures for automated response scoring.
2. Fine-tune a pretrained BART model for a downstream classification task.
3. Predict student-response scores across four categories.
4. Evaluate model performance using appropriate classification and ordinal metrics.
5. Build a reusable pipeline for training, evaluation and inference.

---

## 🧠 Model Architecture

**Base Model:** `facebook/bart-large-cnn`

**Architecture:** BART Sequence Classification

**Task:** Four-class text classification

**Target:** `Score1`

**Output Classes:** 0, 1, 2, 3

### Workflow

```text
Student Response
       |
       v
Text Preprocessing
       |
       v
BART Tokenizer
       |
       v
Input IDs and Attention Mask
       |
       v
BART Encoder
       |
       v
Classification Head
       |
       v
Four-Class Prediction
       |
       v
Predicted Score (0–3)
```

The pretrained BART checkpoint is adapted through supervised fine-tuning. A classification head maps the learned text representation to four score categories.

---

## 🛠️ Technical Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Pretrained Model | BART |
| Tokenisation | BART Tokenizer |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Training | AdamW, CUDA, Mixed Precision |
| Testing | Pytest |

---

## 📜 Original Coursework Implementation

The original academic implementation included multiple notebooks and Python training experiments.

The main implementation, `essay_version10.py`, used the following configuration:

| Parameter | Original configuration |
|---|---|
| Pretrained model | `facebook/bart-large-cnn` |
| Target column | `Score1` |
| Number of classes | 4 |
| Maximum sequence length | 512 |
| Optimiser | AdamW |
| Learning rate | `5e-5` |
| Batch size | 8 |
| Gradient accumulation | 8 steps |
| Training epochs | 10 |
| Loss function | CrossEntropyLoss |

The original implementation reported classification accuracy and included additional evaluation and visualisation experiments.

These historical settings are documented separately from the cleaned implementation.

---

## ⚙️ Cleaned Implementation

The repository reorganises the original coursework into modular components for data processing, model training, evaluation, metrics and inference.

### Improvements

**1. Data Processing**
- Structured TSV data loading.
- Explicit target-label validation.
- Consistent four-class label mapping.
- Dynamic batch padding.
- Stratified train, validation and test splits.

**2. Model Training**
- Modular BART sequence-classification implementation.
- Configurable training parameters.
- Gradient accumulation.
- GPU support.
- Validation-based checkpoint selection.

**3. Evaluation**
- Accuracy, MAE and MSE.
- Cohen's Kappa.
- Quadratic Weighted Kappa.
- Confusion matrix.
- Per-class precision, recall and F1-score.

**4. Model Management**
- Best-checkpoint saving.
- Training history.
- Evaluation metrics.
- Prediction outputs.

### Important Methodological Differences

The cleaned implementation intentionally differs from the historical coursework:

- Preserves stopwords rather than removing them.
- Uses dynamic padding instead of padding every example to 512 tokens.
- Uses stratified train, validation and test splits.
- Selects the best checkpoint using validation QWK.
- Evaluates the selected model on a held-out test split.
- Records explicit `SCORE_0` through `SCORE_3` mappings.

This cleaned implementation is a reproducible baseline, not a claim that the historical coursework metrics have been reproduced.

---

## 📂 Project Structure

```text
bart-automated-short-answer-scoring/
│
├── configs/
│   └── training_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── engine.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   └── test_metrics.py
│
├── docs/
│   └── methodology.md
│
├── results/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

**Dataset:** The Hewlett Foundation – Short Answer Scoring (ASAP-SAS)

**Source:** Kaggle

[Access the official ASAP-SAS competition dataset](https://www.kaggle.com/competitions/asap-sas/data)

### Dataset Fields

| Column | Description |
|---|---|
| `EssayText` | Student-written response |
| `Score1` | First human-assigned score |
| `Score2` | Second rater's score |
| `EssaySet` | Response-set or prompt identifier |

### Target Variable

This implementation uses `Score1` as the prediction target.

`Score2` is not used as the training target.

`EssaySet` is retained for split diagnostics and per-prompt analysis, but the current model does not use it as an input feature.

### Dataset Preparation

Place the dataset locally:

```text
data/
└── raw/
    └── train.tsv
```

Required columns:

```text
EssayText
Score1
EssaySet
```

### Dataset Distribution Notice

The dataset, student responses, leaderboard files and competition materials are not redistributed in this repository.

Users must obtain the dataset directly from Kaggle and comply with its applicable competition rules.

---

## 💻 Installation

### Requirements

- Python 3.10 or 3.11
- PyTorch
- Hugging Face Transformers
- CUDA-compatible GPU recommended for training

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rahul-DSAI/academic-ml-projects.git

cd academic-ml-projects/projects/nlp/bart-automated-short-answer-scoring
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install PyTorch

Install the appropriate PyTorch version for your hardware:

https://pytorch.org/get-started/locally/

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run the training pipeline:

```bash
python -m src.train \
    --config configs/training_config.yaml
```

Optional overrides:

```bash
python -m src.train \
    --config configs/training_config.yaml \
    --data data/raw/train.tsv \
    --output-dir artifacts
```

### Training Outputs

The training process saves:

- Best model checkpoint.
- Tokenizer files.
- Training history.
- Split information.
- Evaluation metrics.
- Prediction outputs.

The best checkpoint is stored in:

```text
artifacts/
└── best_model/
```

These generated artifacts are local outputs and are not automatically committed to Git.

---

## 📈 Evaluation

Evaluate the saved checkpoint:

```bash
python -m src.evaluate \
    --config configs/training_config.yaml \
    --model-dir artifacts/best_model
```

The evaluation script reports classification and ordinal scoring metrics.

---

## 🔍 Inference

Predict the score of a new response:

```bash
python -m src.predict \
    --model-dir artifacts/best_model \
    --text "Your sample student response goes here."
```

The model returns:

- Predicted score class.
- Probability distribution across the four score categories.

Example output format:

```text
Predicted Score: 2

Class Probabilities:
Score 0: ...
Score 1: ...
Score 2: ...
Score 3: ...
```

The probabilities shown above are output placeholders, not measured predictions.

---

## 📊 Evaluation Metrics

The project uses the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| MAE | Average absolute score difference |
| MSE | Average squared score difference |
| Cohen's Kappa | Agreement beyond chance |
| Quadratic Weighted Kappa | Ordinal agreement with larger penalties for distant scores |
| Precision | Correctness of predicted classes |
| Recall | Coverage of actual classes |
| F1-score | Balance between precision and recall |

### Why Quadratic Weighted Kappa?

The target scores are ordered categories.

For example, predicting a score of 2 when the actual score is 3 is a smaller disagreement than predicting 0 when the actual score is 3.

Quadratic Weighted Kappa accounts for this ordering and is therefore an important metric for evaluating automated scoring models.

---

## 📉 Experiments and Results

The original coursework included multiple training experiments, evaluation notebooks and visualisation materials.

Historical reported results should be treated separately from the cleaned implementation.

No final performance claim is made here until the cleaned pipeline has been executed and its results independently verified.

Future evaluation reports should include:

- Test accuracy.
- MAE and MSE.
- Quadratic Weighted Kappa.
- Confusion matrix.
- Per-class precision, recall and F1-score.
- Per-`EssaySet` performance.
- Model configuration and hardware details.

---

## ⚠️ Limitations

- The current model uses response text only.
- `EssaySet` is not included as a model feature.
- The current random row-level split is not a held-out-prompt evaluation.
- Performance may differ across prompts and scoring rubrics.
- Historical coursework metrics have not been established as reproducible results for the cleaned pipeline.
- The model is not validated for real-world educational grading.

A separate group-held-out experiment is recommended to evaluate generalisation to unseen prompts.

---

## 🛡️ Responsible Use

This project is an academic NLP experiment.

It should not be treated as a validated educational grading product or used as the sole basis for academic decisions.

Automated scoring systems may reflect biases in training data, scoring rubrics and student populations.

Human assessment and review remain essential for consequential educational decisions.

---

## 🔮 Future Improvements

- Prompt-aware response scoring.
- Group-held-out evaluation.
- Comparison with other Transformer architectures.
- Class imbalance handling.
- Improved hyperparameter optimisation.
- Model calibration.
- Robustness and fairness evaluation.
- Structured feedback generation.
- API-based inference.
- Interactive demonstration interface.

---

## 🧪 Tests

Run the unit tests:

```bash
pytest -q
```

Tests cover the data-processing and evaluation components.

---

## 🤗 Pretrained Model

**Model repository:** `Rahul-DSAI/bart-short-answer-scoring`

[View the model on Hugging Face](https://huggingface.co/Rahul-DSAI/bart-short-answer-scoring)

The model link is intended for the fine-tuned BART checkpoint and its matching tokenizer.

**Model details:**

| Property | Value |
|---|---|
| Base checkpoint | `facebook/bart-large-cnn` |
| Task | Short-answer scoring |
| Architecture | BART Sequence Classification |
| Number of classes | 4 |
| Output labels | 0, 1, 2, 3 |

The model repository should contain the trained weights, model configuration, tokenizer files and a model card.

The checkpoint must be made publicly available only after verifying that its redistribution is permitted under the applicable dataset and competition terms.

---

## 👨‍💻 Author

**Rahul Sinha**

AI Engineer | Agentic AI | Generative AI | NLP | Computer Vision | Applied AI Research

- GitHub: [Rahul-DSAI](https://github.com/Rahul-DSAI)
- LinkedIn: [Rahul Sinha](https://www.linkedin.com/in/rahul-sinha99/)
- Email: [Contact](mailto:rahul.sinha.da@gmail.com)

---

<p align="center">
  <strong>Building intelligent systems through engineering, experimentation and research.</strong>
</p>
