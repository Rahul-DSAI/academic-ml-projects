
# BART-Based Automated Short-Answer Scoring

### Fine-tuning BART for Automated Assessment of Student Responses

An academic NLP project exploring Transformer-based
automated scoring of student-written responses using
BART, PyTorch and Hugging Face Transformers.

## 🎯 Project Overview

Automated scoring of written responses is a challenging
NLP problem involving semantic understanding, language
quality and alignment with expected answers.

This project explores fine-tuning a pretrained BART
model for four-class automated response scoring using
the Hewlett Foundation's ASAP-SAS dataset.

The original implementation was developed in 2024 as
an academic machine learning project.

## 🧠 Model Architecture

- Base model: facebook/bart-large-cnn
- Architecture: BART sequence classification
- Number of output classes: 4
- Target label: Score1
- Classification objective: Cross-Entropy Loss

The pretrained BART checkpoint is adapted to the
response-scoring task through supervised fine-tuning.

## ⚙️ Technical Stack

- Python
- PyTorch
- Hugging Face Transformers
- BART
- Pandas
- Scikit-learn
- NLTK
- NumPy
- Matplotlib
- Seaborn
- CUDA / Mixed Precision Training

## 🔬 Methodology

### 1. Data Processing

- Load response and scoring information.
- Preprocess response text.
- Tokenize using the BART tokenizer.
- Apply sequence truncation and padding.
- Prepare classification labels.

### 2. Model Training

The original training implementation includes:

- AdamW optimizer
- Learning rate: 5e-5
- Batch size: 8
- Maximum sequence length: 512
- Cross-Entropy Loss
- Gradient accumulation
- Mixed precision training
- GPU acceleration

### 3. Model Evaluation

The historical project explored:

- Classification accuracy
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Pearson correlation
- Cohen's Kappa
- Quadratic Weighted Kappa (QWK)

Final performance metrics will be published after
the evaluation pipeline has been reviewed and
reproduced.

## 🏗️ Project Workflow

1. Student response input
2. Text preprocessing
3. BART tokenization
4. Transformer-based representation learning
5. Classification head
6. Predicted response score
7. Evaluation and error analysis

## 📊 Experiments

The original project includes multiple training
experiments, hyperparameter configurations,
evaluation notebooks and visualisation materials.

These historical experiments are being consolidated
into a reproducible training and evaluation pipeline.

## 📁 Repository Structure

```text
bart-automated-short-answer-scoring/
├── configs/
├── src/
├── notebooks/
├── tests/
├── assets/
├── docs/
├── results/
├── README.md
└── requirements.txt
```

## 📚 Dataset

Dataset: The Hewlett Foundation - Short Answer Scoring

Official source:
https://www.kaggle.com/competitions/asap-sas/data

The dataset is subject to Kaggle competition rules.
It is not redistributed in this repository.

Users must obtain the data independently and comply
with the applicable competition rules.

## ⚠️ Limitations

- Historical experiments require reproducibility checks.
- Performance may vary across response sets.
- The model is not validated for real-world grading decisions.
- Predicted scores should not replace human assessment.
- Attention visualisations are not, by themselves,
  validated explanations of model decisions.

## 🚀 Future Improvements

- Reproducible training pipeline
- Better handling of class imbalance
- Prompt-aware scoring
- Robust validation and cross-validation
- Comparative model evaluation
- Model explainability experiments
- Structured feedback generation
- API-based inference

## 👨‍💻 Author

Rahul Sinha

AI Engineer | NLP | Generative AI | Applied AI Research

GitHub: https://github.com/Rahul-DSAI

LinkedIn:
https://www.linkedin.com/in/rahul-sinha99/

---

Academic project developed in 2024.
