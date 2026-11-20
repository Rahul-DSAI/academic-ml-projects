"""BART sequence-classification model and tokenizer loading."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL2ID = {f"SCORE_{label}": label for label in range(4)}
ID2LABEL = {label: f"SCORE_{label}" for label in range(4)}


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def load_model(model_name_or_path: str, num_labels: int = 4):
    if num_labels != 4:
        raise ValueError("The supplied ASAP-SAS training file contains four Score1 classes (0–3).")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        problem_type="single_label_classification",
    )
    return model


def load_finetuned_model(model_dir: str | Path):
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Fine-tuned model directory not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if model.config.num_labels != 4:
        raise ValueError(
            f"Expected a 4-class model, but checkpoint declares {model.config.num_labels} labels."
        )
    return tokenizer, model
