"""Evaluate a saved model on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES
from .data import load_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model", default="artifacts/best_model.keras")
    parser.add_argument("--output", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    args = build_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _, _, test_ds = load_datasets(args.data, args.batch_size)
    model = tf.keras.models.load_model(model_path)
    y_true_batches, y_probability_batches = [], []
    for images, labels in test_ds:
        y_true_batches.append(labels.numpy())
        y_probability_batches.append(model.predict(images, verbose=0))
    if not y_true_batches:
        raise ValueError("The test dataset is empty")

    y_true = np.concatenate(y_true_batches)
    probabilities = np.concatenate(y_probability_batches)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_samples": int(len(y_true)),
    }
    report = classification_report(
        y_true, y_pred, labels=list(range(10)), target_names=list(CLASS_NAMES),
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label", title="Test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Evaluation outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
