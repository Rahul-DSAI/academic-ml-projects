"""Evaluate a saved CNN once on the untouched Kannada-MNIST test split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .data import load_dataset


def evaluate(data_dir, model_path, output_dir):
    x_train, y_train, x_test, y_test = load_dataset(data_dir)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test, predictions, labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=list(range(10)))
    summary = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_samples": int(len(y_test)),
        "train_samples_available": int(len(y_train)),
        "classification_report": report,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Kannada-MNIST Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", default="models/best_model.keras")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    evaluate(args.data_dir, args.model, args.output_dir)


if __name__ == "__main__":
    main()
