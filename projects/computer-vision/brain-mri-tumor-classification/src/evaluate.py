"""Evaluate a saved CNN on the official Testing/ split."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import DEFAULT_CONFIG
from src.data import make_test_dataset
from src.utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_CONFIG.dataset_dir)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CONFIG.output_dir / "best_model.keras")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CONFIG.output_dir / "metadata.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first.")
    if not args.metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {args.metadata_path}")

    metadata = load_json(args.metadata_path)
    class_names = metadata["class_names"]
    image_size = int(metadata["image_size"])
    test_ds, y_true, _ = make_test_dataset(
        args.dataset_dir / "Testing",
        class_names,
        image_size,
        args.batch_size,
        DEFAULT_CONFIG.random_seed,
    )

    model = tf.keras.models.load_model(args.model_path)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "number_of_test_images": int(len(y_true)),
        "class_names": class_names,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output_dir / "evaluation_metrics.json")
    save_json(report, args.output_dir / "classification_report.json")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    # Per-class one-vs-rest ROC. Skip a class if the test labels do not contain
    # both positive and negative examples for that class.
    y_true_binary = label_binarize(y_true, classes=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    auc_values = {}
    for index, class_name in enumerate(class_names):
        binary_labels = y_true_binary[:, index]
        if np.unique(binary_labels).size < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels, probabilities[:, index]
        )
        area = roc_auc_score(binary_labels, probabilities[:, index])
        auc_values[class_name] = float(area)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} (AUC={area:.3f})")

    axis.plot([0, 1], [0, 1], "--", label="Chance")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="One-vs-rest ROC curves",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(args.output_dir / "roc_curves.png", dpi=160)
    plt.close(figure)

    metrics["per_class_roc_auc"] = auc_values
    save_json(metrics, args.output_dir / "evaluation_metrics.json")
    print("\nTest evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nReports and plots saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
