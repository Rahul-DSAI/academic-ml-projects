from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import make_tf_dataset, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved chest X-ray classifier on the held-out test split.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    root = Path(args.artifacts_dir)
    config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    rows = read_manifest(root / "split_manifest.csv", split="test")
    dataset = make_tf_dataset(rows, image_size=int(config["image_size"]), batch_size=args.batch_size)
    model = tf.keras.models.load_model(root / "best_model.keras")
    probabilities = model.predict(dataset, verbose=1)
    y_true = np.asarray([row["label"] for row in rows])
    y_pred = probabilities.argmax(axis=1)
    class_names = config["class_names"]

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    (root / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(root / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    summary = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_images": int(len(y_true)),
        "class_names": class_names,
        "confusion_matrix_path": str(root / "confusion_matrix.csv"),
    }
    (root / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
