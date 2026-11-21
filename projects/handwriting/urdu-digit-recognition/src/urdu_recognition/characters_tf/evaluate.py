import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .data import load_test_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Urdu character model.")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--output-dir", default="artifacts/characters_tf/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=28)
    args = parser.parse_args()

    class_names = json.loads(Path(args.class_names).read_text(encoding="utf-8"))
    dataset = load_test_dataset(args.test_dir, class_names, args.batch_size, args.image_size)
    model = tf.keras.models.load_model(args.model)

    y_true, y_pred = [], []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "samples": len(y_true)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
