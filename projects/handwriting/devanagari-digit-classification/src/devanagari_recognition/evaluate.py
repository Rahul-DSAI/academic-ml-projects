from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .data import load_npz_dataset, encode_labels


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Devanagari digit CNN")
    p.add_argument("--test-npz", required=True); p.add_argument("--model", default="models/best.keras")
    p.add_argument("--classes", default="models/classes.json"); p.add_argument("--x-key"); p.add_argument("--y-key")
    p.add_argument("--output-dir", default="reports")
    a = p.parse_args()
    x_test, y_raw = load_npz_dataset(a.test_npz, a.x_key, a.y_key)
    names = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    # The labels saved in classes.json are the original class labels, in encoded order.
    mapping = {str(v): i for i, v in enumerate(names)}
    y = np.asarray([mapping[str(v.item() if isinstance(v, np.generic) else v)] for v in y_raw], dtype=np.int64)
    model = tf.keras.models.load_model(a.model)
    probs = model.predict(x_test, verbose=0); pred = probs.argmax(axis=1)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y, pred, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0)
    (out / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    np.savetxt(out / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print(f"Accuracy: {accuracy_score(y, pred):.4f}")
    print(classification_report(y, pred, labels=list(range(len(names))), target_names=names, zero_division=0))

if __name__ == "__main__":
    main()
