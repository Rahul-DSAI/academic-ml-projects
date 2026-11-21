from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .data import load_npz_dataset, encode_labels
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for Devanagari numeral recognition")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--x-key", default=None, help="Image array key (auto-detected when omitted)")
    parser.add_argument("--y-key", default=None, help="Label array key (auto-detected when omitted)")
    parser.add_argument("--test-x-key", default=None)
    parser.add_argument("--test-y-key", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    x_train_all, y_train_raw = load_npz_dataset(args.train_npz, args.x_key, args.y_key)
    x_test, y_test_raw = load_npz_dataset(args.test_npz, args.test_x_key or args.x_key, args.test_y_key or args.y_key)
    y_train_all, y_test, class_names = encode_labels(y_train_raw, y_test_raw)
    if len(class_names) != 10:
        raise ValueError(f"Expected 10 numeral classes; found {len(class_names)}: {class_names}")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all, y_train_all, test_size=0.15, random_state=args.seed, stratify=y_train_all
    )

    model = build_model(num_classes=len(class_names))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(out / "training_log.csv"),
    ]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, callbacks=callbacks, shuffle=True,
    )
    model.save(out / "final.keras")
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    results = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
               "train_samples": int(len(x_train)), "validation_samples": int(len(x_val)),
               "test_samples": int(len(x_test)), "class_names": class_names}
    (out / "training_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)

if __name__ == "__main__":
    main()
