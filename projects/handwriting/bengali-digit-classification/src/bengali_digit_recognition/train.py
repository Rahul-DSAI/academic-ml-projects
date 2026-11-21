"""Train the CNN and save the best model and training diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .data import load_datasets
from .model import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed", help="Directory with train/val/test folders")
    parser.add_argument("--output", default="artifacts", help="Directory for model and training outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def main() -> None:
    import tensorflow as tf
    import matplotlib.pyplot as plt

    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _ = load_datasets(args.data, args.batch_size, args.seed)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(output_dir / "final_model.keras")

    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    (output_dir / "history.json").write_text(
        json.dumps(history_dict, indent=2), encoding="utf-8"
    )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("loss", []), label="Training")
    axes[0].plot(history.history.get("val_loss", []), label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history.get("accuracy", []), label="Training")
    axes[1].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    print(f"Best checkpoint: {best_path}")
    print(f"Training outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
