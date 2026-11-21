"""Train the Kannada-MNIST CNN."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import tensorflow as tf

from .data import load_dataset, split_train_validation
from .model import build_model


def train(data_dir, output_dir, epochs=50, batch_size=128, seed=42, validation_fraction=0.1):
    tf.keras.utils.set_random_seed(seed)
    x_train, y_train, _, _ = load_dataset(data_dir)
    x_tr, y_tr, x_val, y_val = split_train_validation(
        x_train, y_train, validation_fraction=validation_fraction, seed=seed
    )

    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model.keras"

    model = build_model(input_shape=x_tr.shape[1:])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
    ]
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(best_model_path)
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    (reports_dir / "training_history.json").write_text(
        json.dumps(history_dict, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint saved to: {best_model_path}")
    print(f"Training samples: {len(x_tr)} | Validation samples: {len(x_val)}")
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing the four Kannada-MNIST NPZ files")
    parser.add_argument("--output-dir", default=".", help="Project output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.seed, args.validation_fraction)


if __name__ == "__main__":
    main()
