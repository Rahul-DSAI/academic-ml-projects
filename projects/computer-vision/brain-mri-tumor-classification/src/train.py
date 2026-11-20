"""Train the CNN using Training/ and a stratified validation split."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import DEFAULT_CONFIG
from src.data import make_train_validation_datasets
from src.model import build_cnn
from src.utils import save_json, set_random_seeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_CONFIG.dataset_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument(
        "--validation-split", type=float, default=DEFAULT_CONFIG.validation_split
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    return parser.parse_args()


def save_training_plots(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    save_json(history_dict, output_dir / "training_history.json")

    epochs = range(1, len(history.history["accuracy"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = replace(
        DEFAULT_CONFIG,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        random_seed=args.seed,
    )

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    set_random_seeds(config.random_seed)
    tf.keras.utils.set_random_seed(config.random_seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = make_train_validation_datasets(
        config.train_dir,
        config.image_size,
        config.batch_size,
        config.validation_split,
        config.random_seed,
    )

    model = build_cnn(config.image_size, num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(config.output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    # Save the final in-memory model too. best_model.keras is selected by validation loss.
    model.save(config.output_dir / "final_model.keras")
    save_json(
        {
            "class_names": class_names,
            "class_to_index": {name: i for i, name in enumerate(class_names)},
            "image_size": config.image_size,
            "normalization": "RGB float32 pixels scaled to [0, 1]",
            "architecture": "CNN based on the supplied 128x128 MRI notebook",
            "random_seed": config.random_seed,
            "validation_split": config.validation_split,
        },
        config.output_dir / "metadata.json",
    )
    save_training_plots(history, config.output_dir)

    best_epoch = int(min(
        range(len(history.history["val_loss"])),
        key=lambda index: history.history["val_loss"][index],
    )) + 1
    print(f"\nTraining complete. Best validation-loss epoch: {best_epoch}")
    print(f"Artifacts saved in: {config.output_dir.resolve()}")
    print("Run evaluation separately with: python -m src.evaluate")


if __name__ == "__main__":
    main()
