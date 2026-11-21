import argparse
import json
from pathlib import Path

import tensorflow as tf

from .data import load_test_dataset, load_train_validation
from .model import build_character_cnn


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Urdu character CNN.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/characters_tf")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        raise ValueError("epochs/batch-size must be positive and image-size >= 16")

    tf.keras.utils.set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_train_validation(
        args.train_dir, args.batch_size, args.image_size,
        args.validation_split, args.seed
    )
    test_ds = load_test_dataset(
        args.test_dir, class_names, args.batch_size, args.image_size
    )

    model = build_character_cnn(len(class_names), args.image_size)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )
    model = tf.keras.models.load_model(output_dir / "best_model.keras")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_completed": len(history.history.get("loss", [])),
        "class_count": len(class_names),
        "class_names": class_names,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
