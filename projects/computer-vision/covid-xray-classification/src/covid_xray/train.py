from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import make_tf_dataset, scan_image_dataset, stratified_split, write_manifest
from .model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an educational chest X-ray CNN classifier.")
    parser.add_argument("--data-dir", required=True, help="Root directory containing one subfolder per class")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model, manifest and reports")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        parser.error("epochs and batch-size must be positive; image-size must be at least 64")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, audit = scan_image_dataset(args.data_dir)
    rows = stratified_split(
        records,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    manifest_path = write_manifest(rows, output_dir / "split_manifest.csv")
    class_names = audit["class_names"]
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    train_counts = Counter(row["label"] for row in train_rows)
    total_train = len(train_rows)
    class_weights = {
        class_id: total_train / (len(class_names) * count)
        for class_id, count in train_counts.items()
    }

    train_ds = make_tf_dataset(train_rows, image_size=args.image_size, batch_size=args.batch_size, training=True)
    val_ds = make_tf_dataset(val_rows, image_size=args.image_size, batch_size=args.batch_size)
    model = build_model(num_classes=len(class_names), image_size=args.image_size)
    callbacks = [
        __import__("tensorflow").keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"), monitor="val_loss", save_best_only=True
        ),
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        __import__("tensorflow").keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(output_dir / "final_model.keras")
    config = {
        "image_size": args.image_size,
        "class_names": class_names,
        "seed": args.seed,
        "manifest": str(manifest_path),
        "class_weights": class_weights,
        "note": "Research/educational classifier; not for clinical diagnosis.",
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
