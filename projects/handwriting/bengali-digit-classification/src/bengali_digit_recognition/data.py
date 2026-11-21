"""Dataset preparation and TensorFlow input pipeline."""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from .config import CLASS_NAMES, IMAGE_SIZE

SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def collect_class_images(source_dir: str | Path) -> list[tuple[Path, int]]:
    """Collect images from ``source_dir/<class_name>/`` in fixed class order."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[tuple[Path, int]] = []
    missing: list[str] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            missing.append(class_name)
            continue
        paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        samples.extend((path, label) for path in paths)

    if missing:
        raise ValueError(
            "Expected ten class folders named 0 through 9. Missing: " + ", ".join(missing)
        )
    if not samples:
        raise ValueError(f"No supported images found below: {root}")

    counts = Counter(label for _, label in samples)
    insufficient = [CLASS_NAMES[i] for i in range(10) if counts[i] < 3]
    if insufficient:
        raise ValueError(
            "Stratified train/validation/test splitting needs at least three images "
            "per class. Insufficient classes: " + ", ".join(insufficient)
        )
    return samples


def _stratified_split(
    samples: list[tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, int]]]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation and test ratios must be positive and sum to 1.0")

    paths = np.array([str(path) for path, _ in samples], dtype=object)
    labels = np.array([label for _, label in samples], dtype=np.int64)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val_labels,
    )

    def pair(paths_array: Iterable[str], labels_array: Iterable[int]):
        return [(Path(str(path)), int(label)) for path, label in zip(paths_array, labels_array)]

    return {
        "train": pair(train_paths, train_labels),
        "val": pair(val_paths, val_labels),
        "test": pair(test_paths, test_labels),
    }


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, int]:
    """Create stratified split folders and convert source images to 28x28 PNG.

    Expected input layout: ``source_dir/0/image.tif`` through ``source_dir/9``.
    The output contains ``train/``, ``val/`` and ``test/`` class directories.
    """
    random.seed(seed)
    samples = collect_class_images(source_dir)
    splits = _stratified_split(samples, train_ratio, val_ratio, test_ratio, seed)
    destination = Path(output_dir).expanduser().resolve()

    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {destination}. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)

    manifest_rows: list[dict[str, str | int]] = []
    counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split_samples = splits[split_name]
        counts[split_name] = len(split_samples)
        for index, (source_path, label) in enumerate(split_samples):
            class_name = CLASS_NAMES[label]
            class_output = destination / split_name / class_name
            class_output.mkdir(parents=True, exist_ok=True)
            # Prefix with a stable split-local index to avoid duplicate filenames.
            output_name = f"{index:07d}_{source_path.stem}.png"
            output_path = class_output / output_name
            try:
                with Image.open(source_path) as image:
                    grayscale = ImageOps.grayscale(image)
                    resized = grayscale.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                    resized.save(output_path, format="PNG")
            except Exception as exc:  # noqa: BLE001 - include bad source file in error
                raise ValueError(f"Could not decode image {source_path}: {exc}") from exc
            manifest_rows.append({
                "split": split_name,
                "class_name": class_name,
                "label": label,
                "source_path": str(source_path),
                "output_path": str(output_path.relative_to(destination)),
            })

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "label", "source_path", "output_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return counts


def load_datasets(data_dir: str | Path, batch_size: int = 64, seed: int = 42):
    """Load preprocessed train/val/test folders as grayscale TensorFlow datasets."""
    import tensorflow as tf

    root = Path(data_dir)
    datasets = {}
    for split in SPLIT_NAMES:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} split: {split_dir}")
        datasets[split] = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="int",
            class_names=list(CLASS_NAMES),
            color_mode="grayscale",
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=seed,
        )
    return datasets["train"], datasets["val"], datasets["test"]
