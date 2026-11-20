"""Dataset discovery, stratified splitting, and TensorFlow input pipelines."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
AUTOTUNE = tf.data.AUTOTUNE


def discover_class_names(directory: Path) -> list[str]:
    """Return class directory names in a stable alphabetical order."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}. See README.md for setup instructions."
        )
    classes = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected at least two class folders in {directory}, found: {classes}")
    return classes


def collect_samples(
    directory: Path, class_names: Optional[Sequence[str]] = None
) -> tuple[list[str], np.ndarray, list[str]]:
    """Collect image paths and integer labels from class-named subdirectories."""
    discovered = discover_class_names(directory)
    names = list(class_names) if class_names is not None else discovered

    missing = sorted(set(names) - set(discovered))
    unexpected = sorted(set(discovered) - set(names))
    if missing or unexpected:
        raise ValueError(
            f"Class folders in {directory} do not match training classes. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    class_to_index = {name: index for index, name in enumerate(names)}
    paths: list[str] = []
    labels: list[int] = []

    for class_name in names:
        class_dir = directory / class_name
        class_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(f"No supported images found in {class_dir}")
        paths.extend(str(path) for path in class_paths)
        labels.extend([class_to_index[class_name]] * len(class_paths))

    return paths, np.asarray(labels, dtype=np.int32), names


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    # resize returns float32 values on the original 0..255 intensity scale.
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.int32)


def make_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def make_train_validation_datasets(
    train_dir: Path,
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Create a reproducible, class-stratified split of the Training folder."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")

    paths, labels, class_names = collect_samples(train_dir)
    class_counts = np.bincount(labels, minlength=len(class_names))
    if np.any(class_counts < 2):
        raise ValueError("Each class needs at least two images for a stratified split")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
    )

    train_ds = make_dataset(
        train_paths, train_labels, image_size, batch_size, True, seed
    )
    val_ds = make_dataset(
        val_paths, val_labels, image_size, batch_size, False, seed
    )
    return train_ds, val_ds, class_names


def make_test_dataset(
    test_dir: Path,
    class_names: Sequence[str],
    image_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, np.ndarray, list[str]]:
    paths, labels, names = collect_samples(test_dir, class_names=class_names)
    dataset = make_dataset(paths, labels, image_size, batch_size, False, seed)
    return dataset, labels, names
