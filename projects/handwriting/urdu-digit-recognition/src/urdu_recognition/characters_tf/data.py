from pathlib import Path
from typing import Sequence

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def list_class_names(directory: str | Path) -> list[str]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(names) < 2:
        raise ValueError(f"Expected at least two class subdirectories in {root}")
    return names


def _prepare(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)


def load_train_validation(
    train_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 28,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")

    class_names = list_class_names(train_dir)
    common = dict(
        directory=str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **common
    )
    return _prepare(train_ds), _prepare(val_ds), class_names


def load_test_dataset(
    test_dir: str | Path,
    class_names: Sequence[str],
    batch_size: int = 32,
    image_size: int = 28,
) -> tf.data.Dataset:
    test_root = Path(test_dir)
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_root}")
    observed = list_class_names(test_root)
    missing = sorted(set(class_names) - set(observed))
    extra = sorted(set(observed) - set(class_names))
    if missing or extra:
        raise ValueError(
            f"Test class folders do not match training classes. "
            f"Missing: {missing}; extra: {extra}"
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        color_mode="rgb",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    return _prepare(ds)
