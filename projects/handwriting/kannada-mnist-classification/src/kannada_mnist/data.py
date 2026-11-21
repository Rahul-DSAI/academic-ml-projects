"""Dataset loading and deterministic train/validation splitting."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


EXPECTED_FILES = {
    "x_train": "X_kannada_MNIST_train.npz",
    "y_train": "y_kannada_MNIST_train.npz",
    "x_test": "X_kannada_MNIST_test.npz",
    "y_test": "y_kannada_MNIST_test.npz",
}
NUM_CLASSES = 10


def load_npz_array(path: str | Path) -> np.ndarray:
    """Read an array from an NPZ archive, preferring the original arr_0 key."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "arr_0" in archive.files:
            arr = archive["arr_0"]
        elif len(archive.files) == 1:
            arr = archive[archive.files[0]]
        else:
            raise ValueError(
                f"Expected an 'arr_0' array or a single-array NPZ file: {path}; "
                f"available keys: {archive.files}"
            )
    return np.asarray(arr)


def prepare_images(images: np.ndarray) -> np.ndarray:
    """Convert (N,H,W) or (N,H,W,1) grayscale images to float32 [0,1]."""
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    elif images.ndim == 4 and images.shape[1] == 1 and images.shape[-1] != 1:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError(
            "Expected grayscale images shaped (N,H,W), (N,H,W,1), "
            f"or (N,1,H,W); received {images.shape}"
        )
    images = images.astype(np.float32, copy=False)
    if not np.isfinite(images).all():
        raise ValueError("Images contain NaN or infinite values")
    if images.size and images.max() > 1.0:
        images = images / 255.0
    if images.size and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("Image values must be in [0,1] or [0,255]")
    return images


def prepare_labels(labels: np.ndarray) -> np.ndarray:
    """Flatten labels and validate the ten digit classes (0 through 9)."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("Label array is empty")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError("Labels must be integer class IDs")
    labels = labels.astype(np.int64)
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"Labels must be between 0 and {NUM_CLASSES - 1}")
    return labels


def load_dataset(data_dir: str | Path):
    """Load the four official Kannada-MNIST NPZ files from a directory."""
    data_dir = Path(data_dir)
    arrays = {
        name: load_npz_array(data_dir / filename)
        for name, filename in EXPECTED_FILES.items()
    }
    x_train = prepare_images(arrays["x_train"])
    y_train = prepare_labels(arrays["y_train"])
    x_test = prepare_images(arrays["x_test"])
    y_test = prepare_labels(arrays["y_test"])
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Image and label counts do not match")
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError("Train and test image shapes do not match")
    return x_train, y_train, x_test, y_test


def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_fraction: float = 0.1,
    seed: int = 42,
):
    """Create a reproducible stratified split; the test set stays untouched."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(x_train) != len(y_train):
        raise ValueError("Image and label counts do not match")
    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train,
    )
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]
