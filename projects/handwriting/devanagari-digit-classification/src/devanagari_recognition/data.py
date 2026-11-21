from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np


def _to_python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _infer_image_key(npz: Any) -> str:
    candidates = []
    for key in npz.files:
        value = npz[key]
        if value.ndim in (3, 4) and value.shape[0] > 0:
            candidates.append(key)
    preferred = [k for k in candidates if k.lower() in {"x", "x_train", "images", "image", "data", "arr_0"}]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Could not uniquely infer image array. Available keys: {npz.files}. Pass --x-key.")
    raise ValueError(f"No image-like array found in NPZ. Available keys: {npz.files}")


def _infer_label_key(npz: Any, image_key: str, n_images: int) -> str:
    candidates = []
    for key in npz.files:
        if key == image_key:
            continue
        value = npz[key]
        if value.ndim in (1, 2) and value.shape[0] == n_images:
            candidates.append(key)
    preferred = [k for k in candidates if k.lower() in {"y", "y_train", "labels", "label", "target", "targets", "arr_1"}]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Could not uniquely infer label array. Available keys: {npz.files}. Pass --y-key.")
    raise ValueError(f"No label array matching {n_images} images found. Available keys: {npz.files}")


def load_npz_dataset(path: str | Path, x_key: str | None = None, y_key: str | None = None):
    """Load image and label arrays from an NPZ file; key names may be supplied explicitly."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError(f"NPZ file contains no arrays: {path}")
        image_key = x_key or _infer_image_key(archive)
        if image_key not in archive.files:
            raise KeyError(f"Image key {image_key!r} not found. Available keys: {archive.files}")
        raw_images = np.asarray(archive[image_key])
        label_key = y_key or _infer_label_key(archive, image_key, len(raw_images))
        if label_key not in archive.files:
            raise KeyError(f"Label key {label_key!r} not found. Available keys: {archive.files}")
        images = np.asarray(raw_images)
        labels = np.asarray(archive[label_key])
    if images.ndim == 3:
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped (N,H,W) or (N,H,W,C); received {images.shape}")
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 image channels; received shape {images.shape}")
    images = images.astype("float32")
    if images.max(initial=0) > 1.0:
        images /= 255.0
    if images.min(initial=0) < 0 or images.max(initial=0) > 1.0:
        raise ValueError("Image values must be convertible to the [0, 1] range")
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    labels = labels.reshape(-1)
    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch: {len(images)} images vs {len(labels)} labels")
    if len(images) == 0:
        raise ValueError("Dataset is empty")
    return images, labels


def encode_labels(train_labels: np.ndarray, other_labels: np.ndarray | None = None):
    """Map training labels to contiguous IDs and apply the same mapping to optional labels."""
    classes = np.unique(train_labels)
    mapping = {_to_python(label): index for index, label in enumerate(classes)}
    try:
        encoded_train = np.asarray([mapping[_to_python(v)] for v in train_labels], dtype=np.int64)
        encoded_other = None
        if other_labels is not None:
            missing = sorted({_to_python(v) for v in other_labels} - set(mapping), key=str)
            if missing:
                raise ValueError(f"Test set contains labels absent from training set: {missing}")
            encoded_other = np.asarray([mapping[_to_python(v)] for v in other_labels], dtype=np.int64)
    except TypeError as exc:
        raise ValueError("Labels must contain scalar numeric or string class values") from exc
    class_names = [str(_to_python(v)) for v in classes]
    return encoded_train, encoded_other, class_names


def image_for_model(image: np.ndarray, image_size: int = 32) -> np.ndarray:
    """Prepare one image array for inference; accepts grayscale or RGB arrays."""
    from PIL import Image
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected a grayscale or RGB image, received {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype("float32")
        if arr.max(initial=0) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype("uint8")
    pil = Image.fromarray(arr).resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype="float32") / 255.0
