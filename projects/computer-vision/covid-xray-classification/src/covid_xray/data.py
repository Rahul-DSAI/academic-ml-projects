from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_image_dataset(data_dir: str | Path) -> tuple[List[dict], dict]:
    """Scan class-subdirectory images, reject cross-class duplicates and deduplicate copies.

    Expected layout: data_dir/<class_name>/<image-file>.
    The first lexicographically sorted copy of an exact duplicate is retained.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Expected at least two class subdirectories under the dataset root.")

    class_names = [p.name for p in class_dirs]
    records: List[dict] = []
    seen_hashes: Dict[str, tuple[str, str]] = {}
    duplicates = []
    skipped_non_images = 0

    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in image_paths:
            digest = sha256_file(path)
            relative_path = path.relative_to(root).as_posix()
            previous = seen_hashes.get(digest)
            if previous:
                previous_class, previous_path = previous
                if previous_class != class_dir.name:
                    raise ValueError(
                        "Identical image content appears under different class labels: "
                        f"{previous_path!r} ({previous_class}) and "
                        f"{relative_path!r} ({class_dir.name}). Resolve this label conflict."
                    )
                duplicates.append({"duplicate_path": relative_path, "kept_path": previous_path, "sha256": digest})
                continue
            seen_hashes[digest] = (class_dir.name, relative_path)
            records.append({
                "path": str(path.resolve()),
                "relative_path": relative_path,
                "class_name": class_dir.name,
                "label": label,
                "sha256": digest,
            })

    if not records:
        raise ValueError(f"No supported image files found beneath {root}")
    per_class = defaultdict(int)
    for row in records:
        per_class[row["class_name"]] += 1
    audit = {
        "data_dir": str(root),
        "class_names": class_names,
        "raw_image_count": len(records) + len(duplicates),
        "unique_image_count": len(records),
        "exact_duplicate_count": len(duplicates),
        "class_counts_after_deduplication": dict(per_class),
        "duplicates": duplicates,
    }
    return records, audit


def stratified_split(
    records: Sequence[dict], *, validation_fraction: float = 0.1,
    test_fraction: float = 0.1, seed: int = 42,
) -> List[dict]:
    """Create deterministic per-class train/validation/test splits.

    Duplicate content must already have been removed by scan_image_dataset.
    Fractions are computed within each class, avoiding class-skew from batch splits.
    """
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be > 0 and sum to < 1")
    by_class = defaultdict(list)
    for row in records:
        by_class[row["class_name"]].append(dict(row))

    output: List[dict] = []
    for class_index, class_name in enumerate(sorted(by_class)):
        rows = by_class[class_name]
        if len(rows) < 3:
            raise ValueError(f"Class {class_name!r} needs at least 3 unique images; found {len(rows)}")
        rng = random.Random(seed + class_index)
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * test_fraction)))
        n_val = max(1, int(round(len(rows) * validation_fraction)))
        if n_test + n_val >= len(rows):
            raise ValueError(f"Not enough samples in class {class_name!r} for requested split")
        test_rows = rows[:n_test]
        val_rows = rows[n_test:n_test + n_val]
        train_rows = rows[n_test + n_val:]
        for split, subset in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
            for row in subset:
                row["split"] = split
                output.append(row)
    return output


def write_manifest(rows: Sequence[dict], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "relative_path", "class_name", "label", "sha256", "split"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    return destination


def read_manifest(path: str | Path, split: str | None = None) -> List[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    for row in rows:
        row["label"] = int(row["label"])
    if not rows:
        raise ValueError(f"No rows found in manifest for split={split!r}")
    return rows


def make_tf_dataset(rows: Sequence[dict], *, image_size: int = 256, batch_size: int = 16, training: bool = False):
    """Build a TensorFlow image dataset. TensorFlow is imported lazily."""
    import tensorflow as tf

    paths = [row["path"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (image_size, image_size), antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(rows), 2048), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
