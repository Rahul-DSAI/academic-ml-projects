from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bengali_digit_recognition.data import collect_class_images, prepare_dataset
from bengali_digit_recognition.predict import preprocess_image


def make_source(root: Path, samples_per_class: int = 20) -> None:
    for class_index in range(10):
        class_dir = root / str(class_index)
        class_dir.mkdir(parents=True)
        for sample_index in range(samples_per_class):
            pixels = np.full((35, 31), class_index * 20 + sample_index, dtype=np.uint8)
            Image.fromarray(pixels, mode="L").save(class_dir / f"sample_{sample_index:03d}.tif")


def test_collect_class_images_uses_numeric_folder_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source, samples_per_class=4)
    samples = collect_class_images(source)
    assert len(samples) == 40
    labels = {label for _, label in samples}
    assert labels == set(range(10))


def test_prepare_dataset_creates_disjoint_stratified_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "processed"
    make_source(source, samples_per_class=20)

    counts = prepare_dataset(source, output, seed=7)
    assert sum(counts.values()) == 200
    assert counts == {"train": 160, "val": 20, "test": 20}

    for split in ("train", "val", "test"):
        for class_index in range(10):
            images = list((output / split / str(class_index)).glob("*.png"))
            assert len(images) == counts[split] // 10
            with Image.open(images[0]) as image:
                assert image.mode == "L"
                assert image.size == (28, 28)

    manifest = (output / "split_manifest.csv").read_text(encoding="utf-8")
    assert "source_path" in manifest
    assert manifest.count("\n") == 201


def test_prepare_dataset_refuses_to_overwrite_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "processed"
    make_source(source, samples_per_class=20)
    prepare_dataset(source, output)
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output)


def test_preprocess_image_returns_model_input_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "digit.png"
    Image.fromarray(np.zeros((40, 40), dtype=np.uint8), mode="L").save(image_path)
    result = preprocess_image(image_path)
    assert result.shape == (1, 28, 28, 1)
    assert result.dtype == np.float32
