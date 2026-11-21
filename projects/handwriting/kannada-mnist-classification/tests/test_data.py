import numpy as np
import pytest

from kannada_mnist.data import (
    load_npz_array,
    prepare_images,
    prepare_labels,
    split_train_validation,
)


def test_prepare_images_adds_channel_and_scales():
    images = np.full((20, 28, 28), 255, dtype=np.uint8)
    out = prepare_images(images)
    assert out.shape == (20, 28, 28, 1)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_prepare_images_accepts_already_normalized_images():
    images = np.zeros((4, 28, 28, 1), dtype=np.float32)
    out = prepare_images(images)
    assert out.shape == images.shape
    assert out.max() == 0.0


def test_prepare_labels_flattens_and_validates():
    labels = prepare_labels(np.array([[0], [4], [9]]))
    assert labels.tolist() == [0, 4, 9]
    with pytest.raises(ValueError):
        prepare_labels(np.array([0, 10]))


def test_npz_loading_and_stratified_split_are_reproducible(tmp_path):
    x = np.random.default_rng(2).integers(0, 256, size=(100, 28, 28), dtype=np.uint8)
    y = np.tile(np.arange(10), 10)
    np.savez_compressed(tmp_path / "single.npz", x)
    assert load_npz_array(tmp_path / "single.npz").shape == x.shape
    x = prepare_images(x)
    xtr1, ytr1, xval1, yval1 = split_train_validation(x, y, seed=12)
    xtr2, ytr2, xval2, yval2 = split_train_validation(x, y, seed=12)
    assert len(xtr1) == 90 and len(xval1) == 10
    assert np.array_equal(ytr1, ytr2)
    assert np.array_equal(yval1, yval2)
    assert set(np.unique(yval1)) == set(range(10))
