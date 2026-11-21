import numpy as np
import pytest
from devanagari_recognition.data import encode_labels, load_npz_dataset, image_for_model


def test_encode_labels_and_test_mapping():
    train, test, names = encode_labels(np.array([1, 2, 3, 1]), np.array([3, 1]))
    assert train.tolist() == [0, 1, 2, 0]
    assert test.tolist() == [2, 0]
    assert names == ["1", "2", "3"]


def test_encode_labels_rejects_unseen_test_class():
    with pytest.raises(ValueError):
        encode_labels(np.array([0, 1]), np.array([2]))


def test_npz_loader_rgb_and_grayscale(tmp_path):
    images = np.zeros((10, 32, 32), dtype=np.uint8)
    labels = np.tile(np.arange(10), 1)
    path = tmp_path / "sample.npz"
    np.savez(path, images=images, labels=labels)
    x, y = load_npz_dataset(path)
    assert x.shape == (10, 32, 32, 3)
    assert x.dtype == np.float32
    assert y.shape == (10,)
    assert x.max() == 0.0


def test_image_for_model_normalizes_and_resizes():
    image = np.full((16, 16), 255, dtype=np.uint8)
    out = image_for_model(image, image_size=32)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert out.max() <= 1.0
