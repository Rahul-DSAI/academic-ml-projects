import pytest

torch = pytest.importorskip("torch")

from urdu_recognition.digits_torch.model import UrduDigitCNN


def test_digit_model_output_shape():
    model = UrduDigitCNN(num_classes=10)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 28, 28))
    assert tuple(output.shape) == (2, 10)


def test_digit_model_rejects_too_few_classes():
    with pytest.raises(ValueError):
        UrduDigitCNN(num_classes=1)
