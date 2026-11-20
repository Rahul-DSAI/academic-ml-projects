import pytest

from src.metrics import calculate_metrics


def test_metrics_perfect_predictions():
    metrics = calculate_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert metrics["accuracy"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["quadratic_weighted_kappa"] == 1.0


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same shape"):
        calculate_metrics([0, 1], [0])
