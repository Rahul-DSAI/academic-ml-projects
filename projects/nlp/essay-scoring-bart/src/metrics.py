"""Metrics for ordinal short-answer score classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

LABELS = [0, 1, 2, 3]


def calculate_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    """Compute classification and ordinal agreement metrics.

    QWK is included because the task is ordered scoring: predicting 2 for a
    gold score of 3 is less severe than predicting 0 for a gold score of 3.
    """
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.size == 0:
        raise ValueError("Cannot calculate metrics on an empty set.")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not set(np.unique(true)).issubset(set(LABELS)):
        raise ValueError("y_true contains labels outside 0, 1, 2, 3.")
    if not set(np.unique(pred)).issubset(set(LABELS)):
        raise ValueError("y_pred contains labels outside 0, 1, 2, 3.")

    qwk = cohen_kappa_score(true, pred, labels=LABELS, weights="quadratic")
    kappa = cohen_kappa_score(true, pred, labels=LABELS)
    return {
        "n_samples": int(true.size),
        "accuracy": float(accuracy_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "mse": float(mean_squared_error(true, pred)),
        "quadratic_weighted_kappa": float(qwk) if np.isfinite(qwk) else None,
        "cohen_kappa": float(kappa) if np.isfinite(kappa) else None,
        "confusion_matrix": confusion_matrix(true, pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            true,
            pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
