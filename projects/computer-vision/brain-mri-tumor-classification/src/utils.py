"""Small shared utilities."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seeds(seed: int) -> None:
    """Set Python and NumPy seeds; TensorFlow is seeded by the caller."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
