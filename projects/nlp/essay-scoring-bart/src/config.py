"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Project configuration loaded from a YAML file."""

    data_path: str = "data/raw/train.tsv"
    output_dir: str = "artifacts"
    model_name_or_path: str = "facebook/bart-large-cnn"
    num_labels: int = 4
    max_length: int = 512
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    test_size: float = 0.1
    validation_size: float = 0.1
    random_seed: int = 42
    num_workers: int = 0
    use_amp: bool = True
    early_stopping_patience: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            values: dict[str, Any] = yaml.safe_load(file) or {}

        known_fields = cls.__dataclass_fields__.keys()
        unknown = sorted(set(values) - set(known_fields))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")

        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_labels != 4:
            raise ValueError("This project expects four Score1 classes: 0, 1, 2 and 3.")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8.")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.test_size < 1 or not 0 < self.validation_size < 1:
            raise ValueError("test_size and validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
