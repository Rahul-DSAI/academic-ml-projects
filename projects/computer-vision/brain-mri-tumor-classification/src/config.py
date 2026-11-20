"""Central configuration for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Expected layout: data/Training/<class_name>/* and data/Testing/<class_name>/*
    dataset_dir: Path = Path("data")
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 60
    validation_split: float = 0.10
    random_seed: int = 42
    output_dir: Path = Path("artifacts")

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "Training"

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "Testing"


DEFAULT_CONFIG = Config()
