from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_datasets(train_dir, test_dir, image_size=28, validation_split=0.1, seed=42):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_full = datasets.ImageFolder(root=str(train_dir), transform=transform)
    test_set = datasets.ImageFolder(root=str(test_dir), transform=transform)

    if train_full.class_to_idx != test_set.class_to_idx:
        raise ValueError(
            "Training and testing folder-to-index mappings differ. "
            f"Train: {train_full.class_to_idx}; test: {test_set.class_to_idx}"
        )
    if len(train_full.classes) < 2:
        raise ValueError("Expected at least two digit class folders.")

    indices = list(range(len(train_full)))
    labels = train_full.targets
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
    )
    return train_full, Subset(train_full, train_indices), Subset(train_full, val_indices), test_set


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=0):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False
    )
