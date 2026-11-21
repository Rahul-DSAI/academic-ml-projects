import argparse
from pathlib import Path

import torch
from torch import nn, optim

from .data import build_datasets, make_loader
from .model import UrduDigitCNN


def evaluate_loss_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the Urdu digit CNN.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/digits_torch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size != 28:
        raise ValueError("epochs/batch-size must be positive; current CNN expects image-size 28.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_full, train_set, val_set, test_set = build_datasets(
        args.train_dir, args.test_dir, args.image_size, args.validation_split, args.seed
    )
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    model = UrduDigitCNN(len(train_full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    best_val_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss, samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        val_loss, val_accuracy = evaluate_loss_accuracy(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={running_loss / max(samples, 1):.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": train_full.classes,
                "class_to_idx": train_full.class_to_idx,
                "image_size": args.image_size,
                "num_classes": len(train_full.classes),
                "best_validation_accuracy": best_val_accuracy,
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_accuracy = evaluate_loss_accuracy(model, test_loader, criterion, device)
    print(
        f"Test loss={test_loss:.4f} test_accuracy={test_accuracy:.4f} "
        f"device={device} checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
