import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import make_loader
from .model import UrduDigitCNN
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Urdu digit CNN.")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/digits_torch/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
    ])
    test_set = datasets.ImageFolder(args.test_dir, transform=transform)
    if test_set.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError("Test class mapping does not match the model checkpoint.")

    model = UrduDigitCNN(checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = make_loader(test_set, args.batch_size, False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    labels = list(range(len(test_set.classes)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=test_set.classes,
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "samples": len(y_true)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
