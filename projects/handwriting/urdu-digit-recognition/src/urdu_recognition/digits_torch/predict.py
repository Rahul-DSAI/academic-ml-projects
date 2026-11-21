import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import UrduDigitCNN


def main():
    parser = argparse.ArgumentParser(description="Predict one Urdu digit image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
    ])
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = UrduDigitCNN(checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    print(json.dumps({
        "predicted_class": checkpoint["class_names"][index],
        "confidence": float(probabilities[index].item()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
