"""Run Haar-cascade face/eye detection on a still image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detector import HaarFaceEyeDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect faces and eyes in an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--output",
        default="outputs/face-detection-result.jpg",
        help="Path for the annotated image (default: outputs/face-detection-result.jpg)",
    )
    parser.add_argument("--no-eyes", action="store_true", help="Skip eye detection")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, nargs=2, default=(30, 30), metavar=("WIDTH", "HEIGHT"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detector = HaarFaceEyeDetector(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=tuple(args.min_size),
    )
    annotated, metadata = detector.detect(image, detect_eyes=not args.no_eyes)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"Could not write output image: {output_path}")

    print(json.dumps(metadata, indent=2))
    print(f"\nAnnotated image saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
