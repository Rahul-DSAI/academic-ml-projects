"""Run real-time frontal-face detection using a webcam."""

from __future__ import annotations

import argparse

import cv2

from .detector import HaarFaceEyeDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time webcam face detection.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-eyes", action="store_true", help="Skip eye detection")
    args = parser.parse_args()

    detector = HaarFaceEyeDetector()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(
            f"Could not open camera {args.camera}. Check camera permissions or try another index."
        )

    print("Webcam is running. Press Q in the video window to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Unable to read a frame from the webcam.")
                break

            annotated, _ = detector.detect(frame, detect_eyes=not args.no_eyes)
            cv2.imshow("OpenCV Face Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
