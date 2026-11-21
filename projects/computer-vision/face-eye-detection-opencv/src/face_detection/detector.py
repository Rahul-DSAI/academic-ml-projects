"""Face and eye detection using OpenCV Haar cascades."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class HaarFaceEyeDetector:
    """Detect frontal faces and (optionally) eyes in BGR images.

    The default cascade XML files are bundled with the installed OpenCV
    Python package, so users do not need to download separate XML files.
    """

    def __init__(
        self,
        face_cascade_path: str | None = None,
        eye_cascade_path: str | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")
        if len(min_size) != 2 or min_size[0] <= 0 or min_size[1] <= 0:
            raise ValueError("min_size must contain two positive integers")

        cascade_dir = Path(cv2.data.haarcascades)
        face_path = Path(face_cascade_path) if face_cascade_path else (
            cascade_dir / "haarcascade_frontalface_default.xml"
        )
        eye_path = Path(eye_cascade_path) if eye_cascade_path else (
            cascade_dir / "haarcascade_eye_tree_eyeglasses.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(str(face_path))
        self.eye_cascade = cv2.CascadeClassifier(str(eye_path))
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Could not load face cascade: {face_path}")
        if self.eye_cascade.empty():
            raise FileNotFoundError(f"Could not load eye cascade: {eye_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(
        self,
        image: np.ndarray,
        detect_eyes: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return an annotated copy of ``image`` and detection metadata."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty NumPy image array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a 3-channel BGR image")

        annotated = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        detections: list[dict[str, Any]] = []
        total_eyes = 0
        for index, (x, y, width, height) in enumerate(faces, start=1):
            cv2.rectangle(
                annotated,
                (int(x), int(y)),
                (int(x + width), int(y + height)),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"Face {index}",
                (int(x), max(20, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            eye_boxes: list[list[int]] = []
            if detect_eyes:
                face_roi = gray[y : y + height, x : x + width]
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(10, 10),
                )
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        annotated,
                        (int(x + ex), int(y + ey)),
                        (int(x + ex + ew), int(y + ey + eh)),
                        (0, 255, 0),
                        2,
                    )
                    eye_boxes.append([int(ex), int(ey), int(ew), int(eh)])
                total_eyes += len(eyes)

            detections.append(
                {
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "eyes": eye_boxes,
                }
            )

        summary = f"Faces: {len(faces)}"
        if detect_eyes:
            summary += f" | Eyes: {total_eyes}"
        cv2.putText(
            annotated,
            summary,
            (10, max(25, annotated.shape[0] - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        metadata = {
            "face_count": int(len(faces)),
            "eye_count": int(total_eyes) if detect_eyes else None,
            "detections": detections,
        }
        return annotated, metadata
