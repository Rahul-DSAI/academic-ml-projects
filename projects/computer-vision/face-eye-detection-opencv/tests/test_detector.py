import numpy as np
import pytest

from face_detection.detector import HaarFaceEyeDetector


def test_detector_loads_default_opencv_cascades():
    detector = HaarFaceEyeDetector()
    assert detector.face_cascade is not None
    assert detector.eye_cascade is not None


def test_detect_rejects_empty_image():
    detector = HaarFaceEyeDetector()
    with pytest.raises(ValueError, match="non-empty"):
        detector.detect(None)


def test_detect_rejects_grayscale_image():
    detector = HaarFaceEyeDetector()
    with pytest.raises(ValueError, match="3-channel BGR"):
        detector.detect(np.zeros((64, 64), dtype=np.uint8))


def test_invalid_scale_factor_is_rejected():
    with pytest.raises(ValueError, match="scale_factor"):
        HaarFaceEyeDetector(scale_factor=1.0)
