import numpy as np
import pytest
import cv2

# Skip tests if Windows specific dependencies are missing
pytest.importorskip("win32gui")

from ultrakill_env import grab_frame, read_health


def test_grab_frame_shape():
    """grab_frame should return a 360x640 RGB image."""
    for _ in range(3):
        frame = grab_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (360, 640, 3)


def test_read_health_basic():
    frame = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(frame, "75", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    val = read_health(frame)
    assert val == 75
