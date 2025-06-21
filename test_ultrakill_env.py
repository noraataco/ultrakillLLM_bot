import numpy as np
import pytest

# Skip tests if Windows specific dependencies are missing
pytest.importorskip("win32gui")

from ultrakill_env import grab_frame, detect_dashes


def test_grab_frame_shape():
    """grab_frame should return a 360x640 RGB image."""
    for _ in range(3):
        frame = grab_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (360, 640, 3)


def test_detect_dashes_basic():
    frame = np.zeros((360, 640, 3), np.uint8)
    x0 = int(640 * 0.03)
    x1 = int(640 * 0.18)
    y0 = int(360 * 0.84)
    y1 = int(360 * 0.89)
    seg_w = (x1 - x0) // 3
    color = (255, 240, 200)
    import cv2
    for i in range(2):
        cv2.rectangle(
            frame,
            (x0 + i * seg_w + 1, y0 + 1),
            (x0 + (i + 1) * seg_w - 1, y1 - 1),
            color,
            -1,
        )
    assert detect_dashes(frame) == 2
