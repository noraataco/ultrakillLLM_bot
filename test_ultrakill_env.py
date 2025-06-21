import numpy as np
import pytest

# Skip tests if Windows specific dependencies are missing
pytest.importorskip("win32gui")

from ultrakill_env import grab_frame


def test_grab_frame_shape():
    """grab_frame should return a 360x640 RGB image."""
    for _ in range(3):
        frame = grab_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (360, 640, 3)
