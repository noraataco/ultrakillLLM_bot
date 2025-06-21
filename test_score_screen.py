import numpy as np
import pytest
pytest.importorskip("win32gui")
from ultrakill_env import is_score_screen

def test_is_score_screen_basic():
    dark = np.full((360, 640, 3), 20, np.uint8)
    bright = np.full((360, 640, 3), 200, np.uint8)
    assert is_score_screen(dark)
    assert not is_score_screen(bright)
