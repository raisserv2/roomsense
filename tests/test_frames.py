import numpy as np
import pytest

from roomsense import frames
from tests.conftest import jpeg


def test_decode_and_shapes():
    img = frames.decode_jpeg(jpeg(100, (720, 1280)))
    assert img.shape == (720, 1280, 3)
    x = frames.to_input(img)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_normalisation_matches_torchvision():
    # A white pixel must become (1 - mean) / std per channel, exactly like T.Normalize
    x = frames.to_input(np.full((224, 224, 3), 255, dtype=np.uint8))
    expected = (1 - frames.MEAN) / frames.STD
    assert np.allclose(x[0, :, 0, 0], expected, atol=1e-5)


def test_bgr_to_rgb_order():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue in OpenCV's BGR order
    x = frames.to_input(img)[0]
    assert x[2].mean() > x[0].mean()  # lands in the B channel of the RGB tensor


def test_brightness():
    assert frames.brightness(np.full((50, 80, 3), 77, dtype=np.uint8)) == pytest.approx(77, abs=0.5)


def test_garbage_is_rejected():
    with pytest.raises(ValueError):
        frames.decode_jpeg(b"definitely not a jpeg")


def test_softmax():
    p = frames.softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1) and p.argmax() == 2
