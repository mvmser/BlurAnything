"""
-------------------------------------------------------------------------------
-                     Blur Anything blur Unit tests                           -
-------------------------------------------------------------------------------
"""

# tests/test_blur.py

import pytest
from PIL import Image, ImageChops

from backend.blur import blur_regions, draw_boxes


def checkerboard(size: int = 64) -> Image.Image:
    """A high-frequency image so blurring visibly changes pixels."""
    image = Image.new("RGB", (size, size), "white")
    for x in range(size):
        for y in range(size):
            if (x + y) % 2:
                image.putpixel((x, y), (0, 0, 0))
    return image


@pytest.mark.parametrize("method", ["gaussian", "pixelate", "solid"])
def test_blur_changes_only_the_box(method):
    image = checkerboard()
    box = {"x_min": 0, "y_min": 0, "x_max": 32, "y_max": 32}
    result = blur_regions(image, [box], method=method, strength=8)

    diff = ImageChops.difference(image, result)
    assert diff.crop((0, 0, 32, 32)).getbbox() is not None
    assert diff.crop((32, 32, 64, 64)).getbbox() is None
    # The source image is left untouched.
    assert image.getpixel((1, 0)) == (0, 0, 0)


def test_solid_is_black():
    result = blur_regions(checkerboard(), [{"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}], "solid")
    assert result.crop((0, 0, 10, 10)).getcolors() == [(100, (0, 0, 0))]


def test_padding_and_clamping(sample_image):
    box = {"x_min": 40, "y_min": 30, "x_max": 60, "y_max": 50}
    result = blur_regions(sample_image, [box], "solid", padding=0.5)
    assert result.getpixel((31, 21)) == (0, 0, 0)
    assert result.getpixel((29, 19)) == (255, 0, 0)

    outside = {"x_min": -50, "y_min": -50, "x_max": 500, "y_max": 500}
    assert blur_regions(sample_image, [outside], "solid").getcolors() == [(8000, (0, 0, 0))]


def test_empty_box_is_ignored(sample_image):
    box = {"x_min": 200, "y_min": 200, "x_max": 300, "y_max": 300}
    assert ImageChops.difference(sample_image, blur_regions(sample_image, [box])).getbbox() is None


def test_rgba_input_returns_rgb():
    image = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
    assert blur_regions(image, []).mode == "RGB"


def test_invalid_method(sample_image):
    with pytest.raises(ValueError):
        blur_regions(sample_image, [], method="swirl")


def test_draw_boxes(sample_image, detections):
    result = draw_boxes(sample_image, detections)
    assert ImageChops.difference(sample_image, result).getbbox() is not None
    assert result.size == sample_image.size
