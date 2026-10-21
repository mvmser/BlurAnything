"""Shared fixtures."""

# tests/conftest.py

import io

import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """A 100x80 image: left half red, right half blue."""
    image = Image.new("RGB", (100, 80), "red")
    image.paste(Image.new("RGB", (50, 80), "blue"), (50, 0))
    return image


@pytest.fixture
def sample_jpeg(sample_image) -> bytes:
    """The sample image encoded as JPEG."""
    output = io.BytesIO()
    sample_image.save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def detections() -> list:
    """Two fake detections."""
    return [
        {"object": "person", "confidence": 0.9, "x_min": 0, "y_min": 0, "x_max": 40, "y_max": 40},
        {"object": "car", "confidence": 0.6, "x_min": 60, "y_min": 40, "x_max": 100, "y_max": 80},
    ]
