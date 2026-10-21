"""
-------------------------------------------------------------------------------
-                         Blur Anything blur (image ops)                      -
-------------------------------------------------------------------------------

This module anonymises regions of an image. It is pure Pillow, so it is shared
by the FastAPI backend and the Streamlit frontend.
"""

# backend/blur.py

from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFilter

METHODS = ("gaussian", "pixelate", "solid")


def blur_regions(
    image: Image.Image,
    boxes: Iterable[Mapping[str, float]],
    method: str = "gaussian",
    strength: int = 15,
    padding: float = 0.0,
) -> Image.Image:
    """
    Return a copy of the image with every box anonymised.

    Args:
        image (Image.Image): The source image (left untouched).
        boxes: Dicts with "x_min", "y_min", "x_max", "y_max" in pixels.
        method (str): "gaussian", "pixelate" or "solid" (black box).
        strength (int): Blur radius for "gaussian", block size for "pixelate".
        padding (float): Extra margin around each box, as a ratio of its size.

    Returns:
        Image.Image: The anonymised image, in RGB.

    Raises:
        ValueError: If an unknown method is given.
    """
    if method not in METHODS:
        raise ValueError(f"Invalid blur method. Available options: {list(METHODS)}.")

    result = image.convert("RGB")
    strength = max(1, int(strength))

    for obj in boxes:
        box = _clamp_box(obj, result.size, padding)
        if box is None:
            continue

        if method == "solid":
            # paste() excludes the right/bottom edge, like crop() does.
            result.paste((0, 0, 0), box)
            continue

        region = result.crop(box)
        if method == "gaussian":
            region = region.filter(ImageFilter.GaussianBlur(strength))
        else:
            small = (
                max(1, region.width // strength),
                max(1, region.height // strength),
            )
            region = region.resize(small, Image.Resampling.BILINEAR).resize(
                region.size, Image.Resampling.NEAREST
            )
        result.paste(region, box)

    return result


def draw_boxes(image: Image.Image, boxes: Iterable[Mapping]) -> Image.Image:
    """
    Return a copy of the image with labelled bounding boxes drawn on it.

    Args:
        image (Image.Image): The source image (left untouched).
        boxes: Detection dicts with box coordinates, "object" and "confidence".

    Returns:
        Image.Image: The annotated image, in RGB.
    """
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    width = max(2, result.width // 300)

    for i, obj in enumerate(boxes):
        box = _clamp_box(obj, result.size, 0.0)
        if box is None:
            continue
        draw.rectangle(box, outline="red", width=width)
        draw.text(
            (box[0] + width, box[1] + width),
            f"{i}: {obj['object']} {obj['confidence']:.2f}",
            fill="red",
        )
    return result


def _clamp_box(obj: Mapping[str, float], size: tuple, padding: float):
    """
    Pad a box, clamp it to the image and round it to integer pixels.

    Returns None when the box is empty once clamped.
    """
    x_min, y_min = float(obj["x_min"]), float(obj["y_min"])
    x_max, y_max = float(obj["x_max"]), float(obj["y_max"])
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding

    left = max(0, int(x_min - pad_x))
    top = max(0, int(y_min - pad_y))
    right = min(size[0], int(round(x_max + pad_x)))
    bottom = min(size[1], int(round(y_max + pad_y)))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
