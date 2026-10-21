"""
-------------------------------------------------------------------------------
-                    Blur Anything detect Unit tests                          -
-------------------------------------------------------------------------------
"""

# tests/test_detect.py

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from backend import detect


def fake_results():
    """Mimic the Ultralytics Results object for two detections."""
    boxes = SimpleNamespace(
        xyxy=torch.tensor([[1.0, 2.0, 30.0, 40.0], [5.0, 5.0, 10.0, 10.0]]),
        cls=torch.tensor([2.0, 0.0]),
        conf=torch.tensor([0.5, 0.9]),
    )
    return [SimpleNamespace(boxes=boxes, names={0: "person", 2: "car"}, speed={"inference": 1.0})]


def test_extract_detection_results_sorted_by_confidence():
    objects = detect.extract_detection_results(fake_results())
    assert [o["object"] for o in objects] == ["person", "car"]
    assert objects[1] == {
        "object": "car", "confidence": 0.5,
        "x_min": 1.0, "y_min": 2.0, "x_max": 30.0, "y_max": 40.0,
    }


def test_extract_detection_results_without_boxes():
    results = [SimpleNamespace(boxes=None, names={})]
    assert detect.extract_detection_results(results) == []


@patch("backend.detect.load_model")
def test_detect_objects_filters_classes(mock_load):
    model = MagicMock()
    model.predict.return_value = fake_results()
    mock_load.return_value = model

    result = detect.detect_objects(Image.new("RGBA", (64, 48)), "fast", 0.3, ["Car"])

    assert [o["object"] for o in result["detected_objects"]] == ["car"]
    assert (result["width"], result["height"]) == (64, 48)
    image = model.predict.call_args.args[0]
    assert image.mode == "RGB"
    assert model.predict.call_args.kwargs["conf"] == 0.3


def test_load_model_invalid_type():
    with pytest.raises(ValueError):
        detect.load_model("turbo")


@pytest.mark.slow
def test_real_model_detects_a_person():
    """End to end with real YOLOv8n weights (downloads them on first run)."""
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    path = os.path.join("models", "bus.jpg")
    if not os.path.exists(path):
        import urllib.request  # pylint: disable=import-outside-toplevel

        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(url, path)

    result = detect.detect_objects(Image.open(path), "fast", 0.4)
    names = [o["object"] for o in result["detected_objects"]]
    assert "person" in names and "bus" in names
