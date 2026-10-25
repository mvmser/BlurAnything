"""
-------------------------------------------------------------------------------
-                  Blur Anything Backend App Unit tests                       -
-------------------------------------------------------------------------------
"""

# tests/test_app.py

import io
import json
from unittest.mock import patch

import pytest
from fastapi.datastructures import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app, create_response, parse_boxes, parse_classes, validate_file

client = TestClient(app)


@pytest.fixture
def mock_detect(detections):
    with patch("backend.app.detect_objects") as mock:
        mock.return_value = {
            "detected_objects": detections,
            "speed": {"inference": 1.0},
            "width": 100,
            "height": 80,
        }
        yield mock


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("image.jpg", (True, "")),
        ("IMAGE.PNG", (True, "")),
        ("image.jpeg", (True, "")),
        ("image.webp", (True, "")),
        ("image.txt", (False, "Invalid file type. Allowed types: jpg, jpeg, png, webp.")),
        (None, (False, "No file name provided.")),
    ],
)
def test_validate_file(filename, expected):
    file = UploadFile(filename=filename, file=io.BytesIO(b"dummy"))
    assert validate_file(file) == expected


def test_create_response():
    response = create_response({"message": "test"}, 200)
    assert response.status_code == 200
    assert response.body.decode("utf-8") == '{"message":"test"}'


def test_parse_helpers():
    assert parse_classes(" person, car ,,") == ["person", "car"]
    assert parse_boxes('[{"x_min":1,"y_min":2,"x_max":3,"y_max":4,"extra":0}]') == [
        {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0}
    ]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_detect_endpoint_valid(mock_detect, sample_jpeg):
    response = client.post(
        "/detect",
        files={"file": ("photo.jpg", sample_jpeg, "image/jpeg")},
        data={"type_model": "fast", "confidence": "0.4", "classes": "person"},
    )
    assert response.status_code == 200
    assert response.json()["detected_objects"][0]["object"] == "person"
    args = mock_detect.call_args.args
    assert args[1:] == ("fast", 0.4, ["person"])


def test_detect_trailing_slash_still_works(mock_detect, sample_jpeg):
    response = client.post("/detect/", files={"file": ("photo.jpg", sample_jpeg)})
    assert response.status_code == 200


def test_detect_endpoint_invalid_file_type(mock_detect):
    response = client.post("/detect", files={"file": ("test.txt", b"fake", "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["error"]
    mock_detect.assert_not_called()


def test_detect_endpoint_corrupt_image(mock_detect):
    response = client.post("/detect", files={"file": ("photo.jpg", b"not an image")})
    assert response.status_code == 400
    assert response.json()["error"] == "The file is not a valid image."


@pytest.mark.parametrize(
    "data,error",
    [
        ({"type_model": "turbo"}, "Invalid model type"),
        ({"confidence": "2"}, "Confidence"),
    ],
)
def test_detect_endpoint_bad_params(mock_detect, sample_jpeg, data, error):
    response = client.post("/detect", files={"file": ("p.jpg", sample_jpeg)}, data=data)
    assert response.status_code == 400
    assert error in response.json()["error"]


def test_blur_endpoint_auto(mock_detect, sample_jpeg):
    response = client.post(
        "/blur",
        files={"file": ("photo.jpg", sample_jpeg)},
        data={"classes": "person", "method": "solid"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-detections"] == "2"
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (100, 80)
    assert max(image.getpixel((10, 10))) < 30


def test_blur_endpoint_with_boxes_skips_detection(mock_detect, sample_image):
    png = io.BytesIO()
    sample_image.save(png, format="PNG")
    boxes = json.dumps([{"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}])
    response = client.post(
        "/blur",
        files={"file": ("photo.png", png.getvalue())},
        data={"boxes": boxes, "method": "solid", "padding": "0"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.getpixel((5, 5)) == (0, 0, 0)
    assert image.getpixel((20, 20)) == (255, 0, 0)
    mock_detect.assert_not_called()


@pytest.mark.parametrize(
    "data",
    [{"method": "swirl"}, {"strength": "0"}, {"padding": "3"}, {"boxes": "nope"}],
)
def test_blur_endpoint_bad_params(mock_detect, sample_jpeg, data):
    response = client.post("/blur", files={"file": ("p.jpg", sample_jpeg)}, data=data)
    assert response.status_code == 400
    assert "error" in response.json()
