"""
-------------------------------------------------------------------------------
-               Blur Anything Streamlit Application Unit tests                -
-------------------------------------------------------------------------------
"""

# tests/test_streamlit_app.py

from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

from frontend import streamlit_app
from frontend.streamlit_app import image_to_bytes, select_regions, send_image_to_backend


def test_select_regions(detections):
    assert select_regions(detections, ["person"], []) == [detections[0]]
    assert select_regions(detections, ["person"], [0, 1]) == detections
    assert select_regions(detections, [], []) == []


def test_image_to_bytes(sample_image):
    assert image_to_bytes(sample_image, "PNG").startswith(b"\x89PNG")


@patch("frontend.streamlit_app.st")
@patch("frontend.streamlit_app.requests.post")
def test_send_image_to_backend(mock_post, mock_st):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"detected_objects": []})
    with patch.object(streamlit_app, "API_URL", "http://api:8000"):
        assert send_image_to_backend(b"img", "a.jpg", "fast", 0.3) == {"detected_objects": []}
    assert mock_post.call_args.args[0] == "http://api:8000/detect"
    assert mock_post.call_args.kwargs["data"] == {"type_model": "fast", "confidence": 0.3}

    mock_post.return_value = MagicMock(status_code=400, json=lambda: {"error": "boom"})
    assert send_image_to_backend(b"img", "a.jpg", "fast", 0.3) is None
    mock_st.error.assert_called_with("boom")

    mock_post.side_effect = requests.ConnectionError("down")
    assert send_image_to_backend(b"img", "a.jpg", "fast", 0.3) is None


def test_app_renders_without_upload():
    app = AppTest.from_file("../frontend/streamlit_app.py").run()
    assert not app.exception
    assert app.title[0].value == "BlurAnything"
    assert "Upload an image" in app.info[0].value
