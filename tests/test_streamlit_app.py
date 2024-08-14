"""
-------------------------------------------------------------------------------
-               Blur Anything Streamlit Application Unit tests                -
-------------------------------------------------------------------------------
"""

# tests/test_streamlit_app.py

# pylint: disable=W0621:redefined-outer-name

import io
from unittest.mock import patch, MagicMock
import pytest
from frontend.streamlit_app import (
    main,
    upload_image_ui,
    display_uploaded_image,
    send_image_to_backend,
    display_detection_results,
)


@pytest.fixture
def mock_st():
    """
    Fixture to mock the Streamlit module.

    Yields:
        MagicMock: A MagicMock object for mocking the Streamlit module.
    """
    with patch("frontend.streamlit_app.st") as mock_streamlit:
        yield mock_streamlit


@pytest.fixture
def mock_requests():
    """
    Fixture to mock the requests module.

    Yields:
        MagicMock: A MagicMock object for mocking the requests module.
    """
    with patch("frontend.streamlit_app.requests") as mock_req:
        yield mock_req


def test_main_with_uploaded_image(mock_st: MagicMock, mock_requests: MagicMock):
    """
    Test the main function when an image is uploaded.

    This test mocks the Streamlit file uploader and backend response.
    It verifies that the main function calls the appropriate functions
    and displays the detection results correctly.

    Args:
        mock_st: A MagicMock object for mocking Streamlit.
        mock_requests: A MagicMock object for mocking the requests module.
    """
    mock_st.file_uploader.return_value = MagicMock(spec=io.BytesIO)
    mock_st.file_uploader.return_value.read.return_value = b"dummy image data"
    mock_st.file_uploader.return_value.type = "image/jpeg"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"object": "person", "confidence": 0.98}]
    mock_requests.post.return_value = mock_response

    main()

    mock_st.image.assert_called_once()
    mock_requests.post.assert_called_once()

    # Assert that the arguments passed to requests.post match the expected values
    mock_requests.post.assert_called_once_with(
        url="http://127.0.0.1:8000/detect/",
        files={"file": ("image.jpg", b"dummy image data", "image/jpeg")},
        timeout=10,
    )

    mock_st.write.assert_called_once_with(
        "Detection Results:", [{"object": "person", "confidence": 0.98}]
    )


def test_upload_image_ui(mock_st):
    """
    Test the upload_image_ui function.

    This test verifies that the function returns the uploaded file object
    if an image is uploaded, and None otherwise.

    Args:
        mock_st: A MagicMock object for mocking Streamlit.
    """
    mock_uploaded_image = MagicMock(spec=io.BytesIO)
    mock_st.file_uploader.return_value = mock_uploaded_image

    assert upload_image_ui() == mock_uploaded_image

    mock_st.file_uploader.return_value = None
    assert upload_image_ui() is None


def test_display_uploaded_image(mock_st):
    """
    Test the display_uploaded_image function.

    This test verifies that the function displays the uploaded image
    if it's not None.

    Args:
        mock_st: A MagicMock object for mocking Streamlit.
    """
    mock_uploaded_image = MagicMock(spec=io.BytesIO)

    display_uploaded_image(mock_uploaded_image)

    mock_st.image.assert_called_once_with(
        mock_uploaded_image, caption="Uploaded Image", use_column_width=True
    )


def test_send_image_to_backend(mock_requests):
    """
    Test the send_image_to_backend function.

    This test verifies that the function sends the uploaded image
    to the backend API correctly and returns the response.

    Args:
        mock_requests: A MagicMock object for mocking the requests module.
    """
    mock_uploaded_image = MagicMock(spec=io.BytesIO)
    mock_uploaded_image.read.return_value = b"dummy image data"
    mock_uploaded_image.type = "image/jpeg"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.post.return_value = mock_response

    response = send_image_to_backend(mock_uploaded_image)
    assert response == mock_response
    mock_requests.post.assert_called_once()


def test_display_detection_results_success(mock_st):
    """
    Test the display_detection_results function for successful response.

    This test verifies that the function correctly displays the detection
    results when the response status code is 200.

    Args:
        mock_st: A MagicMock object for mocking Streamlit.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"object": "person", "confidence": 0.98}]

    display_detection_results(mock_response)

    mock_st.write.assert_called_once_with(
        "Detection Results:", [{"object": "person", "confidence": 0.98}]
    )


def test_display_detection_results_failure(mock_st):
    """
    Test the display_detection_results function for failed response.

    This test verifies that the function correctly displays the error message
    when the response status code is not 200.

    Args:
        mock_st: A MagicMock object for mocking Streamlit.
    """
    # Mocking response
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Failed to detect objects"}

    display_detection_results(mock_response)

    mock_st.error.assert_called_once_with("Failed to detect objects")
