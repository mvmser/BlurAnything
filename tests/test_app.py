"""
-------------------------------------------------------------------------------
-                  Blur Anything Backend App Unit tests                       -
-------------------------------------------------------------------------------
"""

# tests/test_app.py

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.datastructures import UploadFile
from fastapi.testclient import TestClient
from backend.app import app, validate_file, create_response

client = TestClient(app)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("image.jpg", (True, "")),
        ("image.png", (True, "")),
        ("image.jpeg", (True, "")),
        ("image.txt", (False, "Invalid file type. Allowed types: jpg, jpeg, png.")),
        (None, (False, "No file name provided.")),
    ],
)
def test_validate_file(filename, expected):
    """
    Test the file validation function with various filenames.

    This test checks whether the `validate_file` function correctly identifies
    valid and invalid filenames based on their extensions or lack thereof.

    Args:
        filename (str | None): The name of the file to test,
            or None to simulate missing filenames.
        expected (tuple): A tuple containing a boolean indicating
            if the file is valid, and a message.
    """
    content = b"dummy content"

    if filename is not None:
        file = UploadFile(filename=filename, file=BytesIO(content))
    else:
        # If filename is None, simulate missing file by not setting 'file' parameter
        file = UploadFile(file=BytesIO(content))
    assert validate_file(file) == expected


def test_create_response():
    """
    Test the creation of JSONResponse objects.

    This test verifies that `create_response` correctly sets the content and status code
    of the JSONResponse it returns.
    """
    content = {"message": "test"}
    status_code = 200
    response = create_response(content, status_code)
    assert response.status_code == status_code
    assert response.body.decode("utf-8") == '{"message":"test"}'


@patch("backend.app.detect_objects")
def test_detect_endpoint_valid(mock_detect_objects):
    """
    Test the /detect/ endpoint with a valid image file.

    This test mocks the `detect_objects` function to return a predefined response,
    then it sends a valid image file to the /detect/ endpoint and checks if the
    response is as expected.
    """
    mock_detect_objects.return_value = [{"object": "person", "confidence": 0.98}]
    with open("../static/images/test_image.jpg", "rb") as file:
        response = client.post("/detect/", files={"file": file})
    assert response.status_code == 200
    assert "person" in response.json()[0]["object"]


@patch("backend.app.detect_objects")
def test_detect_endpoint_invalid_file_type(mock_detect_objects):
    """
    Test the /detect/ endpoint with an invalid file type.

    This test mocks the `detect_objects` function to not be called, then it attempts
    to upload a file with an unsupported file extension to the /detect/ endpoint
    and checks if the appropriate error message is returned.
    """
    # Attempt to upload an unsupported file type
    response = client.post(
        "/detect/", files={"file": ("test.txt", "fake content", "text/plain")}
    )

    # Assert that the response indicates an invalid file type
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["error"]

    mock_detect_objects.assert_not_called()
