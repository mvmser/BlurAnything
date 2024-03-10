"""
-------------------------------------------------------------------------------
-                            Blur Anything Backend                            -
-------------------------------------------------------------------------------
This module offers a FastAPI backend for BlurAnything, emphasizing object
detection. It sets up endpoints for image uploads and utilizes YOLOv8 for
detecting objects. These objects can be blurred in the frontend. It mainly
handles image uploads, processes them for detection, and returns results.

Usage:
    Run this script with Uvicorn to start the FastAPI server. Ensure the
    YOLOv8 model files are placed as expected by the detect_objects function.
"""

# backend/app.py

import os
from typing import Tuple

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .detect import detect_objects

origins = ["http://localhost:8501"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["X-Requested-With", "Content-Type"],
)


# -------------------
# Endpoints
# -------------------
@app.post("/detect/")
async def detect(file: UploadFile = File(...), type_model: str = Form(...)):
    """
    Processes an uploaded image file to detect objects using a machine learning model.

    This endpoint accepts an image file, validates its type, reads the image data,
    and uses a machine learning model to identify objects within the image. It returns
    a list of detected objects with their names, confidence scores, and bounding box coordinates.
    If no file is uploaded, or if the file type is not supported, it returns an error message.

    Args:
        file (UploadFile): The image file uploaded by the user. The file must be
                           in .jpg, .jpeg, or .png format.

    Returns:
        JSONResponse: A JSON response that either contains the detection results or
                      an error message. The detection results include the name,
                      confidence score, and bounding box (x_min, y_min, x_max, y_max)
                      for each detected object. If an error occurs, it provides a
                      relevant error message.

    Example of a successful response:
    [
        {
            "object": "person",
            "confidence": 0.98,
            "x_min": 34,
            "y_min": 63,
            "x_max": 200,
            "y_max": 360
        },
        ...
    ]

    Example of an error response:
    {"error": "Invalid file type"}
    """
    if not file:
        return create_response({"error": "No image file provided"}, 400)

    valid, message = validate_file(file)
    if not valid:
        return create_response({"error": message}, 400)

    image_data = await file.read()
    results = await detect_objects(image_data=image_data, type_model=type_model)

    if results:
        return create_response(results, 200)
    return create_response({"message": "No objects detected"}, 200)


# -------------------
# Utility Functions
# -------------------
def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """
    Validate the uploaded file type.

    Checks if the uploaded file is of an allowed type (.jpg, .jpeg, .png)
    and has a filename. Returns a tuple indicating whether the file is valid
    and an error message if it is not.

    Args:
        file (UploadFile): The file uploaded by the user.

    Returns:
        A tuple containing a boolean indicating whether the file is valid,
        and a string containing an error message if the file is not valid.
    """
    if file.filename is None:
        return False, "No file name provided."
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return False, "Invalid file type. Allowed types: jpg, jpeg, png."
    return True, ""


def create_response(content: dict, status_code: int) -> JSONResponse:
    """
    Create a JSONResponse for given content and status code.

    Simplifies creating a FastAPI JSONResponse by encapsulating the content and
    status code into a single function call.

    Args:
        content (dict): The content to return in the response.
        status_code (int): The HTTP status code for the response.

    Returns:
        JSONResponse: A FastAPI JSONResponse object with the specified content and status code.
    """
    return JSONResponse(content=content, status_code=status_code)


# -------------------
# API
# -------------------
def start_server():
    """
    Start the Uvicorn server for the FastAPI application.

    Reads the host and port from the environment variables (or defaults) and
    starts the Uvicorn server with the FastAPI application. This function is
    the entry point for running the server.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    start_server()
