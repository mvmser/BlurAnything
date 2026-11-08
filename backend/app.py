"""
-------------------------------------------------------------------------------
-                            Blur Anything Backend                            -
-------------------------------------------------------------------------------
This module offers a FastAPI backend for BlurAnything. It detects objects with
YOLOv8 and anonymises them (gaussian blur, pixelation or solid mask).

Endpoints:
    GET  /health    Liveness probe.
    GET  /classes   Object classes the model can detect.
    POST /detect    Detect objects, returns JSON.
    POST /blur      Detect and blur objects, returns the anonymised image.

Usage:
    uvicorn backend.app:app --reload
"""

# backend/app.py

import io
import json
import os
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from .blur import METHODS, blur_regions
from .detect import DEFAULT_MODEL, MODEL_WEIGHTS, detect_objects, get_class_names

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_UPLOAD_MB = int(os.getenv("BLURANYTHING_MAX_UPLOAD_MB", "15"))
origins = os.getenv("BLURANYTHING_CORS_ORIGINS", "http://localhost:8501").split(",")

app = FastAPI(
    title="BlurAnything API",
    description="Detect and blur objects in images with YOLOv8.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Requested-With", "Content-Type"],
    expose_headers=["X-Detections"],
)


# -------------------
# Endpoints
# -------------------
@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/classes")
async def classes(type_model: str = DEFAULT_MODEL):
    """List the object classes the selected model can detect."""
    check_model(type_model)
    return await run_in_threadpool(get_class_names, type_model)


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    type_model: str = Form(DEFAULT_MODEL),
    confidence: float = Form(0.25),
    classes: str = Form(""),
):
    """
    Detect objects in an uploaded image.

    Args:
        file (UploadFile): A .jpg, .jpeg, .png or .webp image.
        type_model (str): "fast" (YOLOv8n) or "accurate" (YOLOv8x).
        confidence (float): Minimum confidence score, between 0 and 1.
        classes (str): Optional comma separated class names to keep.

    Returns:
        JSONResponse: {"detected_objects": [...], "speed": {...}, "width", "height"}
            Each object has "object", "confidence", "x_min", "y_min", "x_max", "y_max".

    Example of an error response:
        {"error": "Invalid file type. Allowed types: jpg, jpeg, png, webp."}
    """
    image = await read_image(file)
    check_model(type_model)
    check_confidence(confidence)

    results = await run_in_threadpool(
        detect_objects, image, type_model, confidence, parse_classes(classes)
    )
    return create_response(results, 200)


@app.post("/blur")
async def blur(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    file: UploadFile = File(...),
    type_model: str = Form(DEFAULT_MODEL),
    confidence: float = Form(0.25),
    classes: str = Form("person"),
    method: str = Form("gaussian"),
    strength: int = Form(15),
    padding: float = Form(0.05),
    boxes: Optional[str] = Form(None),
):
    """
    Blur objects in an uploaded image and return the result as an image.

    By default the image is analysed and every object of the given classes is
    blurred. Pass `boxes` (a JSON list of {x_min, y_min, x_max, y_max}) to blur
    exact regions instead, e.g. the ones picked by a user after /detect.

    Returns:
        Response: The anonymised image (same format as the upload, PNG for PNG,
            JPEG otherwise). The X-Detections header holds the number of
            blurred regions.
    """
    image = await read_image(file)
    if method not in METHODS:
        raise HTTPException(400, f"Invalid blur method. Available options: {list(METHODS)}.")
    if not 1 <= strength <= 100:
        raise HTTPException(400, "Strength must be between 1 and 100.")
    if not 0 <= padding <= 1:
        raise HTTPException(400, "Padding must be between 0 and 1.")

    if boxes is not None:
        regions = parse_boxes(boxes)
    else:
        check_model(type_model)
        check_confidence(confidence)
        results = await run_in_threadpool(
            detect_objects, image, type_model, confidence, parse_classes(classes)
        )
        regions = results["detected_objects"]

    blurred = await run_in_threadpool(
        blur_regions, image, regions, method, strength, padding
    )

    output = io.BytesIO()
    is_png = (file.filename or "").lower().endswith(".png")
    blurred.save(output, format="PNG" if is_png else "JPEG", quality=95)
    return Response(
        content=output.getvalue(),
        media_type="image/png" if is_png else "image/jpeg",
        headers={"X-Detections": str(len(regions))},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    """Return errors as {"error": message}."""
    return create_response({"error": exc.detail}, exc.status_code)


# -------------------
# Utility Functions
# -------------------
def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """
    Validate the uploaded file name.

    Args:
        file (UploadFile): The file uploaded by the user.

    Returns:
        A tuple containing a boolean indicating whether the file is valid,
        and a string containing an error message if the file is not valid.
    """
    if not file.filename:
        return False, "No file name provided."
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        return False, "Invalid file type. Allowed types: jpg, jpeg, png, webp."
    return True, ""


async def read_image(file: UploadFile) -> Image.Image:
    """
    Validate and decode an uploaded image.

    Raises:
        HTTPException: 400 for a bad name or content, 413 when too large.
    """
    valid, message = validate_file(file)
    if not valid:
        raise HTTPException(400, message)

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large (max {MAX_UPLOAD_MB} MB).")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "The file is not a valid image.") from exc
    return image


def check_model(type_model: str) -> None:
    """Raise a 400 error for an unknown model type."""
    if type_model not in MODEL_WEIGHTS:
        raise HTTPException(
            400, f"Invalid model type. Available options: {list(MODEL_WEIGHTS)}."
        )


def check_confidence(confidence: float) -> None:
    """Raise a 400 error for a confidence outside [0, 1]."""
    if not 0 <= confidence <= 1:
        raise HTTPException(400, "Confidence must be between 0 and 1.")


def parse_classes(classes: str) -> List[str]:
    """Split a comma separated list of class names."""
    return [c.strip() for c in classes.split(",") if c.strip()]


def parse_boxes(boxes: str) -> List[dict]:
    """
    Parse a JSON list of boxes.

    Raises:
        HTTPException: 400 when the JSON or a box is malformed.
    """
    keys = ("x_min", "y_min", "x_max", "y_max")
    try:
        parsed = json.loads(boxes)
        return [{k: float(box[k]) for k in keys} for box in parsed]
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            400, "Boxes must be a JSON list of {x_min, y_min, x_max, y_max}."
        ) from exc


def create_response(content, status_code: int) -> JSONResponse:
    """
    Create a JSONResponse for given content and status code.

    Args:
        content: The content to return in the response.
        status_code (int): The HTTP status code for the response.

    Returns:
        JSONResponse: A FastAPI JSONResponse object.
    """
    return JSONResponse(content=content, status_code=status_code)


# -------------------
# API
# -------------------
def start_server():
    """
    Start the Uvicorn server for the FastAPI application.

    Reads the host and port from the environment variables (or defaults).
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app:app", host=host, port=port)


if __name__ == "__main__":
    start_server()
