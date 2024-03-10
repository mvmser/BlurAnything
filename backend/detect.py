"""
-------------------------------------------------------------------------------
-                       Blur Anything detect (AI model)                       -
-------------------------------------------------------------------------------

This module provides functionality to detect objects in images using YOLOv8.
"""

# backend/detect.py

import logging
import os
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO  # type: ignore

logger = logging.getLogger("uvicorn")

MODEL_PATHS = {"fast": "models/yolov8n.pt", "accurate": "models/yolov8x.pt"}


async def detect_objects(image_data: bytes, type_model: str = "accurate") -> dict:
    """
    Detect objects in an image using the specified YOLOv8 model.

    Args:
        image_data (bytes): The image data in bytes.
        type_model (str): Specifies the model type to use, either "fast" or "accurate".

    Returns:
        dict: A dictionary containing information about the detected objects,
            the detection speed, and the path to the inferred image.
    """
    image = Image.open(BytesIO(image_data))
    model = load_model(type_model)
    results = perform_detection(model=model, image=image)
    detected_objects = extract_detection_results(results)
    inferred_image_path = render_and_save_image(results)

    return {
        "detected_objects": detected_objects,
        "speed": results[0].speed,
        "inferred_image_path": inferred_image_path,
    }


def load_model(type_model: str) -> YOLO:
    """
    Load the YOLO model based on the specified model type.

    Args:
        type_model (str): Specifies the model type to use, either "fast" or "accurate".

    Returns:
        YOLO: An instance of the YOLO model loaded with the specified weights.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    logger.info("Loading YOLO model: %s", type_model)
    if type_model not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model type. Available options: {list(MODEL_PATHS.keys())}."
        )
    model_path = MODEL_PATHS[type_model]
    return YOLO(model_path)


def perform_detection(model: YOLO, image: Image.Image):
    """
    Perform object detection on the provided image using the given model.

    Args:
        model (YOLO): The YOLO model to use for object detection.
        image (Image.Image): The image to detect objects in.

    Returns:
        The result of the detection process.
    """
    image_np = np.array(image)
    image_np = image_np[:, :, ::-1]

    return model(image_np)


def extract_detection_results(results) -> list:
    """
    Extract and format the detection results from the model's output.

    Args:
        results: The raw detection results from the YOLO model.

    Returns:
        dict: A dictionary containing information about the detected objects
            and potentially other metadata.
            "detected_objects":
                Each dictionary contains the keys:
                    - "object": Name of the detected object.
                    - "confidence": Confidence score of the detection.
                    - "x_min": Minimum x-coordinate of the bounding box.
                    - "y_min": Minimum y-coordinate of the bounding box.
                    - "x_max": Maximum x-coordinate of the bounding box.
                    - "y_max": Maximum y-coordinate of the bounding box.
            "speed":
            "inferred_image_pat":
    """
    logger.info("Extracting YOLO Detection Results")
    result = results[0]

    detected_objects = []
    if hasattr(result, "boxes") and len(result.boxes.data):
        for i, box_data in enumerate(result.boxes.data):
            class_id = int(result.boxes.cls[i].item())
            conf = result.boxes.conf[i].item()

            detected_objects.append(
                {
                    "object": result.names[class_id],
                    "confidence": conf,
                    "x_min": box_data[0].item(),
                    "y_min": box_data[1].item(),
                    "x_max": box_data[2].item(),
                    "y_max": box_data[3].item(),
                }
            )

    logger.info("YOLO Detection Results: %s", result)
    logger.info("YOLO Detection Results Boxes: %s ", result.boxes)

    return detected_objects


def render_and_save_image(results: list) -> str:
    """
    Render the image with bounding boxes for detected objects and save it.

    Args:


    Returns:
        str: The path to the saved image with bounding boxes.
    """
    inferred_image_path = os.path.join("static", "inferences", "inferred_image.jpg")
    results[0].save(filename=inferred_image_path)

    return inferred_image_path
