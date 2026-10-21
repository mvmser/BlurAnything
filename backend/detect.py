"""
-------------------------------------------------------------------------------
-                       Blur Anything detect (AI model)                       -
-------------------------------------------------------------------------------

This module provides functionality to detect objects in images using YOLOv8.
Models are loaded once and cached. Weights live in the models/ folder and are
downloaded there by Ultralytics on first use.
"""

# backend/detect.py

import logging
import os
from functools import lru_cache
from typing import Iterable, List, Optional

from PIL import Image

logger = logging.getLogger("uvicorn")

MODELS_DIR = os.getenv("BLURANYTHING_MODELS_DIR", "models")
MODEL_WEIGHTS = {"fast": "yolov8n.pt", "accurate": "yolov8x.pt"}
DEFAULT_MODEL = os.getenv("BLURANYTHING_DEFAULT_MODEL", "fast")


@lru_cache(maxsize=len(MODEL_WEIGHTS))
def load_model(type_model: str):
    """
    Load (once) the YOLO model for the given model type.

    Args:
        type_model (str): Specifies the model type to use, either "fast" or "accurate".

    Returns:
        YOLO: An instance of the YOLO model loaded with the specified weights.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    if type_model not in MODEL_WEIGHTS:
        raise ValueError(
            f"Invalid model type. Available options: {list(MODEL_WEIGHTS)}."
        )

    # Imported here so the API and the UI can start (and be tested) without
    # paying the torch import cost until a model is really needed.
    from ultralytics import YOLO  # type: ignore # pylint: disable=import-outside-toplevel

    model_path = os.path.join(MODELS_DIR, MODEL_WEIGHTS[type_model])
    logger.info("Loading YOLO model %s from %s", type_model, model_path)
    return YOLO(model_path)


def get_class_names(type_model: str = DEFAULT_MODEL) -> List[str]:
    """
    List the object classes the model can detect.

    Args:
        type_model (str): The model type.

    Returns:
        list: Class names, ordered by class id.
    """
    names = load_model(type_model).names
    return [names[i] for i in sorted(names)]


def detect_objects(
    image: Image.Image,
    type_model: str = DEFAULT_MODEL,
    confidence: float = 0.25,
    classes: Optional[Iterable[str]] = None,
) -> dict:
    """
    Detect objects in an image using the specified YOLOv8 model.

    Args:
        image (Image.Image): The image to analyse.
        type_model (str): Specifies the model type to use, either "fast" or "accurate".
        confidence (float): Minimum confidence score to keep a detection.
        classes: Optional class names to keep; all classes are kept when empty.

    Returns:
        dict: {"detected_objects": [...], "speed": {...}, "width": int, "height": int}
            Each detected object has the keys "object", "confidence",
            "x_min", "y_min", "x_max", "y_max" (pixels).
    """
    model = load_model(type_model)
    image = image.convert("RGB")
    results = model.predict(image, conf=confidence, verbose=False)
    detected_objects = extract_detection_results(results)

    wanted = {c.strip().lower() for c in classes or [] if c.strip()}
    if wanted:
        detected_objects = [
            obj for obj in detected_objects if obj["object"].lower() in wanted
        ]

    return {
        "detected_objects": detected_objects,
        "speed": results[0].speed,
        "width": image.width,
        "height": image.height,
    }


def extract_detection_results(results) -> list:
    """
    Extract and format the detection results from the model's output.

    Args:
        results: The raw detection results from the YOLO model.

    Returns:
        list: One dict per detection, sorted by decreasing confidence.
    """
    result = results[0]
    detected_objects = []

    if result.boxes is not None:
        for xyxy, cls, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
        ):
            detected_objects.append(
                {
                    "object": result.names[int(cls)],
                    "confidence": round(float(conf), 4),
                    "x_min": round(xyxy[0], 1),
                    "y_min": round(xyxy[1], 1),
                    "x_max": round(xyxy[2], 1),
                    "y_max": round(xyxy[3], 1),
                }
            )

    detected_objects.sort(key=lambda obj: obj["confidence"], reverse=True)
    logger.info("YOLO detected %d objects", len(detected_objects))
    return detected_objects
