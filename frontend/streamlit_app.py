"""
-------------------------------------------------------------------------------
-                    Blur Anything Streamlit Application                      -
-------------------------------------------------------------------------------

This Streamlit application detects objects on an uploaded image and lets the
user blur them, either automatically by class (e.g. every person) or one by one.

By default detection runs in-process. Set BLURANYTHING_API_URL (for example
http://localhost:8000) to send images to the FastAPI backend instead.
"""

# frontend/streamlit_app.py

import io
import os
import sys
from pathlib import Path
from typing import List, Optional

import requests
import streamlit as st
from PIL import Image

# Allow `streamlit run frontend/streamlit_app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.blur import METHODS, blur_regions, draw_boxes  # noqa: E402

API_URL = os.getenv("BLURANYTHING_API_URL", "").rstrip("/")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))
METHOD_LABELS = {"gaussian": "Gaussian blur", "pixelate": "Pixelate", "solid": "Black box"}


def send_image_to_backend(
    image_bytes: bytes, filename: str, type_model: str, confidence: float
) -> Optional[dict]:
    """
    Send the image to the FastAPI backend for object detection.

    Returns:
        The detection results, or None if the request failed.
    """
    try:
        response = requests.post(
            f"{API_URL}/detect",
            files={"file": (filename, image_bytes)},
            data={"type_model": type_model, "confidence": confidence},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code != 200:
        st.error(payload.get("error", "Failed to detect objects in the image."))
        return None
    return payload


def run_detection(
    image_bytes: bytes, filename: str, type_model: str, confidence: float
) -> Optional[dict]:
    """Detect objects through the API when configured, in-process otherwise."""
    if API_URL:
        return send_image_to_backend(image_bytes, filename, type_model, confidence)

    from backend.detect import detect_objects  # pylint: disable=import-outside-toplevel

    image = Image.open(io.BytesIO(image_bytes))
    return detect_objects(image, type_model=type_model, confidence=confidence)


def select_regions(
    detected_objects: List[dict], auto_classes: List[str], manual_indices: List[int]
) -> List[dict]:
    """
    Pick the regions to blur: every object of the auto classes plus the manual picks.

    Returns:
        list: The selected detections, without duplicates, in detection order.
    """
    wanted = set(auto_classes)
    picked = set(manual_indices)
    return [
        obj
        for i, obj in enumerate(detected_objects)
        if obj["object"] in wanted or i in picked
    ]


def image_to_bytes(image: Image.Image, fmt: str) -> bytes:
    """Encode an image for the download button."""
    output = io.BytesIO()
    image.save(output, format=fmt, quality=95)
    return output.getvalue()


def sidebar_settings() -> dict:
    """Render the settings sidebar and return the chosen values."""
    st.sidebar.header("Detection")
    type_model = st.sidebar.radio(
        "Model",
        ["fast", "accurate"],
        format_func=lambda m: {"fast": "Fast (YOLOv8n)", "accurate": "Accurate (YOLOv8x)"}[m],
        horizontal=True,
    )
    confidence = st.sidebar.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)

    st.sidebar.header("Blur")
    method = st.sidebar.selectbox("Style", METHODS, format_func=METHOD_LABELS.get)
    strength = st.sidebar.slider(
        "Intensity", 2, 60, 18, disabled=method == "solid"
    )
    padding = st.sidebar.slider("Margin around objects", 0.0, 0.5, 0.05, 0.05)
    show_boxes = st.sidebar.checkbox("Show detection boxes", value=True)

    st.sidebar.caption(
        f"Backend: {'API ' + API_URL if API_URL else 'in-process'}"
    )
    return {
        "type_model": type_model,
        "confidence": confidence,
        "method": method,
        "strength": strength,
        "padding": padding,
        "show_boxes": show_boxes,
    }


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="BlurAnything", page_icon=":eye:", layout="wide")
    st.title("BlurAnything")
    st.caption("Detect people, cars, screens and 77 other kinds of objects, then blur them in one click.")

    settings = sidebar_settings()
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Detection is cached per image + model + confidence, so moving the blur
    # sliders does not re-run the model.
    cache_key = (uploaded.file_id, settings["type_model"], settings["confidence"])
    if st.session_state.get("cache_key") != cache_key:
        with st.spinner("Detecting objects..."):
            results = run_detection(
                image_bytes, uploaded.name, settings["type_model"], settings["confidence"]
            )
        if results is None:
            return
        st.session_state.cache_key = cache_key
        st.session_state.results = results

    detected = st.session_state.results["detected_objects"]
    if not detected:
        st.warning("No objects detected. Try lowering the minimum confidence.")
        st.image(image, use_container_width=True)
        return

    found_classes = sorted({obj["object"] for obj in detected})
    default_auto = [c for c in ("person",) if c in found_classes]
    auto_classes = st.multiselect(
        "Blur every…", found_classes, default=default_auto,
        help="All detected objects of these classes are blurred.",
    )
    labels = [f"{i}: {obj['object']} ({obj['confidence']:.0%})" for i, obj in enumerate(detected)]
    manual = st.multiselect(
        "…and these objects", range(len(detected)), format_func=labels.__getitem__
    )

    regions = select_regions(detected, auto_classes, manual)
    blurred = blur_regions(
        image, regions, settings["method"], settings["strength"], settings["padding"]
    )

    left, right = st.columns(2)
    with left:
        st.image(
            draw_boxes(image, detected) if settings["show_boxes"] else image,
            caption=f"{len(detected)} objects detected",
            use_container_width=True,
        )
    with right:
        st.image(blurred, caption=f"{len(regions)} regions blurred", use_container_width=True)
        is_png = uploaded.name.lower().endswith(".png")
        st.download_button(
            "Download blurred image",
            data=image_to_bytes(blurred, "PNG" if is_png else "JPEG"),
            file_name=f"{Path(uploaded.name).stem}_blurred.{'png' if is_png else 'jpg'}",
            mime="image/png" if is_png else "image/jpeg",
            type="primary",
        )

    with st.expander("Detection details"):
        st.json(st.session_state.results)


if __name__ == "__main__":
    main()
