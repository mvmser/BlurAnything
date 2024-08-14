"""
-------------------------------------------------------------------------------
-                    Blur Anything Streamlit Application                      -
-------------------------------------------------------------------------------

This Streamlit application interfaces with a FastAPI backend to perform object
detection and blurring on uploaded images. Users can upload images, and the app
will display detected objects. The backend uses YOLOv8 for object detection.
"""

# frontend/streamlit_app.py

import os

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from requests.exceptions import JSONDecodeError


def upload_image_ui():
    """
    Create a UI for image upload.

    Returns:
        The uploaded file object if an image is uploaded, otherwise None.
    """
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    return uploaded_image


def display_uploaded_image(uploaded_image):
    """
    Display the uploaded image in the Streamlit app.

    Args:
        uploaded_image: The image file uploaded by the user.
    """
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)


def send_image_to_backend(uploaded_image, type_model):
    """
    Send the uploaded image to the FastAPI backend for object detection.

    Args:
        uploaded_image: The image file uploaded by the user.

    Returns:
        The response from the backend API.
    """
    img_bytes = uploaded_image.read()
    mime_type = uploaded_image.type
    files = {"file": ("image.jpg", img_bytes, mime_type)}
    data = {"type_model": type_model}

    fastapi_backend_url = "http://localhost:8000/detect/"
    timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "10"))

    try:
        response = requests.post(
            fastapi_backend_url, files=files, data=data, timeout=timeout_seconds
        )
        return response
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None


def display_detection_results(response, display_image) -> dict:
    """
    Display the object detection results or an error message.

    Args:
        response: The response from the backend API.
    """
    if response.status_code == 200:
        try:
            detection_results = response.json()
            st.write("Detection JSON Results:", detection_results)

            if detection_results["detected_objects"]:
                display_image_boxes = draw_boxes(display_image, detection_results)
                st.image(
                    display_image_boxes,
                    caption="Detected Objects",
                    use_column_width=True,
                )
                return detection_results["detected_objects"]
            st.warning("No objects detected.")

        except JSONDecodeError:
            st.error("Failed to decode the response from the server.")
    else:
        try:
            error_message = response.json().get(
                "detail", "Failed to detect objects in the image."
            )
        except JSONDecodeError:
            error_message = "Failed to detect objects in the image."
        st.error(error_message)
    return {}


def draw_boxes(image: Image.Image, detection_results: dict):
    """
    Draw bounding boxes around detected objects on the image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        detection_results (dict): The dictionary containing detection results.

    Returns:
        PIL.Image.Image: The image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    for obj in detection_results["detected_objects"]:
        box = (
            obj["x_min"],
            obj["y_min"],
            obj["x_max"],
            obj["y_max"],
        )
        draw.rectangle(box, outline="red", width=1)
        draw.text(
            (obj["x_min"], obj["y_min"]),
            f"{obj['object']} {obj['confidence']:.2f}",
            fill="red",
        )
    return image


def blur_objects(image, detection_results, selected_indices, blur_radius=5):
    """
    Blur the selected objects in the image based on their indices.

    Args:
        image (PIL.Image.Image): The image to blur.
        detection_results (list): The list containing detection results.
        selected_indices (list): The list of indices of selected objects to blur.
        blur_radius (int): The radius of the Gaussian blur.
            Higher values produce a more pronounced blur.

    Returns:
        PIL.Image.Image: The image with selected objects blurred.
    """
    for index in selected_indices:
        obj = detection_results[index]

        box = (
            int(obj["x_min"]),
            int(obj["y_min"]),
            int(obj["x_max"]),
            int(obj["y_max"]),
        )
        region = image.crop(box)

        blurred_region = region.filter(ImageFilter.GaussianBlur(blur_radius))
        image.paste(blurred_region, box)
    return image


def select_object_blur(display_image, detected_objects):
    """
    Allow users to select detected objects by their indices for blurring and select blur intensity.

    Args:
        display_image (PIL.Image.Image): The image displayed.
        detected_objects (list): A list of dictionaries containing detected object info.
    """
    st.sidebar.title("Select Objects to Blur")

    # Slider for selecting blur radius (moved outside the if condition)
    blur_radius = st.sidebar.slider(
        "Blur Intensity", min_value=1, max_value=30, value=10, key="blur_radius"
    )

    object_labels = [
        f"{i}: {obj['object']} ({obj['confidence']:.2f})"
        for i, obj in enumerate(detected_objects)
    ]

    selected_indices = st.multiselect(
        "Objects",
        options=range(len(detected_objects)),
        format_func=lambda x: object_labels[x],
    )

    if st.button("Blur"):
        if selected_indices:
            blurred_image = blur_objects(
                display_image,
                detected_objects,
                selected_indices,
                blur_radius=blur_radius,
            )
            st.image(blurred_image, caption="Blurred Image", use_column_width=True)
        else:
            st.warning("No objects selected for blurring.")


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="BlurAnything",
        page_icon=":eye:",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("Blur Anything - Object Detection and Blurring")

    type_model = st.selectbox("Select the model type:", ["fast", "accurate"], index=0)
    uploaded_image = upload_image_ui()
    display_uploaded_image(uploaded_image)

    if uploaded_image is not None:
        if "selected_objects" not in st.session_state:
            st.session_state.selected_objects = []
        if "detected_objects" not in st.session_state:
            st.session_state.detected_objects = {}
        if "display_image" not in st.session_state:
            st.session_state.display_image = None

        response = None
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                response = send_image_to_backend(
                    uploaded_image=uploaded_image, type_model=type_model
                )
            display_image = Image.open(uploaded_image).convert("RGB")
            original_image = Image.open(uploaded_image).convert("RGB")
            st.session_state.display_image = display_image
            st.session_state.original_image = original_image

            st.session_state.detected_objects = display_detection_results(
                response, display_image
            )

        if st.session_state.detected_objects:
            select_object_blur(
                st.session_state.original_image, st.session_state.detected_objects
            )

        if response and st.button("Retry"):
            st.rerun()


if __name__ == "__main__":
    main()
