# BlurAnything API + Streamlit UI (CPU).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY backend backend
COPY frontend frontend

# Bake the fast model in the image so the first request does not download it.
RUN python -c "from backend.detect import load_model; load_model('fast')"

EXPOSE 8000 8501
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
