// YOLOv8 pre/post-processing, framework free so it can be unit tested in Node.

export const CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

export const INPUT_SIZE = 640;

/**
 * Letterbox geometry: scale the image to fit INPUT_SIZE and center it.
 */
export function letterbox(width, height, size = INPUT_SIZE) {
  const scale = Math.min(size / width, size / height);
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);
  return { scale, w, h, padX: Math.floor((size - w) / 2), padY: Math.floor((size - h) / 2) };
}

/**
 * Convert RGBA pixels (size x size) into a normalised CHW float tensor.
 */
export function toTensorData(rgba, size = INPUT_SIZE) {
  const area = size * size;
  const data = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    data[i] = rgba[i * 4] / 255;
    data[i + area] = rgba[i * 4 + 1] / 255;
    data[i + 2 * area] = rgba[i * 4 + 2] / 255;
  }
  return data;
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a.x_max, b.x_max) - Math.max(a.x_min, b.x_min));
  const h = Math.max(0, Math.min(a.y_max, b.y_max) - Math.max(a.y_min, b.y_min));
  const inter = w * h;
  const area = (o) => (o.x_max - o.x_min) * (o.y_max - o.y_min);
  return inter / (area(a) + area(b) - inter || 1);
}

/**
 * Turn the raw [1, 84, N] output into detections in original image pixels.
 * Applies the confidence threshold and a per-class non max suppression.
 */
export function decode(output, numBoxes, geometry, imageWidth, imageHeight, confidence = 0.25, iouThreshold = 0.45) {
  const { scale, padX, padY } = geometry;
  const numClasses = CLASSES.length;
  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    let best = 0;
    let classId = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = output[(4 + c) * numBoxes + i];
      if (score > best) {
        best = score;
        classId = c;
      }
    }
    if (best < confidence) continue;

    const cx = output[i];
    const cy = output[numBoxes + i];
    const w = output[2 * numBoxes + i];
    const h = output[3 * numBoxes + i];
    const clampX = (v) => Math.min(imageWidth, Math.max(0, v));
    const clampY = (v) => Math.min(imageHeight, Math.max(0, v));
    candidates.push({
      object: CLASSES[classId],
      confidence: best,
      x_min: clampX((cx - w / 2 - padX) / scale),
      y_min: clampY((cy - h / 2 - padY) / scale),
      x_max: clampX((cx + w / 2 - padX) / scale),
      y_max: clampY((cy + h / 2 - padY) / scale),
    });
  }

  candidates.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const cand of candidates) {
    if (!kept.some((k) => k.object === cand.object && iou(k, cand) > iouThreshold)) kept.push(cand);
  }
  return kept;
}

/**
 * Pad a box by a ratio of its size, clamp it and round to integer pixels.
 * Returns null when the box is empty.
 */
export function padBox(box, padding, width, height) {
  const px = (box.x_max - box.x_min) * padding;
  const py = (box.y_max - box.y_min) * padding;
  const x = Math.max(0, Math.floor(box.x_min - px));
  const y = Math.max(0, Math.floor(box.y_min - py));
  const r = Math.min(width, Math.round(box.x_max + px));
  const b = Math.min(height, Math.round(box.y_max + py));
  return r > x && b > y ? { x, y, w: r - x, h: b - y } : null;
}
