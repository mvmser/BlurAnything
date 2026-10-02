// Static configuration shared by the UI and the detector.

// Absolute URLs resolved from this file, so they work from the page *and* from
// the worker, and under any base path (e.g. https://example.com/BlurAnything/).
const asset = (path) => new URL(`../${path}`, import.meta.url).href;

/**
 * Models served from /models. `bytes` is the exact file size (integrity check +
 * progress). `sha256` is checked by the unit tests; its first characters version
 * the URL, so a re-exported model can never be served from a stale HTTP cache.
 */
const model = (id, name, file, bytes, sha256) => ({
  id,
  name,
  url: `${asset(`models/${file}`)}?v=${sha256.slice(0, 10)}`,
  bytes,
  sha256,
});
export const MODELS = {
  fast: model(
    'fast',
    'YOLOv8n-seg',
    'yolov8n-seg.onnx',
    7003966,
    '4688816ae6ef67cc971aa29607fdab550f0bf4bf10c7c24fdc668e0a6f949818',
  ),
  accurate: model(
    'accurate',
    'YOLOv8s-seg',
    'yolov8s-seg.onnx',
    23827387,
    '234a2f9b8eca151a4c2f7535a4e5a43ffec188199ff1870fff475772bebabcdb',
  ),
};

/** Bump when the model files change so that cached copies are not reused. */
export const MODEL_CACHE = 'blur-anything-models-v1';

export const SAMPLES = ['bus', 'astronaut', 'cat', 'coffee'].map((id) => ({ id, url: asset(`samples/${id}.jpg`) }));

export const LIMITS = {
  /** Larger images are downscaled (canvas memory limits, notably on iOS). */
  maxPixels: 16_000_000,
  /** Longest side of the interactive preview. Export always uses full resolution. */
  previewMax: 1600,
  /** Detections below this score are never kept; the UI slider filters above it. */
  minConf: 0.15,
  iou: 0.7,
  maxDetections: 100,
};

export const DEFAULTS = {
  model: 'fast',
  style: 'blur',
  strength: 55,
  margin: 25,
  conf: 0.3,
};

/** COCO class names, in the order used by the model. */
export const COCO_CLASSES = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];
