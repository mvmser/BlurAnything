// Absolute urls, so they work from the page and from the worker, under any base path.
const asset = (path) => new URL(`../${path}`, import.meta.url).href;

// bytes and sha256 are checked by the unit tests. The start of the hash goes in the url,
// so a re-exported model never comes from a stale cache.
const model = (id, name, file, bytes, sha256) => ({
  id,
  name,
  bytes,
  sha256,
  url: `${asset(`models/${file}`)}?v=${sha256.slice(0, 10)}`,
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

// bump when the model files change
export const MODEL_CACHE = 'blur-anything-models-v1';

export const SAMPLES = ['bus', 'astronaut', 'cat', 'coffee'].map((id) => ({ id, url: asset(`samples/${id}.jpg`) }));

export const LIMITS = {
  maxPixels: 16_000_000, // bigger images are downscaled (canvas limits, iOS especially)
  previewMax: 1600, // longest side of the preview, the export uses the full image
  minConf: 0.15, // lowest score kept, the slider filters above it
  iou: 0.7,
  maxDetections: 100,
};

export const DEFAULTS = { model: 'fast', style: 'blur', strength: 55, margin: 25, conf: 0.3 };

// the 80 COCO classes, in the order of the model outputs
export const COCO_CLASSES = `person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat,
baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed,
dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush`.split(/,\s*/);
