// Before/after the YOLOv8-seg model (Ultralytics export, 640x640):
//   input    [1, 3, 640, 640]    RGB 0..1, the picture letterboxed on gray
//   output0  [1, 4+80+32, 8400]  per anchor: cx, cy, w, h | 80 class scores | 32 mask coefficients
//   output1  [1, 32, 160, 160]   mask prototypes
// No DOM in here, so it runs in the worker and in node tests.

export const INPUT_SIZE = 640;
export const PROTO_SIZE = 160;
export const NUM_CLASSES = 80;
export const NUM_MASK_COEFFS = 32;
export const LETTERBOX_GRAY = 114;

// where the picture goes inside the square input: scaled to fit, centered, padded
export function letterbox(width, height, size = INPUT_SIZE) {
  const scale = Math.min(size / width, size / height);
  const drawW = Math.max(1, Math.round(width * scale));
  const drawH = Math.max(1, Math.round(height * scale));
  const padX = Math.floor((size - drawW) / 2);
  const padY = Math.floor((size - drawH) / 2);
  return { size, width, height, drawW, drawH, padX, padY, sx: drawW / width, sy: drawH / height };
}

// RGBA bytes -> planar float32 (all R, then all G, then all B), 0..1
export function rgbaToChw(rgba, size = INPUT_SIZE) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = rgba[j] / 255;
    out[n + i] = rgba[j + 1] / 255;
    out[2 * n + i] = rgba[j + 2] / 255;
  }
  return out;
}

export function iou(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

// greedy non-max suppression, one class at a time
export function nms(boxes, iouThreshold = 0.7, maxDet = 100) {
  const kept = [];
  for (const box of [...boxes].sort((a, b) => b.score - a.score)) {
    const duplicate = kept.some((k) => k.classId === box.classId && iou(k, box) > iouThreshold);
    if (duplicate) continue;
    kept.push(box);
    if (kept.length >= maxDet) break;
  }
  return kept;
}

// The head output is channel-major: value c of anchor i is at c * numAnchors + i.
export function decodeCandidates(out, numAnchors, numClasses, confThreshold, maxCandidates = 1000) {
  const found = [];
  for (let i = 0; i < numAnchors; i++) {
    let classId = -1;
    let score = confThreshold;
    for (let c = 0; c < numClasses; c++) {
      const s = out[(4 + c) * numAnchors + i];
      if (s >= score) {
        score = s;
        classId = c;
      }
    }
    if (classId < 0) continue;
    const cx = out[i];
    const cy = out[numAnchors + i];
    const w = out[2 * numAnchors + i];
    const h = out[3 * numAnchors + i];
    found.push({ anchor: i, classId, score, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  return found.sort((a, b) => b.score - a.score).slice(0, maxCandidates);
}

// mask of one object = its 32 coefficients combined with the 32 prototype images
export function maskLogits(coeffs, protos, protoSize = PROTO_SIZE) {
  const plane = protoSize * protoSize;
  const out = new Float32Array(plane);
  coeffs.forEach((c, k) => {
    if (c === 0) return;
    for (let p = 0; p < plane; p++) out[p] += c * protos[k * plane + p];
  });
  return out;
}

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

// Boxes come back in picture pixels, masks stay at the model's 160x160 (see masks.js).
export function decodeSegmentation(output0, output1, dims, lb, opts = {}) {
  const { conf = 0.15, iou: iouThreshold = 0.7, maxDet = 100 } = opts;
  const { numAnchors, numClasses = NUM_CLASSES, numMasks = NUM_MASK_COEFFS, protoSize = PROTO_SIZE } = dims;

  const found = nms(decodeCandidates(output0, numAnchors, numClasses, conf), iouThreshold, maxDet);
  return found.map((d) => {
    const coeffs = new Float32Array(numMasks);
    for (let k = 0; k < numMasks; k++) coeffs[k] = output0[(4 + numClasses + k) * numAnchors + d.anchor];
    return {
      classId: d.classId,
      score: d.score,
      box: [
        clamp((d.x1 - lb.padX) / lb.sx, 0, lb.width),
        clamp((d.y1 - lb.padY) / lb.sy, 0, lb.height),
        clamp((d.x2 - lb.padX) / lb.sx, 0, lb.width),
        clamp((d.y2 - lb.padY) / lb.sy, 0, lb.height),
      ],
      logits: maskLogits(coeffs, output1, protoSize),
    };
  });
}
