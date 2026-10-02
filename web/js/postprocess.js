// YOLOv8-seg pre/post-processing. Pure functions (no DOM) so they run in the
// worker, in the browser and under `node --test`.
//
// Model contract (Ultralytics export, imgsz=640):
//   input   images   float32 [1, 3, 640, 640]   RGB, 0..1, letterboxed
//   output0          float32 [1, 4+80+32, 8400]  cx, cy, w, h | class scores | mask coeffs
//   output1          float32 [1, 32, 160, 160]   mask prototypes

export const INPUT_SIZE = 640;
export const PROTO_SIZE = 160;
export const NUM_CLASSES = 80;
export const NUM_MASK_COEFFS = 32;
export const LETTERBOX_GRAY = 114;

/**
 * Geometry of the letterbox used to fit an image into the square model input.
 * The image is scaled uniformly and centered; the rest is padding.
 */
export function letterbox(width, height, size = INPUT_SIZE) {
  const scale = Math.min(size / width, size / height);
  const drawW = Math.max(1, Math.round(width * scale));
  const drawH = Math.max(1, Math.round(height * scale));
  const padX = Math.floor((size - drawW) / 2);
  const padY = Math.floor((size - drawH) / 2);
  return { size, width, height, drawW, drawH, padX, padY, sx: drawW / width, sy: drawH / height };
}

/** RGBA bytes (size x size) -> planar float32 CHW in 0..1. */
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
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/** Class-aware greedy NMS. `cands` need x1,y1,x2,y2,score,classId. */
export function nms(cands, iouThreshold = 0.7, maxDet = 100) {
  const sorted = [...cands].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of sorted) {
    let suppressed = false;
    for (const k of kept) {
      if (k.classId === c.classId && iou(k, c) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push(c);
      if (kept.length >= maxDet) break;
    }
  }
  return kept;
}

/**
 * Read candidate detections out of the channel-major head output.
 * Element (channel c, anchor i) lives at `c * numAnchors + i`.
 */
export function decodeCandidates(out, numAnchors, numClasses, confThreshold, maxCandidates = 1000) {
  const cands = [];
  for (let i = 0; i < numAnchors; i++) {
    let best = -1;
    let bestScore = confThreshold;
    for (let c = 0; c < numClasses; c++) {
      const s = out[(4 + c) * numAnchors + i];
      if (s >= bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best < 0) continue;
    const cx = out[i];
    const cy = out[numAnchors + i];
    const w = out[2 * numAnchors + i];
    const h = out[3 * numAnchors + i];
    cands.push({
      anchor: i,
      classId: best,
      score: bestScore,
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
    });
  }
  if (cands.length > maxCandidates) {
    cands.sort((a, b) => b.score - a.score);
    cands.length = maxCandidates;
  }
  return cands;
}

/** coeffs (K) x protos (K x P x P)  ->  logits (P x P). */
export function maskLogits(coeffs, protos, protoSize = PROTO_SIZE) {
  const plane = protoSize * protoSize;
  const k = coeffs.length;
  const out = new Float32Array(plane);
  for (let m = 0; m < k; m++) {
    const c = coeffs[m];
    if (c === 0) continue;
    const off = m * plane;
    for (let p = 0; p < plane; p++) out[p] += c * protos[off + p];
  }
  return out;
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Full decode: threshold -> NMS -> map boxes back to the source image ->
 * per-detection mask logits (still in prototype space, see masks.js).
 *
 * @param {Float32Array} output0 head output  [1, 4+nc+nm, N]
 * @param {Float32Array} output1 prototypes   [1, nm, P, P]
 * @param {{numAnchors:number, numClasses?:number, numMasks?:number, protoSize?:number}} dims
 * @param {ReturnType<typeof letterbox>} lb
 */
export function decodeSegmentation(output0, output1, dims, lb, opts = {}) {
  const { conf = 0.15, iou: iouThr = 0.7, maxDet = 100 } = opts;
  const { numAnchors, numClasses = NUM_CLASSES, numMasks = NUM_MASK_COEFFS, protoSize = PROTO_SIZE } = dims;

  const cands = decodeCandidates(output0, numAnchors, numClasses, conf);
  const kept = nms(cands, iouThr, maxDet);

  return kept.map((c) => {
    const coeffs = new Float32Array(numMasks);
    for (let k = 0; k < numMasks; k++) coeffs[k] = output0[(4 + numClasses + k) * numAnchors + c.anchor];
    const box = [
      clamp((c.x1 - lb.padX) / lb.sx, 0, lb.width),
      clamp((c.y1 - lb.padY) / lb.sy, 0, lb.height),
      clamp((c.x2 - lb.padX) / lb.sx, 0, lb.width),
      clamp((c.y2 - lb.padY) / lb.sy, 0, lb.height),
    ];
    return {
      classId: c.classId,
      score: c.score,
      box,
      logits: maskLogits(coeffs, output1, protoSize),
    };
  });
}
