// Turn model output / user shapes into alpha masks at any target resolution.
// A mask is { x0, y0, w, h, alpha } : an 8-bit alpha tile placed at (x0, y0)
// of a target raster of size (tw x th).

import { INPUT_SIZE, PROTO_SIZE } from './postprocess.js';

const CELL = INPUT_SIZE / PROTO_SIZE; // model pixels covered by one prototype cell
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Characteristic size of a box (geometric mean of its sides), in box units. */
export function boxSize(box) {
  return Math.sqrt(Math.max(1, (box[2] - box[0]) * (box[3] - box[1])));
}

/**
 * Rasterize a detection's mask.
 *
 * The 160x160 logits live in letterboxed model space; each target pixel is
 * mapped back to it and bilinearly sampled, so edges stay smooth whatever the
 * target resolution. The mask is cropped to the detection box (like
 * Ultralytics does) and `margin` (0..1) grows it a little so that edges are
 * fully covered.
 *
 * @param {Float32Array} logits   PROTO_SIZE x PROTO_SIZE
 * @param {{width:number,height:number,padX:number,padY:number,sx:number,sy:number}} lb
 * @param {number[]} box          [x1,y1,x2,y2] in source image pixels
 * @param {number} tw             target raster width
 * @param {number} th             target raster height
 * @param {{margin?:number}} [opts]
 */
export function rasterizeObjectMask(logits, lb, box, tw, th, { margin = 0.25 } = {}) {
  const kx = tw / lb.width;
  const ky = th / lb.height;
  const size = boxSize(box);
  const grow = margin * 0.06 * size; // source px
  const bias = margin * 4; // logits

  const X0 = Math.max(0, Math.floor((box[0] - grow) * kx));
  const X1 = Math.min(tw, Math.ceil((box[2] + grow) * kx));
  const Y0 = Math.max(0, Math.floor((box[1] - grow) * ky));
  const Y1 = Math.min(th, Math.ceil((box[3] + grow) * ky));
  const w = X1 - X0;
  const h = Y1 - Y0;
  if (w <= 0 || h <= 0) return null;

  const last = PROTO_SIZE - 1;
  const xi = new Int32Array(w);
  const xf = new Float32Array(w);
  for (let i = 0; i < w; i++) {
    const mx = lb.padX + ((X0 + i + 0.5) / kx) * lb.sx;
    const g = clamp(mx / CELL - 0.5, 0, last);
    const i0 = Math.min(last - 1, Math.floor(g));
    xi[i] = i0;
    xf[i] = g - i0;
  }

  const alpha = new Uint8ClampedArray(w * h);
  for (let j = 0; j < h; j++) {
    const my = lb.padY + ((Y0 + j + 0.5) / ky) * lb.sy;
    const g = clamp(my / CELL - 0.5, 0, last);
    const j0 = Math.min(last - 1, Math.floor(g));
    const fy = g - j0;
    const r0 = j0 * PROTO_SIZE;
    const r1 = r0 + PROTO_SIZE;
    const row = j * w;
    for (let i = 0; i < w; i++) {
      const a = xi[i];
      const fx = xf[i];
      const top = logits[r0 + a] * (1 - fx) + logits[r0 + a + 1] * fx;
      const bot = logits[r1 + a] * (1 - fx) + logits[r1 + a + 1] * fx;
      const logit = top * (1 - fy) + bot * fy;
      alpha[row + i] = (0.5 + (logit + bias) / 4) * 255; // clamped by Uint8ClampedArray
    }
  }
  return { x0: X0, y0: Y0, w, h, alpha };
}

/**
 * Rasterize a user-drawn shape ('rect' | 'ellipse') with anti-aliased edges.
 * `box` is in source pixels (srcW x srcH); the target raster is tw x th.
 */
export function rasterizeShapeMask(shape, box, srcW, srcH, tw, th) {
  const kx = tw / srcW;
  const ky = th / srcH;
  const bx1 = box[0] * kx;
  const by1 = box[1] * ky;
  const bx2 = box[2] * kx;
  const by2 = box[3] * ky;

  const X0 = Math.max(0, Math.floor(bx1));
  const X1 = Math.min(tw, Math.ceil(bx2));
  const Y0 = Math.max(0, Math.floor(by1));
  const Y1 = Math.min(th, Math.ceil(by2));
  const w = X1 - X0;
  const h = Y1 - Y0;
  if (w <= 0 || h <= 0) return null;

  const alpha = new Uint8ClampedArray(w * h);
  if (shape === 'ellipse') {
    const cx = (bx1 + bx2) / 2;
    const cy = (by1 + by2) / 2;
    const rx = Math.max(0.5, (bx2 - bx1) / 2);
    const ry = Math.max(0.5, (by2 - by1) / 2);
    const rmin = Math.min(rx, ry);
    for (let j = 0; j < h; j++) {
      const dy = (Y0 + j + 0.5 - cy) / ry;
      for (let i = 0; i < w; i++) {
        const dx = (X0 + i + 0.5 - cx) / rx;
        // distance to the outline in pixels (first-order approximation)
        const d = (1 - Math.sqrt(dx * dx + dy * dy)) * rmin;
        alpha[j * w + i] = clamp(d + 0.5, 0, 1) * 255;
      }
    }
  } else {
    for (let j = 0; j < h; j++) {
      const py = Y0 + j;
      const cy = clamp(Math.min(py + 1, by2) - Math.max(py, by1), 0, 1);
      for (let i = 0; i < w; i++) {
        const px = X0 + i;
        const cx = clamp(Math.min(px + 1, bx2) - Math.max(px, bx1), 0, 1);
        alpha[j * w + i] = cx * cy * 255;
      }
    }
  }
  return { x0: X0, y0: Y0, w, h, alpha };
}
