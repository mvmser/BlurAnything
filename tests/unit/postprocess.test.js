import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  letterbox,
  rgbaToChw,
  iou,
  nms,
  maskLogits,
  decodeCandidates,
  decodeSegmentation,
} from '../../web/js/postprocess.js';

test('letterbox: landscape image is padded top/bottom', () => {
  const lb = letterbox(1280, 720, 640);
  assert.equal(lb.drawW, 640);
  assert.equal(lb.drawH, 360);
  assert.equal(lb.padX, 0);
  assert.equal(lb.padY, 140);
  assert.equal(lb.sx, 0.5);
  assert.equal(lb.sy, 0.5);
});

test('letterbox: portrait image is padded left/right, upscales small images', () => {
  const lb = letterbox(300, 600, 640);
  assert.equal(lb.drawW, 320);
  assert.equal(lb.drawH, 640);
  assert.equal(lb.padX, 160);
  assert.equal(lb.padY, 0);
  assert.ok(lb.sx > 1);
});

test('rgbaToChw: planar layout, 0..1 range, alpha dropped', () => {
  // 2x2 image, pixels: red, green, blue, white
  const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
  const chw = rgbaToChw(rgba, 2);
  assert.equal(chw.length, 12);
  assert.deepEqual(Array.from(chw.slice(0, 4)), [1, 0, 0, 1]); // R plane
  assert.deepEqual(Array.from(chw.slice(4, 8)), [0, 1, 0, 1]); // G plane
  assert.deepEqual(Array.from(chw.slice(8, 12)), [0, 0, 1, 1]); // B plane
});

test('iou: identical, disjoint and half-overlapping boxes', () => {
  const a = { x1: 0, y1: 0, x2: 10, y2: 10 };
  assert.equal(iou(a, a), 1);
  assert.equal(iou(a, { x1: 20, y1: 20, x2: 30, y2: 30 }), 0);
  assert.ok(Math.abs(iou(a, { x1: 5, y1: 0, x2: 15, y2: 10 }) - 1 / 3) < 1e-9);
});

test('nms: suppresses same-class duplicates, keeps other classes and far boxes', () => {
  const box = (x1, score, classId) => ({ x1, y1: 0, x2: x1 + 100, y2: 100, score, classId });
  const kept = nms([box(0, 0.9, 0), box(5, 0.8, 0), box(5, 0.7, 1), box(500, 0.6, 0)], 0.7, 10);
  assert.deepEqual(
    kept.map((k) => [k.score, k.classId]),
    [
      [0.9, 0],
      [0.7, 1],
      [0.6, 0],
    ],
  );
});

test('nms: honours maxDet', () => {
  const cands = Array.from({ length: 20 }, (_, i) => ({
    x1: i * 200,
    y1: 0,
    x2: i * 200 + 100,
    y2: 100,
    score: 1 - i / 100,
    classId: 0,
  }));
  assert.equal(nms(cands, 0.7, 5).length, 5);
});

test('maskLogits: linear combination of prototypes', () => {
  const protos = new Float32Array([1, 2, 3, 4, /* proto 1 */ 10, 20, 30, 40]);
  const out = maskLogits(new Float32Array([2, -1]), protos, 2);
  assert.deepEqual(Array.from(out), [2 * 1 - 10, 2 * 2 - 20, 2 * 3 - 30, 2 * 4 - 40]);
});

// Head output with `n` anchors, `nc` classes, `nm` mask coefficients, channel-major.
function makeHead(n, nc, nm) {
  return new Float32Array((4 + nc + nm) * n);
}
function setAnchor(out, n, nc, i, { cx, cy, w, h, cls, score, coeffs }) {
  out[i] = cx;
  out[n + i] = cy;
  out[2 * n + i] = w;
  out[3 * n + i] = h;
  out[(4 + cls) * n + i] = score;
  coeffs.forEach((c, k) => (out[(4 + nc + k) * n + i] = c));
}

test('decodeCandidates: picks best class and respects threshold', () => {
  const n = 3,
    nc = 3,
    nm = 2;
  const out = makeHead(n, nc, nm);
  setAnchor(out, n, nc, 0, { cx: 50, cy: 50, w: 20, h: 40, cls: 2, score: 0.9, coeffs: [0, 0] });
  setAnchor(out, n, nc, 1, { cx: 80, cy: 80, w: 20, h: 20, cls: 1, score: 0.1, coeffs: [0, 0] });
  const c = decodeCandidates(out, n, nc, 0.25);
  assert.equal(c.length, 1);
  assert.equal(c[0].classId, 2);
  assert.equal(c[0].score, Math.fround(0.9));
  assert.deepEqual([c[0].x1, c[0].y1, c[0].x2, c[0].y2], [40, 30, 60, 70]);
});

test('decodeSegmentation: maps boxes back through the letterbox and builds mask logits', () => {
  const n = 4,
    nc = 3,
    nm = 2,
    P = 4;
  const out = makeHead(n, nc, nm);
  // box in model space: 270..370 x 220..420 ; 1280x720 image => pad (0,140), scale .5
  setAnchor(out, n, nc, 1, { cx: 320, cy: 320, w: 100, h: 200, cls: 2, score: 0.9, coeffs: [1, -1] });
  const protos = new Float32Array(nm * P * P);
  protos.fill(2, 0, P * P); // proto 0 = 2
  protos.fill(1, P * P); // proto 1 = 1
  const lb = letterbox(1280, 720, 640);

  const dets = decodeSegmentation(out, protos, { numAnchors: n, numClasses: nc, numMasks: nm, protoSize: P }, lb, {
    conf: 0.25,
  });
  assert.equal(dets.length, 1);
  const d = dets[0];
  assert.equal(d.classId, 2);
  assert.deepEqual(d.box, [540, 160, 740, 560]);
  assert.equal(d.logits.length, P * P);
  assert.ok(d.logits.every((v) => v === 1)); // 1*2 + (-1)*1
});

test('decodeSegmentation: clamps boxes to the image', () => {
  const n = 1,
    nc = 1,
    nm = 1,
    P = 2;
  const out = makeHead(n, nc, nm);
  setAnchor(out, n, nc, 0, { cx: 10, cy: 320, w: 100, h: 100, cls: 0, score: 0.8, coeffs: [0] });
  const lb = letterbox(640, 640, 640);
  const [d] = decodeSegmentation(
    out,
    new Float32Array(nm * P * P),
    { numAnchors: n, numClasses: nc, numMasks: nm, protoSize: P },
    lb,
  );
  assert.equal(d.box[0], 0);
  assert.equal(d.box[2], 60);
});
