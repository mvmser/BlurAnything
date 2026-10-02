import { test } from 'node:test';
import assert from 'node:assert/strict';
import { letterbox, PROTO_SIZE, INPUT_SIZE } from '../../web/js/postprocess.js';
import { rasterizeObjectMask, rasterizeShapeMask, boxSize } from '../../web/js/masks.js';

// Disk of radius `rCells` (in 160-grid cells) centred in the prototype grid.
function diskLogits(rCells, cx = 80, cy = 80, hi = 8, lo = -8) {
  const l = new Float32Array(PROTO_SIZE * PROTO_SIZE);
  for (let j = 0; j < PROTO_SIZE; j++)
    for (let i = 0; i < PROTO_SIZE; i++) l[j * PROTO_SIZE + i] = Math.hypot(i - cx, j - cy) <= rCells ? hi : lo;
  return l;
}
const at = (m, w, x, y) => {
  const i = x - m.x0;
  const j = y - m.y0;
  if (i < 0 || j < 0 || i >= m.w || j >= m.h) return 0;
  return m.alpha[j * m.w + i];
};
const area = (m) => m.alpha.reduce((s, v) => s + v, 0) / 255;

test('boxSize is the geometric mean of the sides', () => {
  assert.equal(boxSize([0, 0, 100, 400]), 200);
});

test('rasterizeObjectMask: identity letterbox (640x640 square image)', () => {
  const lb = letterbox(640, 640, INPUT_SIZE);
  const m = rasterizeObjectMask(diskLogits(30), lb, [190, 190, 450, 450], 640, 640, { margin: 0 });
  assert.equal(at(m, 640, 320, 320), 255);
  assert.equal(at(m, 640, 320 + 100, 320), 255); // inside (radius 120 px)
  assert.equal(at(m, 640, 320 + 140, 320), 0); // outside
});

test('rasterizeObjectMask: non-square image, mask lands on the right pixels', () => {
  // 1280x720 => pad (0,140), scale 0.5. Disk centre (model 320,320) => image (640,360), radius 240 px.
  const lb = letterbox(1280, 720, INPUT_SIZE);
  const box = [390, 110, 890, 610];
  const m = rasterizeObjectMask(diskLogits(30), lb, box, 1280, 720, { margin: 0 });
  assert.equal(at(m, 1280, 640, 360), 255);
  assert.equal(at(m, 1280, 640 + 215, 360), 255);
  assert.equal(at(m, 1280, 640 - 215, 360), 255);
  assert.equal(at(m, 1280, 640, 360 + 215), 255);
  assert.equal(at(m, 1280, 640 + 265, 360), 0);
  assert.equal(at(m, 1280, 640, 360 - 265), 0);
});

test('rasterizeObjectMask: same shape at half resolution (resolution independent)', () => {
  const lb = letterbox(1280, 720, INPUT_SIZE);
  const box = [390, 110, 890, 610];
  const full = rasterizeObjectMask(diskLogits(30), lb, box, 1280, 720, { margin: 0 });
  const half = rasterizeObjectMask(diskLogits(30), lb, box, 640, 360, { margin: 0 });
  const ratio = area(full) / (area(half) * 4);
  assert.ok(Math.abs(ratio - 1) < 0.02, `area ratio ${ratio}`);
});

test('rasterizeObjectMask: mask is cropped to the box', () => {
  const lb = letterbox(640, 640, INPUT_SIZE);
  // disk bigger than the box: nothing may leak outside the box
  const m = rasterizeObjectMask(diskLogits(60), lb, [300, 300, 340, 340], 640, 640, { margin: 0 });
  assert.equal(m.x0, 300);
  assert.equal(m.w, 40);
  assert.equal(at(m, 640, 299, 320), 0);
  assert.equal(at(m, 640, 341, 320), 0);
  assert.equal(at(m, 640, 320, 320), 255);
});

test('rasterizeObjectMask: margin grows the mask', () => {
  const lb = letterbox(640, 640, INPUT_SIZE);
  const soft = (logit) => {
    // smooth radial logit so the 0.5 level can actually move
    const l = new Float32Array(PROTO_SIZE * PROTO_SIZE);
    for (let j = 0; j < PROTO_SIZE; j++)
      for (let i = 0; i < PROTO_SIZE; i++) l[j * PROTO_SIZE + i] = (30 - Math.hypot(i - 80, j - 80)) * logit;
    return l;
  };
  const box = [150, 150, 490, 490];
  const a0 = area(rasterizeObjectMask(soft(1), lb, box, 640, 640, { margin: 0 }));
  const a1 = area(rasterizeObjectMask(soft(1), lb, box, 640, 640, { margin: 1 }));
  assert.ok(a1 > a0 * 1.05, `${a1} vs ${a0}`);
});

test('rasterizeObjectMask: box fully outside the target returns null', () => {
  const lb = letterbox(640, 640, INPUT_SIZE);
  assert.equal(rasterizeObjectMask(diskLogits(30), lb, [700, 700, 800, 800], 640, 640), null);
});

test('rasterizeShapeMask: rect area is exact (anti-aliased edges)', () => {
  const m = rasterizeShapeMask('rect', [10.5, 10, 20, 20.5], 100, 100, 100, 100);
  assert.ok(Math.abs(area(m) - 9.5 * 10.5) < 0.5, `${area(m)}`);
  const sharp = rasterizeShapeMask('rect', [10, 10, 20, 20], 100, 100, 100, 100);
  assert.equal(area(sharp), 100);
});

test('rasterizeShapeMask: ellipse area ~ pi*rx*ry and centre/corner behave', () => {
  const m = rasterizeShapeMask('ellipse', [20, 20, 80, 60], 100, 100, 100, 100);
  const expected = Math.PI * 30 * 20;
  assert.ok(Math.abs(area(m) - expected) / expected < 0.03, `${area(m)} vs ${expected}`);
  assert.equal(at(m, 100, 50, 40), 255);
  assert.equal(at(m, 100, 21, 21), 0); // bounding-box corner is outside the ellipse
});

test('rasterizeShapeMask: scales with the target resolution', () => {
  const a = area(rasterizeShapeMask('rect', [100, 100, 300, 200], 1000, 500, 1000, 500));
  const b = area(rasterizeShapeMask('rect', [100, 100, 300, 200], 1000, 500, 500, 250));
  assert.ok(Math.abs(a / (b * 4) - 1) < 0.01);
});
