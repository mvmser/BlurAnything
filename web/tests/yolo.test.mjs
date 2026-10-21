import assert from "node:assert/strict";
import test from "node:test";

import { CLASSES, decode, letterbox, padBox, toTensorData } from "../yolo.js";

test("letterbox centers a wide image", () => {
  assert.deepEqual(letterbox(1280, 640), { scale: 0.5, w: 640, h: 320, padX: 0, padY: 160 });
});

test("toTensorData converts RGBA to normalised CHW", () => {
  const rgba = new Uint8ClampedArray([255, 0, 51, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
  const data = toTensorData(rgba, 2);
  assert.equal(data[0], 1);
  assert.equal(data[5], 1);
  assert.ok(Math.abs(data[8] - 0.2) < 1e-6);
});

function fakeOutput(boxes) {
  // boxes: [cx, cy, w, h, classId, score]
  const n = boxes.length;
  const out = new Float32Array((4 + CLASSES.length) * n);
  boxes.forEach(([cx, cy, w, h, cls, score], i) => {
    out[i] = cx; out[n + i] = cy; out[2 * n + i] = w; out[3 * n + i] = h;
    out[(4 + cls) * n + i] = score;
  });
  return out;
}

test("decode maps boxes back, thresholds and applies NMS per class", () => {
  const geometry = letterbox(1280, 640);
  const out = fakeOutput([
    [100, 260, 40, 40, 0, 0.9], // person
    [102, 260, 40, 40, 0, 0.8], // duplicate person, suppressed
    [102, 260, 40, 40, 2, 0.7], // car at the same spot, kept (other class)
    [300, 300, 10, 10, 0, 0.1], // below threshold
  ]);
  const dets = decode(out, 4, geometry, 1280, 640, 0.25);
  assert.equal(dets.length, 2);
  assert.equal(dets[0].object, "person");
  assert.ok(Math.abs(dets[0].confidence - 0.9) < 1e-6);
  assert.deepEqual(
    [dets[0].x_min, dets[0].y_min, dets[0].x_max, dets[0].y_max],
    [160, 160, 240, 240],
  );
  assert.equal(dets[1].object, "car");
});

test("padBox pads, clamps and rejects empty boxes", () => {
  assert.deepEqual(padBox({ x_min: 10, y_min: 10, x_max: 20, y_max: 20 }, 0.5, 100, 100), { x: 5, y: 5, w: 20, h: 20 });
  assert.deepEqual(padBox({ x_min: -5, y_min: 0, x_max: 200, y_max: 50 }, 0, 100, 100), { x: 0, y: 0, w: 100, h: 50 });
  assert.equal(padBox({ x_min: 150, y_min: 0, x_max: 200, y_max: 50 }, 0, 100, 100), null);
});
