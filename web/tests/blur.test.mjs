import assert from "node:assert/strict";
import test from "node:test";

import { boxSizesForGauss, gaussianBlurRGBA } from "../blur.js";

function image(width, height, fill) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) data.set(fill(i % width, Math.floor(i / width)), i * 4);
  return { data, width, height };
}

const energy = ({ data }) => {
  let sum = 0;
  for (let i = 0; i < data.length - 4; i += 4) sum += Math.abs(data[i] - data[i + 4]);
  return sum;
};

test("box sizes are odd and grow with sigma", () => {
  for (const sigma of [1, 4, 18, 60]) {
    const sizes = boxSizesForGauss(sigma);
    assert.equal(sizes.length, 3);
    assert.ok(sizes.every((s) => s % 2 === 1));
  }
  assert.ok(boxSizesForGauss(40)[0] > boxSizesForGauss(4)[0]);
});

test("a flat image is unchanged", () => {
  const img = image(40, 30, () => [10, 120, 250, 255]);
  const before = Uint8ClampedArray.from(img.data);
  gaussianBlurRGBA(img, 8);
  assert.deepEqual(img.data, before);
});

test("blur removes high frequencies but keeps the mean and the alpha", () => {
  const img = image(64, 64, (x, y) => ((x + y) % 2 ? [0, 0, 0, 200] : [255, 255, 255, 200]));
  const mean = () => img.data.reduce((s, v, i) => (i % 4 === 0 ? s + v : s), 0) / (64 * 64);
  const before = { energy: energy(img), mean: mean() };
  gaussianBlurRGBA(img, 6);
  assert.ok(energy(img) < before.energy / 10);
  assert.ok(Math.abs(mean() - before.mean) < 6);
  assert.ok(img.data.every((v, i) => i % 4 !== 3 || v === 200));
});

test("a sharp edge becomes a gradient", () => {
  const img = image(60, 4, (x) => (x < 30 ? [0, 0, 0, 255] : [255, 255, 255, 255]));
  gaussianBlurRGBA(img, 5);
  const at = (x) => img.data[x * 4];
  assert.ok(at(0) < 5 && at(59) > 250);
  assert.ok(at(29) > 40 && at(29) < 215);
  assert.ok(at(20) < at(29) && at(29) < at(40));
});

test("tiny sigma is a no-op", () => {
  const img = image(8, 8, (x) => [x * 30, 0, 0, 255]);
  const before = Uint8ClampedArray.from(img.data);
  gaussianBlurRGBA(img, 0.2);
  assert.deepEqual(img.data, before);
});
