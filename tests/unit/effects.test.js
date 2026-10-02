import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boxesForGauss, gaussianBlurRGBA, pixelateRGBA, effectParams, applyEffects } from '../../web/js/effects.js';

function image(w, h, fn) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fn(x, y);
      const i = (y * w + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  return { width: w, height: h, data };
}
const px = (img, x, y) => Array.from(img.data.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4));
const fullMask = (w, h, a = 255) => ({ x0: 0, y0: 0, w, h, alpha: new Uint8ClampedArray(w * h).fill(a) });

test('boxesForGauss: odd sizes whose variance matches sigma^2', () => {
  // odd box widths make the approximation coarse for very small sigmas
  for (const sigma of [2, 5, 12, 40]) {
    const sizes = boxesForGauss(sigma, 3);
    assert.equal(sizes.length, 3);
    assert.ok(sizes.every((s) => s % 2 === 1));
    const variance = sizes.reduce((v, s) => v + (s * s - 1) / 12, 0);
    assert.ok(
      Math.abs(variance - sigma * sigma) / (sigma * sigma) < (sigma < 4 ? 0.2 : 0.1),
      `sigma ${sigma}: var ${variance}`,
    );
  }
});

test('gaussianBlurRGBA: constant image is unchanged, alpha untouched', () => {
  const img = image(17, 13, () => [50, 100, 150]);
  const before = new Uint8ClampedArray(img.data);
  gaussianBlurRGBA(img.data, img.width, img.height, 4);
  assert.deepEqual(img.data, before);
});

test('gaussianBlurRGBA: symmetric spread that roughly conserves energy', () => {
  const W = 61;
  const img = image(W, W, (x, y) => (x >= 26 && x <= 34 && y >= 26 && y <= 34 ? [255, 255, 255] : [0, 0, 0]));
  const sum = (c) => {
    let s = 0;
    for (let i = c; i < img.data.length; i += 4) s += img.data[i];
    return s;
  };
  const before = sum(0);
  gaussianBlurRGBA(img.data, W, W, 3);
  assert.ok(Math.abs(sum(0) - before) / before < 0.03);
  assert.deepEqual(px(img, 30 + 7, 30), px(img, 30 - 7, 30));
  assert.deepEqual(px(img, 30, 30 + 7), px(img, 30, 30 - 7));
  assert.ok(px(img, 30, 30)[0] < 255 && px(img, 30, 30)[0] > px(img, 30 + 7, 30)[0]);
  // R, G, B treated identically
  assert.equal(px(img, 33, 31)[0], px(img, 33, 31)[1]);
  assert.equal(px(img, 0, 0)[3], 255);
});

test('gaussianBlurRGBA: edge pixels stay put on a horizontal gradient', () => {
  const img = image(40, 4, (x) => [x * 6, x * 6, x * 6]);
  gaussianBlurRGBA(img.data, 40, 4, 2);
  // gradient is symmetric about the middle: values stay monotonic non-decreasing
  for (let x = 1; x < 40; x++) assert.ok(px(img, x, 1)[0] >= px(img, x - 1, 1)[0]);
});

test('pixelateRGBA: tiles become their average', () => {
  const img = image(4, 4, (x, y) => [x < 2 && y < 2 ? [0, 10, 20, 30][y * 2 + x] : 200, 0, 0]);
  pixelateRGBA(img.data, 4, 4, 2);
  for (const [x, y] of [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ])
    assert.equal(px(img, x, y)[0], 15);
  assert.equal(px(img, 3, 3)[0], 200);
});

test('pixelateRGBA: handles sizes that are not a multiple of the block', () => {
  const img = image(5, 3, (x) => [x * 10, 0, 0]);
  pixelateRGBA(img.data, 5, 3, 2);
  assert.equal(px(img, 4, 2)[0], 40); // last 1-wide tile keeps its own value
});

test('effectParams: scale with object size and strength', () => {
  const small = effectParams('blur', 50, 50).sigma;
  const big = effectParams('blur', 50, 500).sigma;
  assert.ok(big > small * 5);
  assert.ok(effectParams('blur', 90, 100).sigma > effectParams('blur', 10, 100).sigma);
  assert.ok(effectParams('pixelate', 90, 100).block > effectParams('pixelate', 10, 100).block);
});

test('applyEffects solid: fills inside the mask only, honours alpha', () => {
  const img = image(6, 4, () => [200, 200, 200]);
  const mask = { x0: 1, y0: 1, w: 2, h: 2, alpha: new Uint8ClampedArray([255, 0, 128, 255]) };
  applyEffects(img, [{ mask, size: 2 }], { style: 'solid', color: [0, 0, 0] });
  assert.deepEqual(px(img, 1, 1), [0, 0, 0, 255]); // alpha 255
  assert.deepEqual(px(img, 2, 1), [200, 200, 200, 255]); // alpha 0
  assert.equal(px(img, 1, 2)[0], 100); // alpha 128: 200 * (1 - 128/255)
  assert.deepEqual(px(img, 0, 0), [200, 200, 200, 255]); // outside
  assert.deepEqual(px(img, 5, 3), [200, 200, 200, 255]);
});

test('applyEffects blur: pixels outside the mask are byte-identical', () => {
  const img = image(40, 30, (x, y) => ((x + y) % 2 ? [255, 255, 255] : [0, 0, 0]));
  const original = new Uint8ClampedArray(img.data);
  const mask = { x0: 10, y0: 8, w: 15, h: 12, alpha: new Uint8ClampedArray(15 * 12).fill(255) };
  applyEffects(img, [{ mask, size: 15 }], { style: 'blur', strength: 60 });
  let changedInside = 0;
  for (let y = 0; y < 30; y++)
    for (let x = 0; x < 40; x++) {
      const inside = x >= 10 && x < 25 && y >= 8 && y < 20;
      const same = px(img, x, y).every((v, k) => v === original[(y * 40 + x) * 4 + k]);
      if (!inside) assert.ok(same, `pixel ${x},${y} outside mask changed`);
      else if (!same) changedInside++;
    }
  assert.ok(changedInside > 100);
});

test('applyEffects blur: a checkerboard turns into mid-grey', () => {
  const img = image(60, 60, (x, y) => ((x + y) % 2 ? [255, 255, 255] : [0, 0, 0]));
  applyEffects(img, [{ mask: fullMask(60, 60), size: 60 }], { style: 'blur', strength: 50 });
  const v = px(img, 30, 30)[0];
  assert.ok(v > 110 && v < 145, `centre value ${v}`);
});

test('applyEffects pixelate: coarse tiles inside the mask', () => {
  const img = image(32, 32, (x) => [x * 8, 0, 0]);
  applyEffects(img, [{ mask: fullMask(32, 32), size: 32 }], { style: 'pixelate', strength: 100 });
  const distinct = new Set();
  for (let x = 0; x < 32; x++) distinct.add(px(img, x, 5)[0]);
  assert.ok(distinct.size <= 8, `distinct columns ${distinct.size}`);
});

test('applyEffects ignores null masks', () => {
  const img = image(4, 4, () => [1, 2, 3]);
  const before = new Uint8ClampedArray(img.data);
  applyEffects(img, [{ mask: null, size: 4 }]);
  assert.deepEqual(img.data, before);
});
