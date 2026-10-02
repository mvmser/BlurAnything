// Blur, pixelate and cover for RGBA pixels ({ width, height, data }, same shape as ImageData).
// Plain functions, no DOM, so they also run in the node tests.

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export const STYLES = ['blur', 'pixelate', 'solid'];

// widths of the 3 box blurs that approximate a gaussian of this sigma
export function boxesForGauss(sigma, n = 3) {
  const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal);
  if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
  const m = Math.round(mIdeal);
  return Array.from({ length: n }, (_, i) => (i < m ? wl : wu));
}

// Box blur with a sliding window over `lines` lines of `len` samples, edge pixels repeated.
// `stride` is the step along a line, `lineStep` the step between lines. Only R, G, B are touched.
function boxBlurLines(src, dst, lines, len, lineStep, stride, r) {
  const inv = 1 / (2 * r + 1);
  const lastIdx = len - 1;
  for (let l = 0; l < lines; l++) {
    const base = l * lineStep;
    for (let c = 0; c < 3; c++) {
      const o = base + c;
      let acc = (r + 1) * src[o];
      for (let k = 1; k <= r; k++) acc += src[o + (k < lastIdx ? k : lastIdx) * stride];
      for (let i = 0; i < len; i++) {
        dst[o + i * stride] = acc * inv; // Uint8ClampedArray rounds to nearest
        const add = i + 1 + r;
        const sub = i - r;
        acc += src[o + (add < lastIdx ? add : lastIdx) * stride] - src[o + (sub > 0 ? sub : 0) * stride];
      }
    }
  }
}

// gaussian blur in place: three box blurs in each direction
export function gaussianBlurRGBA(data, w, h, sigma) {
  if (!(sigma >= 0.6) || w < 2 || h < 2) return;
  const sizes = boxesForGauss(sigma, 3);
  let a = data;
  let b = new Uint8ClampedArray(data); // a copy, so alpha is right in both buffers
  for (const s of sizes) {
    boxBlurLines(a, b, h, w, w * 4, 4, (s - 1) / 2);
    [a, b] = [b, a];
  }
  for (const s of sizes) {
    boxBlurLines(a, b, w, h, 4, w * 4, (s - 1) / 2);
    [a, b] = [b, a];
  }
  if (a !== data) data.set(a);
}

// every block x block tile becomes its average color
export function pixelateRGBA(data, w, h, block) {
  const bs = Math.max(2, Math.round(block));
  for (let by = 0; by < h; by += bs) {
    const bh = Math.min(bs, h - by);
    for (let bx = 0; bx < w; bx += bs) {
      const bw = Math.min(bs, w - bx);
      let r = 0;
      let g = 0;
      let b = 0;
      for (let y = 0; y < bh; y++) {
        let idx = ((by + y) * w + bx) * 4;
        for (let x = 0; x < bw; x++, idx += 4) {
          r += data[idx];
          g += data[idx + 1];
          b += data[idx + 2];
        }
      }
      const n = bw * bh;
      r = Math.round(r / n);
      g = Math.round(g / n);
      b = Math.round(b / n);
      for (let y = 0; y < bh; y++) {
        let idx = ((by + y) * w + bx) * 4;
        for (let x = 0; x < bw; x++, idx += 4) {
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
        }
      }
    }
  }
}

// Strength is relative to the size of the object, so a face and a bus get a similar look,
// and the preview matches the full resolution export.
export function effectParams(style, strength, size) {
  const t = clamp(strength, 0, 100) / 100;
  if (style === 'pixelate') return { block: Math.max(2, size / (28 - 24 * t)) }; // 28 -> 4 tiles across
  if (style === 'solid') return {};
  return { sigma: Math.max(1, size * (0.02 + 0.18 * t)) };
}

// Applies the effect to every { mask, size } target, in place. `strength` goes from 0 to 100.
export function applyEffects(image, targets, { style = 'blur', strength = 55, color = [0, 0, 0] } = {}) {
  const { width: W, height: H, data } = image;

  for (const { mask, size } of targets) {
    if (!mask || mask.w <= 0 || mask.h <= 0) continue;
    const p = effectParams(style, strength, size);

    // blur a bit more than the mask, so the edges don't get darker or brighter
    const pad = style === 'blur' ? Math.ceil(p.sigma * 2.5) : 0;
    const rx0 = Math.max(0, mask.x0 - pad);
    const ry0 = Math.max(0, mask.y0 - pad);
    const rx1 = Math.min(W, mask.x0 + mask.w + pad);
    const ry1 = Math.min(H, mask.y0 + mask.h + pad);
    const rw = rx1 - rx0;
    const rh = ry1 - ry0;

    const region = new Uint8ClampedArray(rw * rh * 4);
    for (let y = 0; y < rh; y++) {
      const s = ((ry0 + y) * W + rx0) * 4;
      region.set(data.subarray(s, s + rw * 4), y * rw * 4);
    }

    if (style === 'solid') {
      for (let i = 0; i < region.length; i += 4) {
        region[i] = color[0];
        region[i + 1] = color[1];
        region[i + 2] = color[2];
      }
    } else if (style === 'pixelate') {
      pixelateRGBA(region, rw, rh, p.block);
    } else {
      gaussianBlurRGBA(region, rw, rh, p.sigma);
    }

    const ox = mask.x0 - rx0;
    const oy = mask.y0 - ry0;
    for (let y = 0; y < mask.h; y++) {
      let ai = y * mask.w;
      let di = ((mask.y0 + y) * W + mask.x0) * 4;
      let ri = ((oy + y) * rw + ox) * 4;
      for (let x = 0; x < mask.w; x++, ai++, di += 4, ri += 4) {
        const a = mask.alpha[ai];
        if (a === 0) continue;
        if (a === 255) {
          data[di] = region[ri];
          data[di + 1] = region[ri + 1];
          data[di + 2] = region[ri + 2];
        } else {
          const k = a / 255;
          data[di] += (region[ri] - data[di]) * k;
          data[di + 1] += (region[ri + 1] - data[di + 1]) * k;
          data[di + 2] += (region[ri + 2] - data[di + 2]) * k;
        }
      }
    }
  }
  return image;
}
