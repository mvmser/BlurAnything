// Pixel operations that do not depend on the browser: they work on { data, width, height }
// objects (ImageData or a plain object) so they behave the same in every browser and in Node.
// Safari and older iOS do not implement CanvasRenderingContext2D.filter, so the blur is computed here.

/**
 * Box sizes whose successive application approximates a gaussian of the given sigma.
 */
export function boxSizesForGauss(sigma, passes = 3) {
  const ideal = Math.sqrt((12 * sigma * sigma) / passes + 1);
  let lower = Math.floor(ideal);
  if (lower % 2 === 0) lower--;
  const upper = lower + 2;
  const m = Math.round(
    (12 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes) / (-4 * lower - 4),
  );
  return Array.from({ length: passes }, (_, i) => (i < m ? lower : upper));
}

// One box blur pass over the RGB channels, with the edge pixels repeated outside the image.
function boxPass(src, dst, width, height, radius, horizontal) {
  const length = horizontal ? width : height;
  const lines = horizontal ? height : width;
  const step = horizontal ? 4 : width * 4;
  const lineStep = horizontal ? width * 4 : 4;
  const divisor = 2 * radius + 1;

  for (let line = 0; line < lines; line++) {
    const base = line * lineStep;
    for (let c = 0; c < 3; c++) {
      let sum = src[base + c] * radius;
      for (let k = 0; k <= radius; k++) sum += src[base + Math.min(k, length - 1) * step + c];
      for (let i = 0; i < length; i++) {
        dst[base + i * step + c] = (sum / divisor + 0.5) | 0;
        sum += src[base + Math.min(i + radius + 1, length - 1) * step + c];
        sum -= src[base + Math.max(i - radius, 0) * step + c];
      }
    }
  }
}

/**
 * Gaussian blur, in place, of the RGB channels of an RGBA image. Alpha is left untouched.
 */
export function gaussianBlurRGBA(image, sigma) {
  if (!(sigma > 0.5)) return image;
  const { data, width, height } = image;
  let src = data;
  let dst = new Uint8ClampedArray(data.length);
  dst.set(data); // keeps alpha

  for (const size of boxSizesForGauss(sigma)) {
    const radius = (size - 1) / 2;
    boxPass(src, dst, width, height, radius, true);
    [src, dst] = [dst, src];
    boxPass(src, dst, width, height, radius, false);
    [src, dst] = [dst, src];
  }
  if (src !== data) data.set(src);
  return image;
}

/**
 * Pixelate, in place, an RGBA image: every block x block cell becomes its average RGB colour.
 */
export function pixelateRGBA(image, block) {
  const size = Math.max(1, Math.round(block));
  if (size <= 1) return image;
  const { data, width, height } = image;

  for (let y0 = 0; y0 < height; y0 += size) {
    const y1 = Math.min(y0 + size, height);
    for (let x0 = 0; x0 < width; x0 += size) {
      const x1 = Math.min(x0 + size, width);
      let r = 0, g = 0, b = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * width + x) * 4;
          r += data[i]; g += data[i + 1]; b += data[i + 2];
        }
      }
      const n = (x1 - x0) * (y1 - y0);
      r = Math.round(r / n); g = Math.round(g / n); b = Math.round(b / n);
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * width + x) * 4;
          data[i] = r; data[i + 1] = g; data[i + 2] = b;
        }
      }
    }
  }
  return image;
}
