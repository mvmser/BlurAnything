// Image loading helpers (DOM required).

import { LIMITS } from './config.js';

/**
 * Decode a File/Blob/URL into an opaque canvas.
 * - EXIF orientation is applied by the browser's decoder.
 * - Transparency is flattened onto white (exports are opaque).
 * - Huge images are downscaled to LIMITS.maxPixels (canvas memory limits).
 */
export async function loadImage(source, name = 'image') {
  const isUrl = typeof source === 'string';
  const url = isUrl ? source : URL.createObjectURL(source);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    try {
      await img.decode();
    } catch {
      throw new Error('decode');
    }
    let w = img.naturalWidth;
    let h = img.naturalHeight;
    if (!w || !h) throw new Error('decode');

    let downscaled = false;
    if (w * h > LIMITS.maxPixels) {
      const k = Math.sqrt(LIMITS.maxPixels / (w * h));
      w = Math.max(1, Math.floor(w * k));
      h = Math.max(1, Math.floor(h * k));
      downscaled = true;
    }
    const full = document.createElement('canvas');
    full.width = w;
    full.height = h;
    const ctx = full.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, w, h);

    const mime = (!isUrl && source.type) || (/\.jpe?g($|\?)/i.test(name) ? 'image/jpeg' : '');
    return { name: name.replace(/\.[^./\\]+$/, '') || 'image', mime, width: w, height: h, full, downscaled };
  } finally {
    if (!isUrl) URL.revokeObjectURL(url);
  }
}

/** Downscaled pixels used for the interactive preview (export uses the full canvas). */
export function buildPreview(image) {
  const scale = Math.min(1, LIMITS.previewMax / Math.max(image.width, image.height));
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image.full, 0, 0, width, height);
  return { width, height, base: ctx.getImageData(0, 0, width, height) };
}
