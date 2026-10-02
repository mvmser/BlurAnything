// Main-thread side of the detector: image -> model input, and a promise API
// over the inference worker.

import { INPUT_SIZE, LETTERBOX_GRAY, letterbox, rgbaToChw } from './postprocess.js';
import { MODELS, MODEL_CACHE, LIMITS } from './config.js';

/** Letterbox `source` (canvas/image/bitmap, W x H) into the 640x640 CHW float tensor. */
export function prepareInput(source, width, height) {
  const lb = letterbox(width, height, INPUT_SIZE);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = `rgb(${LETTERBOX_GRAY},${LETTERBOX_GRAY},${LETTERBOX_GRAY})`;
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, width, height, lb.padX, lb.padY, lb.drawW, lb.drawH);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  return { input: rgbaToChw(data), lb };
}

export class Detector {
  constructor() {
    this.worker = null;
    this.pending = new Map();
    this.seq = 0;
    this.modelId = null; // model currently resident in the worker
    this.latest = null; // most recently requested load: { key, listeners, promise }
  }

  static isSupported() {
    return typeof Worker !== 'undefined' && typeof WebAssembly === 'object';
  }

  #ensureWorker() {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('./detector.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const msg = e.data;
      const req = this.pending.get(msg.id);
      if (!req) return;
      if (msg.type === 'progress') {
        req.onProgress?.(msg.loaded, msg.total);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === 'error') req.reject(new Error(msg.message));
      else req.resolve(msg);
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || 'Worker failed to start');
      for (const req of this.pending.values()) req.reject(err);
      this.pending.clear();
      this.worker = null; // allow a clean restart on the next call
      this.modelId = null;
      this.latest = null;
    };
    this.worker = worker;
    return worker;
  }

  #request(message, { transfer = [], onProgress } = {}) {
    const worker = this.#ensureWorker();
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      this.pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ ...message, id }, transfer);
    });
  }

  /**
   * Make sure `modelKey` ('fast' | 'accurate') is loaded. Calls made while the
   * same model is already being requested share that download (every caller
   * still gets progress). The worker handles requests in order, so only the
   * latest request may be reused.
   */
  load(modelKey, onProgress) {
    const model = MODELS[modelKey];
    const latest = this.latest;
    if (latest && latest.key === modelKey) {
      if (onProgress) latest.listeners.add(onProgress);
      return latest.promise;
    }
    if (!latest && this.worker && this.modelId === model.id) return Promise.resolve({ cached: true, ms: 0 });

    const listeners = new Set(onProgress ? [onProgress] : []);
    const entry = { key: modelKey, listeners, promise: null };
    const settle = () => {
      if (this.latest === entry) this.latest = null;
    };
    entry.promise = this.#request(
      { type: 'load', model, cacheName: MODEL_CACHE },
      { onProgress: (loaded, total) => listeners.forEach((fn) => fn(loaded, total)) },
    ).then(
      (res) => {
        settle();
        this.modelId = model.id;
        return res;
      },
      (err) => {
        settle();
        throw err;
      },
    );
    this.latest = entry;
    return entry.promise;
  }

  /**
   * Run detection on `source` (W x H). Resolves with
   * { detections: [{classId, score, box, logits}], lb, inferenceMs, totalMs }.
   */
  async detect(source, width, height, opts = {}) {
    const { input, lb } = prepareInput(source, width, height);
    const res = await this.#request(
      {
        type: 'detect',
        input,
        lb,
        opts: { conf: LIMITS.minConf, iou: LIMITS.iou, maxDet: LIMITS.maxDetections, ...opts },
      },
      { transfer: [input.buffer] },
    );
    return { detections: res.detections, lb, inferenceMs: res.inferenceMs, totalMs: res.totalMs };
  }
}
