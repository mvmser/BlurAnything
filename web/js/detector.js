import { INPUT_SIZE, LETTERBOX_GRAY, letterbox, rgbaToChw } from './postprocess.js';
import { MODELS, LIMITS } from './config.js';

// Draws the image into a 640x640 canvas (gray padding) and converts it to a CHW float tensor.
export function prepareInput(source, width, height) {
  const lb = letterbox(width, height, INPUT_SIZE);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = `rgb(${LETTERBOX_GRAY},${LETTERBOX_GRAY},${LETTERBOX_GRAY})`;
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, width, height, lb.padX, lb.padY, lb.drawW, lb.drawH);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  return { input: rgbaToChw(data), lb };
}

// Talks to the inference worker. The worker handles messages one at a time,
// so a detect() sent after a load() always runs on that model.
export class Detector {
  constructor() {
    this.worker = null;
    this.calls = new Map();
    this.nextId = 1;
    this.onProgress = null; // (loaded, total) while a model downloads
  }

  start() {
    const worker = new Worker(new URL('./detector.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') return this.onProgress?.(data.loaded, data.total);
      const call = this.calls.get(data.id);
      this.calls.delete(data.id);
      if (data.type === 'error') call.reject(new Error(data.message));
      else call.resolve(data);
    };
    worker.onerror = (e) => {
      for (const call of this.calls.values()) call.reject(new Error(e.message || 'worker failed'));
      this.calls.clear();
      this.worker = null; // next call starts a fresh worker
    };
    return worker;
  }

  send(message, transfer = []) {
    this.worker ??= this.start();
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.calls.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }

  load(modelKey) {
    return this.send({ type: 'load', model: MODELS[modelKey] });
  }

  async detect(source, width, height) {
    const { input, lb } = prepareInput(source, width, height);
    const opts = { conf: LIMITS.minConf, iou: LIMITS.iou, maxDet: LIMITS.maxDetections };
    const res = await this.send({ type: 'detect', input, lb, opts }, [input.buffer]);
    return { detections: res.detections, lb, ms: res.ms };
  }
}
