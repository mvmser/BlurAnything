// Inference worker: owns the ONNX Runtime Web session so that model download,
// graph compilation and inference never block the UI thread.
//
// Protocol (all replies carry the request `id`):
//   -> { type: 'load',   id, model: {id,name,url,bytes}, cacheName }
//   <- { type: 'progress', id, loaded, total }   (zero or more)
//   <- { type: 'loaded',   id, cached, ms }
//   -> { type: 'detect', id, input: Float32Array(1*3*640*640), lb, opts }
//   <- { type: 'result',   id, detections, inferenceMs, totalMs }
//   <- { type: 'error',    id, message }

import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { decodeSegmentation, INPUT_SIZE } from './postprocess.js';

const ortDir = new URL('../vendor/ort/', import.meta.url);
ort.env.wasm.wasmPaths = {
  mjs: new URL('ort-wasm-simd-threaded.mjs', ortDir).href,
  wasm: new URL('ort-wasm-simd-threaded.wasm', ortDir).href,
};
// Static hosts cannot send COOP/COEP headers, so SharedArrayBuffer (multi-threaded
// WASM) is unavailable: run single-threaded.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

let session = null;
let loadedModelId = null;
let queue = Promise.resolve();

// Messages are handled strictly one after the other (handlers await).
self.onmessage = (event) => {
  queue = queue.then(() => handle(event.data));
};

async function handle(msg) {
  try {
    if (msg.type === 'load') await onLoad(msg);
    else if (msg.type === 'detect') await onDetect(msg);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: err?.message || String(err) });
  }
}

async function onLoad({ id, model, cacheName }) {
  const t0 = performance.now();
  if (session && loadedModelId === model.id) {
    self.postMessage({ type: 'loaded', id, cached: true, ms: 0 });
    return;
  }
  if (session) {
    await session.release();
    session = null;
    loadedModelId = null;
  }
  const { bytes, fromCache } = await fetchModel(model, cacheName, (loaded, total) =>
    self.postMessage({ type: 'progress', id, loaded, total }),
  );
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  loadedModelId = model.id;
  self.postMessage({ type: 'loaded', id, cached: fromCache, ms: performance.now() - t0 });
}

/** Download a model with progress, using the Cache Storage for instant/offline reloads. */
async function fetchModel(model, cacheName, onProgress) {
  let cache = null;
  try {
    // Drop caches left behind by older model versions.
    for (const key of await caches.keys()) {
      if (key.startsWith('blur-anything-models-') && key !== cacheName) await caches.delete(key);
    }
    cache = await caches.open(cacheName);
    const hit = await cache.match(model.url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      if (buf.byteLength === model.bytes) {
        onProgress(model.bytes, model.bytes);
        return { bytes: buf, fromCache: true };
      }
      await cache.delete(model.url); // truncated or outdated entry
    }
  } catch {
    cache = null; // Cache Storage unavailable (private mode, insecure context...)
  }

  const res = await fetch(model.url);
  if (!res.ok) throw new Error(`HTTP ${res.status} while downloading ${model.url}`);
  const bytes = new Uint8Array(model.bytes);
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) throw new Error('Model file is larger than expected');
    bytes.set(value, loaded);
    loaded += value.length;
    onProgress(loaded, model.bytes);
  }
  if (loaded !== model.bytes) throw new Error(`Model file is incomplete (${loaded} of ${model.bytes} bytes)`);

  if (cache) {
    try {
      await cache.put(model.url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch {
      /* quota exceeded: the model simply won't be cached */
    }
  }
  return { bytes, fromCache: false };
}

async function onDetect({ id, input, lb, opts }) {
  if (!session) throw new Error('Model not loaded');
  const t0 = performance.now();
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
  const out = await session.run(feeds);
  const t1 = performance.now();

  let head = null;
  let protos = null;
  for (const name of session.outputNames) {
    const tensor = out[name];
    if (tensor.dims.length === 4) protos = tensor;
    else head = tensor;
  }
  if (!head || !protos) throw new Error('Unexpected model outputs (expected a YOLOv8-seg export)');

  const numMasks = protos.dims[1];
  const detections = decodeSegmentation(
    head.data,
    protos.data,
    {
      numAnchors: head.dims[2],
      numClasses: head.dims[1] - 4 - numMasks,
      numMasks,
      protoSize: protos.dims[2],
    },
    lb,
    opts,
  );
  const t2 = performance.now();
  self.postMessage(
    { type: 'result', id, detections, inferenceMs: t1 - t0, totalMs: t2 - t0 },
    detections.map((d) => d.logits.buffer),
  );
}
