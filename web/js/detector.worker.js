// Runs the model off the main thread.
// in:  { type: 'load', id, model } | { type: 'detect', id, input, lb, opts }
// out: { type: 'loaded' | 'result' | 'error', id, ... } and { type: 'progress', loaded, total }

import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { decodeSegmentation, INPUT_SIZE } from './postprocess.js';
import { MODEL_CACHE } from './config.js';

const ortDir = new URL('../vendor/ort/', import.meta.url);
ort.env.wasm.wasmPaths = {
  mjs: new URL('ort-wasm-simd-threaded.mjs', ortDir).href,
  wasm: new URL('ort-wasm-simd-threaded.wasm', ortDir).href,
};
// multi-threaded wasm needs COOP/COEP headers, which static hosts can't send
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

let session = null;
let loadedId = null;
let queue = Promise.resolve();

// one message at a time
self.onmessage = (e) => {
  queue = queue.then(() => handle(e.data));
};

async function handle(msg) {
  try {
    if (msg.type === 'load') await loadModel(msg);
    if (msg.type === 'detect') await detect(msg);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: err?.message || String(err) });
  }
}

async function loadModel({ id, model }) {
  if (loadedId === model.id) return self.postMessage({ type: 'loaded', id });

  await session?.release();
  session = null;
  loadedId = null;

  const bytes = await fetchModel(model);
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  loadedId = model.id;
  self.postMessage({ type: 'loaded', id });
}

// Downloads the model (with progress) and keeps it in the Cache Storage for next time.
async function fetchModel(model) {
  let cache = null;
  try {
    for (const key of await caches.keys()) {
      if (key.startsWith('blur-anything-models-') && key !== MODEL_CACHE) await caches.delete(key);
    }
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(model.url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      if (bytes.length === model.bytes) {
        self.postMessage({ type: 'progress', loaded: bytes.length, total: bytes.length });
        return bytes;
      }
      await cache.delete(model.url); // truncated copy
    }
  } catch {
    cache = null; // no Cache Storage (private mode...)
  }

  const res = await fetch(model.url);
  if (!res.ok) throw new Error(`${model.url}: HTTP ${res.status}`);
  const bytes = new Uint8Array(model.bytes);
  const reader = res.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) throw new Error('model file is bigger than expected');
    bytes.set(value, loaded);
    loaded += value.length;
    self.postMessage({ type: 'progress', loaded, total: bytes.length });
  }
  if (loaded !== bytes.length) throw new Error(`model file is incomplete (${loaded}/${bytes.length} bytes)`);

  try {
    await cache?.put(model.url, new Response(bytes));
  } catch {
    // quota exceeded, not a big deal
  }
  return bytes;
}

async function detect({ id, input, lb, opts }) {
  if (!session) throw new Error('model not loaded');
  const start = performance.now();
  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const ms = performance.now() - start;

  // the 4D output holds the mask prototypes, the 3D one the boxes
  const outputs = session.outputNames.map((name) => out[name]);
  const protos = outputs.find((o) => o.dims.length === 4);
  const head = outputs.find((o) => o.dims.length === 3);
  const numMasks = protos.dims[1];
  const dims = {
    numAnchors: head.dims[2],
    numClasses: head.dims[1] - 4 - numMasks,
    numMasks,
    protoSize: protos.dims[2],
  };
  const detections = decodeSegmentation(head.data, protos.data, dims, lb, opts);
  self.postMessage(
    { type: 'result', id, detections, ms },
    detections.map((d) => d.logits.buffer),
  );
}
