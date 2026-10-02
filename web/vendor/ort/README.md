# ONNX Runtime Web 1.30.0 (WASM build)

Vendored copy of [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) 1.30.0 (MIT, © Microsoft).

| File | Role |
| --- | --- |
| `ort.wasm.min.mjs` | JavaScript API (CPU / WASM execution provider only) |
| `ort-wasm-simd-threaded.mjs` | WebAssembly loader |
| `ort-wasm-simd-threaded.wasm` | The runtime itself |

Update with `npm run vendor`, then bump `CACHE` in `web/sw.js`.
The app runs it single-threaded (`numThreads = 1`) because static hosts cannot send the
COOP/COEP headers that multi-threaded WebAssembly requires.
