// Copy the ONNX Runtime Web files the app needs into web/vendor/.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "node_modules", "onnxruntime-web", "dist");
const vendor = join(root, "vendor");
const files = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

mkdirSync(vendor, { recursive: true });
for (const file of files) copyFileSync(join(dist, file), join(vendor, file));
console.log(`Copied ${files.length} ONNX Runtime files to vendor/`);
