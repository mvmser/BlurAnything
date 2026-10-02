// Copies the onnxruntime-web wasm runtime into web/vendor/ort (it is committed, no CDN).
//   npm run vendor              uses the version below
//   npm run vendor -- 1.31.0    another version

import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ORT_VERSION = '1.30.0';
const version = process.argv[2] || ORT_VERSION;
const files = ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];

const root = fileURLToPath(new URL('..', import.meta.url));
const dest = join(root, 'web', 'vendor', 'ort');
const tmp = mkdtempSync(join(tmpdir(), 'ort-vendor-'));

try {
  execFileSync(
    'npm',
    ['install', '--no-save', '--no-audit', '--no-fund', '--prefix', tmp, `onnxruntime-web@${version}`],
    {
      stdio: 'inherit',
    },
  );
  const pkg = join(tmp, 'node_modules', 'onnxruntime-web');
  mkdirSync(dest, { recursive: true });
  for (const f of files) {
    copyFileSync(join(pkg, 'dist', f), join(dest, f));
    console.log('copied', f);
  }
  const installed = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8')).version;
  console.log(`onnxruntime-web ${installed} vendored into web/vendor/ort`);
  console.log('Bump CACHE in web/sw.js so returning visitors get the new runtime.');
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
