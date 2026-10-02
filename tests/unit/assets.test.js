import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODELS, SAMPLES } from '../../web/js/config.js';

const web = fileURLToPath(new URL('../../web/', import.meta.url));
const rel = (p) => relative(web, p).split('\\').join('/');
const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)],
  );

const sw = readFileSync(join(web, 'sw.js'), 'utf8');
const precache = [...sw.match(/const PRECACHE = \[([\s\S]*?)\];/)[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);

test('declared model sizes match the files on disk (the worker checks them strictly)', () => {
  for (const model of Object.values(MODELS)) {
    const file = join(web, 'models', model.url.split('?')[0].split('/').pop());
    assert.ok(existsSync(file), `${model.name}: ${file} is missing`);
    assert.equal(statSync(file).size, model.bytes, `${model.name}: update MODELS.${model.id}.bytes in config.js`);
  }
});

test('model hashes in config.js match the files (they version the URLs)', () => {
  for (const model of Object.values(MODELS)) {
    const file = join(web, 'models', model.url.split('?')[0].split('/').pop());
    const sha = createHash('sha256').update(readFileSync(file)).digest('hex');
    assert.equal(sha, model.sha256, `${model.name}: re-exported? update its sha256 and bytes in config.js`);
    assert.ok(model.url.endsWith(`?v=${sha.slice(0, 10)}`));
  }
});

test('every sample image exists', () => {
  for (const s of SAMPLES) assert.ok(existsSync(join(web, 'samples', s.url.split('?')[0].split('/').pop())), s.id);
});

test('service worker: every precached file exists', () => {
  for (const f of precache) assert.ok(f === './' || existsSync(join(web, f)), `${f} is precached but missing`);
});

test('service worker: every code file is precached (offline support)', () => {
  const code = [...walk(join(web, 'js')), ...walk(join(web, 'css')), ...walk(join(web, 'vendor'))]
    .map(rel)
    .filter((f) => /\.(m?js|css|wasm)$/.test(f));
  for (const f of code) assert.ok(precache.includes(f), `${f} is not in the service worker PRECACHE list`);
});

test('manifest icons exist', () => {
  const manifest = JSON.parse(readFileSync(join(web, 'manifest.webmanifest'), 'utf8'));
  assert.ok(manifest.icons.length >= 2);
  for (const icon of manifest.icons) assert.ok(existsSync(join(web, icon.src)), icon.src);
});

test('index.html only references local files that exist', () => {
  const html = readFileSync(join(web, 'index.html'), 'utf8');
  const refs = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)]
    .map((m) => m[1])
    .filter((u) => !/^(https?:|data:|mailto:)/.test(u));
  for (const ref of refs) assert.ok(existsSync(join(web, ref)), `${ref} referenced by index.html`);
});

test('no file is bigger than 25 MiB', () => {
  for (const f of walk(web)) assert.ok(statSync(f).size < 25 * 1024 * 1024, `${rel(f)} is too large`);
});

test('the CSP in index.html only allows the own origin', () => {
  const html = readFileSync(join(web, 'index.html'), 'utf8');
  const csp = html.match(/http-equiv="Content-Security-Policy" content="([^"]+)"/)[1];
  assert.match(csp, /connect-src 'self'(;|$)/);
  assert.match(csp, /default-src 'self'/);
  assert.doesNotMatch(csp, /https?:|unsafe-inline/);
});
