// Minimal static file server for local development and the e2e tests.
//   node scripts/serve.mjs [--port 4173] [--dir web] [--base /BlurAnything/]
// Sends correct MIME types for .wasm/.mjs/.onnx and disables caching.
// --base mounts the site under a sub-path, to test hosting from a folder (e.g. a portfolio site).

import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : fallback;
}

const root = resolve(fileURLToPath(new URL('..', import.meta.url)), arg('dir', 'web'));
const port = Number(arg('port', process.env.PORT || 4173));
const base = `/${arg('base', '/').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (base !== '/') {
      if (url.pathname === base.slice(0, -1)) {
        res.writeHead(301, { location: base }).end();
        return;
      }
      if (!url.pathname.startsWith(base)) {
        res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found (outside base path)');
        return;
      }
    }
    let path = normalize(decodeURIComponent(url.pathname.slice(base.length - 1)));
    let file = join(root, path);
    if (file !== root && !file.startsWith(root + sep)) {
      res.writeHead(403).end('Forbidden');
      return;
    }
    let info = await stat(file).catch(() => null);
    if (info?.isDirectory()) {
      file = join(file, 'index.html');
      info = await stat(file).catch(() => null);
    }
    if (!info) {
      res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
      return;
    }
    res.writeHead(200, {
      'content-type': MIME[extname(file).toLowerCase()] || 'application/octet-stream',
      'content-length': info.size,
      'cache-control': 'no-cache',
    });
    if (req.method === 'HEAD') res.end();
    else createReadStream(file).pipe(res);
  } catch (err) {
    res.writeHead(500, { 'content-type': 'text/plain' }).end(String(err));
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`BlurAnything: http://127.0.0.1:${port}${base}  (serving ${root})`);
});
