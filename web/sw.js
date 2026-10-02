// Makes the app work offline.
// - code and pages: network first, the last copy when offline
// - wasm runtime, samples, icons: cache first
// - .onnx models: not handled here, the worker keeps them in its own cache
// Bump CACHE when vendor/ort, the samples or the icons change.

const CACHE = 'blur-anything-v1';

const PRECACHE = [
  './',
  'index.html',
  'manifest.webmanifest',
  'favicon.svg',
  'css/app.css',
  'js/config.js',
  'js/detector.js',
  'js/detector.worker.js',
  'js/effects.js',
  'js/i18n.js',
  'js/image.js',
  'js/main.js',
  'js/masks.js',
  'js/postprocess.js',
  'js/theme-init.js',
  'vendor/ort/ort.wasm.min.mjs',
  'vendor/ort/ort-wasm-simd-threaded.mjs',
  'vendor/ort/ort-wasm-simd-threaded.wasm',
  'samples/bus.jpg',
  'samples/astronaut.jpg',
  'samples/cat.jpg',
  'samples/coffee.jpg',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

// bypass the HTTP cache, so a new version never precaches old files
async function precache() {
  const cache = await caches.open(CACHE);
  await Promise.all(
    PRECACHE.map(async (path) => {
      const res = await fetch(new Request(path, { cache: 'reload' }));
      if (!res.ok) throw new Error(`precache failed: ${path} (${res.status})`);
      await cache.put(path, res);
    }),
  );
}

self.addEventListener('install', (event) => {
  event.waitUntil(precache().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith('blur-anything-') && k !== CACHE && !k.startsWith('blur-anything-models'))
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

const isHeavy = (url) => /\.(wasm|jpg|png)$/.test(url.pathname);

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) cache.put(request, res.clone());
  return res;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    // a new Request, so page navigations skip the HTTP cache too
    const res = await fetch(new Request(request.url, { cache: 'no-cache' }));
    if (res.ok) cache.put(request, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(request, { ignoreSearch: true });
    if (hit) return hit;
    if (request.mode === 'navigate') {
      const shell = await cache.match('index.html');
      if (shell) return shell;
    }
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || request.headers.has('range')) return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || url.pathname.endsWith('.onnx')) return;
  event.respondWith(isHeavy(url) ? cacheFirst(request) : networkFirst(request));
});
