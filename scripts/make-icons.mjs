// Renders the PNG icons (PWA, Apple touch icon) from the logo with headless Chromium.
//   npx playwright install chromium && node scripts/make-icons.mjs

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const out = join(fileURLToPath(new URL('..', import.meta.url)), 'web', 'icons');
mkdirSync(out, { recursive: true });

const glyph = (size, { bleed }) => `
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 32 32">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient>
    <filter id="b" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="${bleed ? 1.5 : 2}"/></filter>
  </defs>
  <rect width="32" height="32" rx="${bleed ? 0 : 9}" fill="url(#g)"/>
  <g transform="translate(16 16) scale(${bleed ? 0.72 : 1}) translate(-16 -16)">
    <circle cx="16" cy="16" r="8" fill="#fff" opacity=".9" filter="url(#b)"/>
    <circle cx="16" cy="16" r="3.4" fill="#fff"/>
  </g>
</svg>`;

const icons = [
  { file: 'icon-192.png', size: 192, bleed: false, transparent: true },
  { file: 'icon-512.png', size: 512, bleed: false, transparent: true },
  { file: 'maskable-512.png', size: 512, bleed: true },
  { file: 'icon-180.png', size: 180, bleed: true }, // Apple touch icon (opaque, no rounded corners)
];

const browser = await chromium.launch();
const page = await browser.newPage();
for (const { file, size, bleed, transparent } of icons) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<body style="margin:0;background:transparent">${glyph(size, { bleed })}</body>`);
  await page.screenshot({ path: join(out, file), omitBackground: Boolean(transparent) });
  console.log('wrote', file);
}
await browser.close();
