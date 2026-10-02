// Regenerates the README screenshots (docs/) and the social preview (web/og-image.png)
// from the real app:  npx playwright install chromium && node scripts/make-screenshots.mjs

import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = fileURLToPath(new URL('..', import.meta.url));
const docs = join(root, 'docs');
mkdirSync(docs, { recursive: true });

const PORT = 4188;
const server = spawn('node', [join(root, 'scripts', 'serve.mjs'), '--port', String(PORT)], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));
const url = `http://127.0.0.1:${PORT}/`;

const browser = await chromium.launch();
try {
  async function openEditor(page) {
    await page.goto(url);
    await page.click('[data-sample="bus"]');
    await page.waitForFunction(
      () => /\d+ found/.test(document.querySelector('#modelStatus')?.textContent || ''),
      null,
      {
        timeout: 90_000,
      },
    );
    await page.click('#chips .chip:has-text("person")');
    await page.mouse.move(2, 2);
    await page.waitForTimeout(600);
  }

  for (const scheme of ['light', 'dark']) {
    const ctx = await browser.newContext({
      viewport: { width: 1360, height: 880 },
      colorScheme: scheme,
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage();
    await openEditor(page);
    await page.screenshot({ path: join(docs, `screenshot-${scheme}.png`) });
    if (scheme === 'light') {
      // social preview: the blurred photo next to the pitch
      const photo = await page.evaluate(() => document.querySelector('#view').toDataURL('image/jpeg', 0.9));
      const card = await ctx.newPage();
      await card.setViewportSize({ width: 1200, height: 630 });
      await card.setContent(`<!doctype html><meta charset="utf-8"><style>
        *{box-sizing:border-box} body{margin:0;width:1200px;height:630px;display:flex;align-items:center;gap:56px;padding:0 72px;
        font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#12141c;
        background:radial-gradient(900px 500px at 15% 10%,#d9dbff,transparent),radial-gradient(700px 420px at 95% 90%,#eadcff,transparent),#f5f6fb}
        .logo{display:flex;align-items:center;gap:14px;font-weight:800;font-size:34px;letter-spacing:-.01em}
        .logo i{width:52px;height:52px;border-radius:15px;background:linear-gradient(135deg,#6366f1,#a855f7);display:grid;place-items:center}
        .logo i b{width:20px;height:20px;border-radius:50%;background:#fff;box-shadow:0 0 18px 8px rgba(255,255,255,.7)}
        h1{font-size:68px;line-height:1.02;letter-spacing:-.035em;margin:34px 0 22px}
        h1 span{background:linear-gradient(135deg,#4346d9,#8b5cf6);-webkit-background-clip:text;color:transparent}
        p{font-size:27px;line-height:1.4;color:#5f667b;margin:0}
        .pill{display:inline-block;margin-top:30px;padding:10px 20px;border-radius:99px;background:rgba(84,87,240,.12);color:#3f42d6;font-weight:700;font-size:22px}
        img{height:470px;border-radius:22px;box-shadow:0 30px 70px rgba(30,30,80,.35);flex:none}
      </style>
      <div style="flex:1"><div class="logo"><i><b></b></i>BlurAnything</div>
      <h1>Blur <span>anything</span><br>in your photos.</h1>
      <p>AI object detection and blurring that runs entirely in your browser.</p>
      <div class="pill">Nothing is uploaded</div></div>
      <img src="${photo}">`);
      await card.screenshot({ path: join(root, 'web', 'og-image.png') });
      await card.close();
    }
    await ctx.close();
  }

  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    colorScheme: 'light',
  });
  const mp = await mobile.newPage();
  await openEditor(mp);
  await mp.evaluate(() => window.scrollTo(0, 0));
  await mp.screenshot({ path: join(docs, 'screenshot-mobile.png') });
  await mobile.close();
  console.log('screenshots written to docs/ and web/og-image.png');
} finally {
  await browser.close();
  server.kill();
}
