// End-to-end tests: real browser, real ONNX models, real inference.

import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const reference = JSON.parse(
  readFileSync(fileURLToPath(new URL('../fixtures/bus.reference.json', import.meta.url)), 'utf8'),
);
const fixture = (name) => fileURLToPath(new URL(`../../web/samples/${name}`, import.meta.url));

/** Wait until detection has finished for the current image. */
const detected = (page) => expect(page.locator('#modelStatus')).toHaveText(/\d+ (found|trouvé)/, { timeout: 90_000 });

/** RGBA pixels of the preview canvas inside a box given as fractions of the image. */
async function previewPixels(page, [fx1, fy1, fx2, fy2]) {
  return page.evaluate(
    ([a, b, c, d]) => {
      const canvas = document.querySelector('#view');
      const x = Math.round(canvas.width * a);
      const y = Math.round(canvas.height * b);
      const width = Math.max(1, Math.round(canvas.width * (c - a)));
      const height = Math.max(1, Math.round(canvas.height * (d - b)));
      return { width, height, data: Array.from(canvas.getContext('2d').getImageData(x, y, width, height).data) };
    },
    [fx1, fy1, fx2, fy2],
  );
}
/** Mean absolute RGB difference between two pixel regions. */
const meanAbsDiff = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    sum +=
      Math.abs(a.data[i] - b.data[i]) +
      Math.abs(a.data[i + 1] - b.data[i + 1]) +
      Math.abs(a.data[i + 2] - b.data[i + 2]);
  }
  return sum / ((a.data.length / 4) * 3);
};
/** Local contrast (mean |horizontal gradient| of the red channel): drops when a region is blurred. */
const sharpness = ({ data, width }) => {
  let sum = 0;
  let n = 0;
  for (let i = 0; i + 4 < data.length; i += 4) {
    if ((i / 4) % width === width - 1) continue;
    sum += Math.abs(data[i] - data[i + 4]);
    n++;
  }
  return sum / n;
};
const diffFrom = (page, before, area) => async () => meanAbsDiff(before, await previewPixels(page, area));

let pageErrors = [];
test.beforeEach(async ({ page }) => {
  pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && pageErrors.push(m.text()));
  await page.goto('/');
});
test.afterEach(() => {
  expect(pageErrors, 'no uncaught errors or console errors').toEqual([]);
});

test('landing page shows the privacy promise and samples', async ({ page }) => {
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Blur anything');
  await expect(page.locator('.badge')).toContainText('nothing is uploaded');
  await expect(page.locator('#sampleList .sample')).toHaveCount(4);
  await expect(page.locator('#editor')).toBeHidden();
});

test('detects the objects of the street sample like the reference implementation', async ({ page }) => {
  await page.click('[data-sample="bus"]');
  await detected(page);

  // The reference was produced by Ultralytics + ONNX Runtime on the same model.
  const stable = reference.fast.detections.filter((d) => d.score >= 0.35);
  const names = await page.locator('#regionList .row .name').allTextContents();
  for (const d of stable) {
    expect(
      names.some((n) => n.startsWith(d.name)),
      `${d.name} should be listed in ${names}`,
    ).toBe(true);
  }
  const persons = names.filter((n) => n.startsWith('person')).length;
  expect(persons).toBeGreaterThanOrEqual(stable.filter((d) => d.name === 'person').length);
  await expect(page.locator('.chip', { hasText: 'bus' })).toBeVisible();

  // Overlay boxes sit where the reference boxes are (±2% of the image).
  const [W, H] = reference.fast.size;
  const boxes = await page.$$eval('#overlay .region', (els) =>
    els.map((e) => ({
      x: parseFloat(e.style.left),
      y: parseFloat(e.style.top),
      w: parseFloat(e.style.width),
      h: parseFloat(e.style.height),
    })),
  );
  const bus = reference.fast.detections.find((d) => d.name === 'bus');
  expect(
    boxes.some(
      (b) =>
        Math.abs(b.x - (bus.box[0] / W) * 100) < 2 &&
        Math.abs(b.y - (bus.box[1] / H) * 100) < 2 &&
        Math.abs(b.w - ((bus.box[2] - bus.box[0]) / W) * 100) < 2 &&
        Math.abs(b.h - ((bus.box[3] - bus.box[1]) / H) * 100) < 2,
    ),
    'a region matches the reference bus box',
  ).toBe(true);
});

test('selecting a class blurs exactly that area; "hold to compare" restores the original', async ({ page }) => {
  await page.click('[data-sample="bus"]');
  await detected(page);

  const [W, H] = reference.fast.size;
  const person = reference.fast.detections.find((d) => d.name === 'person' && d.score > 0.8);
  // torso/legs of a reference person, and a patch of building facade far from every object
  const inside = [
    person.box[0] / W + 0.02,
    person.box[1] / H + 0.15,
    person.box[2] / W - 0.02,
    person.box[1] / H + 0.4,
  ];
  const outside = [0.35, 0.02, 0.65, 0.12];

  const beforeInside = await previewPixels(page, inside);
  const beforeOutside = await previewPixels(page, outside);
  const insideDiff = diffFrom(page, beforeInside, inside);

  await page.click('#chips .chip:has-text("person")');
  await expect(page.locator('#chips .chip:has-text("person")')).toHaveAttribute('aria-pressed', 'true');

  await expect.poll(insideDiff).toBeGreaterThan(3);
  expect(sharpness(await previewPixels(page, inside))).toBeLessThan(sharpness(beforeInside) * 0.6);
  expect(await diffFrom(page, beforeOutside, outside)()).toBeLessThan(0.5);

  // hold to compare
  const box = await page.locator('#compareBtn').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect.poll(insideDiff).toBeLessThan(0.5);
  await page.mouse.up();
  await expect.poll(insideDiff).toBeGreaterThan(3);

  // deselect everything -> original again
  await page.click('#selectNone');
  await expect.poll(insideDiff).toBeLessThan(0.5);
});

test('clicking an object on the image toggles it; the checkbox follows', async ({ page }) => {
  await page.click('[data-sample="coffee"]');
  await detected(page);
  const cup = page.locator('#overlay .region', { has: page.locator('.region-label', { hasText: /^cup$/ }) });
  await expect(cup).toHaveCount(1);
  await cup.click();
  await expect(cup).toHaveClass(/is-selected/);
  await expect(page.locator('#regionList .row', { hasText: 'cup' }).locator('input')).toBeChecked();
  await cup.click();
  await expect(cup).not.toHaveClass(/is-selected/);
});

test('all three effects change the pixels differently', async ({ page }) => {
  await page.click('[data-sample="astronaut"]');
  await detected(page);
  const area = [0.2, 0.25, 0.7, 0.9];
  const original = await previewPixels(page, area);
  await page.click('#selectAll');

  const results = {};
  for (const style of ['blur', 'pixelate', 'solid']) {
    await page.click(`label.seg-item:has(input[value="${style}"])`);
    await expect.poll(diffFrom(page, original, area)).toBeGreaterThan(5);
    results[style] = await previewPixels(page, area);
  }
  expect(meanAbsDiff(results.blur, results.pixelate)).toBeGreaterThan(1);
  expect(meanAbsDiff(results.pixelate, results.solid)).toBeGreaterThan(1);
  // "cover" is opaque black inside the mask: the centre of the person is pure black
  const centre = await previewPixels(page, [0.45, 0.55, 0.5, 0.6]);
  expect(Math.max(...centre.data.filter((_, i) => i % 4 !== 3))).toBeLessThan(8);
});

test('draw a custom rectangle: it is listed, blurred, and removable', async ({ page }) => {
  await page.click('[data-sample="cat"]');
  await detected(page);
  const stage = await page.locator('#stageInner').boundingBox();
  const region = [0.05, 0.05, 0.3, 0.3];
  const before = await previewPixels(page, region);

  await page.click('#toolRect');
  await expect(page.locator('#hint')).toContainText('Drag on the image');
  await page.mouse.move(stage.x + stage.width * 0.05, stage.y + stage.height * 0.05);
  await page.mouse.down();
  await page.mouse.move(stage.x + stage.width * 0.3, stage.y + stage.height * 0.3, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.press('Escape');

  await expect(page.locator('#regionList .row', { hasText: 'Area 1' })).toBeVisible();
  await expect.poll(diffFrom(page, before, region)).toBeGreaterThan(2);

  await page.locator('#regionList [data-remove]').click();
  await expect(page.locator('#regionList .row', { hasText: 'Area 1' })).toHaveCount(0);
  await expect.poll(diffFrom(page, before, region)).toBeLessThan(0.5);
});

test('confidence slider hides low-confidence detections', async ({ page }) => {
  await page.click('[data-sample="bus"]');
  await detected(page);
  const before = await page.locator('#regionList .row').count();
  await page.locator('#conf').fill('85');
  const after = await page.locator('#regionList .row').count();
  expect(after).toBeLessThan(before);
  await expect(page.locator('#confOut')).toHaveText('85%');
});

test('the accurate model can be selected and detects the same scene', async ({ page }) => {
  await page.click('[data-sample="bus"]');
  await detected(page);
  await page.click('label.seg-item:has(input[value="accurate"])');
  await expect(page.locator('#modelStatus')).toHaveText(/found in/, { timeout: 90_000 });
  const names = await page.locator('#regionList .row .name').allTextContents();
  expect(names.some((n) => n.startsWith('bus'))).toBe(true);
  expect(names.filter((n) => n.startsWith('person')).length).toBeGreaterThanOrEqual(3);
});

test('download keeps the original size and carries no metadata', async ({ page }) => {
  await page.click('[data-sample="astronaut"]');
  await detected(page);
  await page.click('#selectAll');
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#downloadBtn')]);
  expect(download.suggestedFilename()).toBe('astronaut-blurred.jpg');
  const path = await download.path();
  const bytes = readFileSync(path);
  expect(bytes.subarray(0, 3).toString('hex')).toBe('ffd8ff'); // JPEG
  expect(bytes.includes(Buffer.from('Exif'))).toBe(false); // no EXIF segment
  // Decode in the page to check dimensions (512x512 sample).
  const dims = await page.evaluate(async (b64) => {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([raw], { type: 'image/jpeg' }));
    return [bitmap.width, bitmap.height];
  }, bytes.toString('base64'));
  expect(dims).toEqual([512, 512]);
});

test('opening a non-image file shows an error and stays on the landing page', async ({ page }) => {
  await page.setInputFiles('#fileInput', { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') });
  await expect(page.locator('#toast')).toContainText('image');
  await expect(page.locator('#editor')).toBeHidden();
});

test('an image with nothing to detect shows the empty state', async ({ page }) => {
  // plain gradient built in the page
  const dataUrl = await page.evaluate(() => {
    const c = document.createElement('canvas');
    c.width = 320;
    c.height = 240;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 320, 240);
    grad.addColorStop(0, '#223');
    grad.addColorStop(1, '#eef');
    g.fillStyle = grad;
    g.fillRect(0, 0, 320, 240);
    return c.toDataURL('image/png');
  });
  await page.setInputFiles('#fileInput', {
    name: 'gradient.png',
    mimeType: 'image/png',
    buffer: Buffer.from(dataUrl.split(',')[1], 'base64'),
  });
  await detected(page);
  await expect(page.locator('#emptyObjects')).toBeVisible();
});

test('language switch to French translates the interface and the class names', async ({ page }) => {
  await page.click('#langBtn');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Floutez');
  await page.click('[data-sample="bus"]');
  await detected(page);
  await expect(page.locator('.chip', { hasText: 'personne' })).toBeVisible();
  await expect(page.locator('#downloadBtn')).toContainText('Télécharger');
  expect(await page.evaluate(() => document.documentElement.lang)).toBe('fr');
});

test('works offline once loaded (service worker + cached model)', async ({ page, context }) => {
  // First visit: let the app prefetch the model and the service worker precache the shell.
  await page.click('[data-sample="coffee"]');
  await detected(page);
  await page.evaluate(() => navigator.serviceWorker.ready);
  await expect
    .poll(() => page.evaluate(async () => (await caches.keys()).sort().join(',')), { timeout: 30_000 })
    .toContain('blur-anything-models-v1');
  await page.reload(); // now controlled by the service worker
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Blur anything');
  await page.click('[data-sample="astronaut"]');
  await detected(page);
  await expect(page.locator('#regionList .row .name').first()).toHaveText('person');
  await context.setOffline(false);
});

test('privacy: a full session never talks to another origin and never sends data', async ({
  page,
  context,
  baseURL,
}) => {
  // Context level: also sees the requests of the inference worker and of the service worker.
  const requests = [];
  context.on('request', (r) => requests.push({ url: r.url(), method: r.method(), hasBody: Boolean(r.postData()) }));

  await page.click('[data-sample="bus"]');
  await detected(page);
  await page.click('#selectAll');
  await page.click('label.seg-item:has(input[value="pixelate"])');
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#downloadBtn')]);
  await download.path();
  await page.click('#newBtn');
  await page.setInputFiles('#fileInput', fixture('cat.jpg'));
  await detected(page);

  const origin = new URL(baseURL).origin;
  const external = requests.filter((r) => !/^(blob:|data:)/.test(r.url) && new URL(r.url).origin !== origin);
  expect(external, 'requests to other origins').toEqual([]);
  expect(
    requests.filter((r) => r.method !== 'GET' || r.hasBody),
    'non-GET requests',
  ).toEqual([]);
  // sanity: the listener really saw the model and runtime downloads made by the workers
  expect(requests.some((r) => r.url.includes('.onnx'))).toBe(true);
  expect(requests.some((r) => r.url.includes('.wasm'))).toBe(true);
});

test('the content security policy blocks every other origin', async ({ page }) => {
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute('content');
  expect(csp).toContain("connect-src 'self'");
  expect(csp).toContain("default-src 'self'");
  expect(csp).not.toMatch(/unsafe-inline|unsafe-eval(?!')|https?:/);
  // and the browser really enforces it
  const blocked = await page.evaluate(() =>
    fetch('https://example.com/collect', { method: 'POST', body: 'x' }).then(
      () => 'sent',
      () => 'blocked',
    ),
  );
  expect(blocked).toBe('blocked');
  pageErrors = pageErrors.filter((e) => !/Content Security Policy/i.test(e)); // that violation was the point
});

test('EXIF orientation is honoured (phone photos taken in portrait)', async ({ page }) => {
  // Stored as 120x80 (red | blue) with orientation 6: it must be shown as 80x120, red on top.
  await page.setInputFiles('#fileInput', fixture('../../tests/fixtures/exif-rotated.jpg'));
  await expect(page.locator('#editor')).toBeVisible();
  const info = await page.evaluate(() => {
    const canvas = document.querySelector('#view');
    const px = (x, y) => Array.from(canvas.getContext('2d').getImageData(x, y, 1, 1).data.slice(0, 3));
    return { size: [canvas.width, canvas.height], top: px(40, 10), bottom: px(40, 110) };
  });
  expect(info.size).toEqual([80, 120]);
  expect(info.top[0]).toBeGreaterThan(150); // red
  expect(info.bottom[2]).toBeGreaterThan(150); // blue
});

test('a corrupt image file shows a friendly error and keeps the landing page', async ({ page }) => {
  await page.setInputFiles('#fileInput', {
    name: 'broken.png',
    mimeType: 'image/png',
    buffer: Buffer.from('definitely not a png'),
  });
  await expect(page.locator('#toast')).toContainText("can't be read");
  await expect(page.locator('#editor')).toBeHidden();
  // Playwright's trace snapshotter (not the app) tries to read blob: URLs, which the CSP refuses.
  pageErrors = pageErrors.filter((e) => !/Failed to load resource|ERR_|Refused to connect to 'blob:/i.test(e));
});
