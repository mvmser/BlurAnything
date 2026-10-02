// The site must work when hosted under a sub-path (e.g. https://example.com/BlurAnything/,
// a folder of a portfolio site). Nothing may assume it lives at "/".

import { test, expect } from '@playwright/test';

test('works under /BlurAnything/: assets, worker, models and service worker', async ({ page, baseURL }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const failed = [];
  page.on('response', (r) => r.status() >= 400 && failed.push(`${r.status()} ${r.url()}`));

  await page.goto(baseURL);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Blur anything');
  await page.click('[data-sample="bus"]');
  await expect(page.locator('#modelStatus')).toHaveText(/found/, { timeout: 90_000 });
  await page.click('#chips .chip:has-text("person")');
  await expect(page.locator('#regionList .row input:checked')).not.toHaveCount(0);

  // the service worker is scoped to the sub-path
  const scope = await page.evaluate(async () => (await navigator.serviceWorker.ready).scope);
  expect(scope).toBe(baseURL);

  expect(failed, 'no failing requests').toEqual([]);
  expect(errors, 'no console errors').toEqual([]);
});
