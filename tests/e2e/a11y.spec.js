// Automated accessibility audit (axe-core) of the main screens, in both themes.

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const audit = async (page) => {
  const { violations } = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(
    violations.map(
      (v) =>
        `${v.id}: ${v.help} -> ${v.nodes
          .map((n) => n.target.join(' '))
          .slice(0, 3)
          .join(' | ')}`,
    ),
  ).toEqual([]);
};

for (const theme of ['light', 'dark']) {
  test.describe(`${theme} theme`, () => {
    test.use({ colorScheme: theme });

    test('landing page has no WCAG A/AA violations', async ({ page }) => {
      await page.goto('/');
      await audit(page);
    });

    test('editor has no WCAG A/AA violations', async ({ page }) => {
      await page.goto('/');
      await page.click('[data-sample="bus"]');
      await expect(page.locator('#modelStatus')).toHaveText(/found/, { timeout: 90_000 });
      await page.click('#chips .chip:has-text("person")');
      await audit(page);
    });
  });
}

test('the whole flow is keyboard accessible', async ({ page }) => {
  await page.goto('/');
  await page.focus('#dropzone');
  await expect(page.locator('#dropzone')).toBeFocused();
  // open a sample with the keyboard
  await page.focus('[data-sample="coffee"]');
  await page.keyboard.press('Enter');
  await expect(page.locator('#modelStatus')).toHaveText(/found/, { timeout: 90_000 });
  // toggle the first object with Space and make sure focus stays on the checkbox
  const first = page.locator('#regionList .row input').first();
  await first.focus();
  await page.keyboard.press('Space');
  await expect(first).toBeChecked();
  await expect(first).toBeFocused();
  // draw tool toggles with the keyboard and Escape leaves it
  await page.focus('#toolRect');
  await page.keyboard.press('Enter');
  await expect(page.locator('#toolRect')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Escape');
  await expect(page.locator('#toolRect')).toHaveAttribute('aria-pressed', 'false');
});
