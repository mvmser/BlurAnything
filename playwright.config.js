import { defineConfig, devices } from '@playwright/test';

const PORT = 4173;
const SUBPATH_PORT = 4174; // same site mounted under /BlurAnything/ (GitHub Pages project site)

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false, // real model inference: keep CPU usage predictable
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', testIgnore: /subpath/, use: { ...devices['Desktop Chrome'] } },
    {
      name: 'subpath',
      testMatch: /subpath/,
      use: { ...devices['Desktop Chrome'], baseURL: `http://127.0.0.1:${SUBPATH_PORT}/BlurAnything/` },
    },
  ],
  webServer: [
    {
      command: `node scripts/serve.mjs --port ${PORT}`,
      url: `http://127.0.0.1:${PORT}`,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: `node scripts/serve.mjs --port ${SUBPATH_PORT} --base /BlurAnything/`,
      url: `http://127.0.0.1:${SUBPATH_PORT}/BlurAnything/`,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
