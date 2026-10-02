import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { STRINGS, t, tn, setLang, className, detectLang, formatBytes } from '../../web/js/i18n.js';
import { COCO_CLASSES } from '../../web/js/config.js';

test('English and French define exactly the same keys', () => {
  const en = Object.keys(STRINGS.en).sort();
  const fr = Object.keys(STRINGS.fr).sort();
  assert.deepEqual(fr, en);
});

test('every data-i18n key used in index.html exists', () => {
  const html = readFileSync(fileURLToPath(new URL('../../web/index.html', import.meta.url)), 'utf8');
  const keys = [...html.matchAll(/data-i18n(?:-\w+)?="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length > 20);
  for (const k of keys) assert.ok(k in STRINGS.en, `missing key ${k}`);
});

test('every t("...") key used in the code exists', () => {
  const keys = new Set();
  for (const f of ['main.js', 'i18n.js']) {
    const src = readFileSync(fileURLToPath(new URL(`../../web/js/${f}`, import.meta.url)), 'utf8');
    for (const m of src.matchAll(/\bt\(\s*'([\w.]+)'/g)) keys.add(m[1]);
    for (const m of src.matchAll(/\btn\(\s*'([\w.]+)'/g)) (keys.add(`${m[1]}.one`), keys.add(`${m[1]}.other`));
  }
  for (const k of keys) assert.ok(k in STRINGS.en, `missing key ${k}`);
});

test('placeholders are the same in both languages', () => {
  const vars = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
  for (const k of Object.keys(STRINGS.en)) assert.deepEqual(vars(STRINGS.fr[k]), vars(STRINGS.en[k]), k);
});

test('80 COCO class names in both languages', () => {
  assert.equal(COCO_CLASSES.length, 80);
  setLang('fr');
  assert.equal(className(0), 'personne');
  assert.equal(className(5), 'bus');
  assert.equal(className(79), 'brosse à dents');
  setLang('en');
  assert.equal(className(0), 'person');
  assert.equal(className(79), 'toothbrush');
});

test('t / tn interpolate and pluralise', () => {
  setLang('en');
  assert.equal(tn('objects.count', 1), '1 object');
  assert.equal(tn('objects.count', 3), '3 objects');
  assert.equal(t('status.done', { n: 2, ms: 120 }), '2 found in 120 ms');
  assert.equal(t('does.not.exist'), 'does.not.exist');
  setLang('fr');
  assert.equal(tn('objects.count', 0), '0 objets');
  setLang('en');
});

test('detectLang prefers the saved choice, then the browser language', () => {
  assert.equal(detectLang('fr', 'en-US'), 'fr');
  assert.equal(detectLang(null, 'fr-FR'), 'fr');
  assert.equal(detectLang(null, 'de-DE'), 'en');
  assert.equal(detectLang('xx', 'fr-CA'), 'fr');
});

test('formatBytes', () => {
  setLang('en');
  assert.equal(formatBytes(7003966), '7.0 MB');
  assert.equal(formatBytes(23827387), '24 MB');
  setLang('fr');
  assert.equal(formatBytes(23827387), '24 Mo');
  setLang('en');
});
