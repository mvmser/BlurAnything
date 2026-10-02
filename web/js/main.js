// BlurAnything — application logic. Everything runs locally in the browser:
// decoding, YOLOv8-seg inference (worker), masks, blur and export.

import { MODELS, SAMPLES, LIMITS, DEFAULTS } from './config.js';
import { t, tn, setLang, getLang, detectLang, applyI18n, className, formatBytes } from './i18n.js';
import { Detector } from './detector.js';
import { rasterizeObjectMask, rasterizeShapeMask, boxSize } from './masks.js';
import { applyEffects, STYLES } from './effects.js';
import { loadImage, buildPreview } from './image.js';

const $ = (id) => document.getElementById(id);
const el = Object.fromEntries(
  [
    'landing',
    'editor',
    'dropzone',
    'fileInput',
    'sampleList',
    'stage',
    'stageInner',
    'view',
    'overlay',
    'draft',
    'busy',
    'busyText',
    'busyProgress',
    'busyBar',
    'hint',
    'toolRect',
    'toolEllipse',
    'compareBtn',
    'newBtn',
    'copyBtn',
    'shareBtn',
    'downloadBtn',
    'modelStatus',
    'modelFastSub',
    'modelAccurateSub',
    'objectCount',
    'selectAll',
    'selectNone',
    'chips',
    'regionList',
    'emptyObjects',
    'conf',
    'confOut',
    'strength',
    'strengthOut',
    'strengthField',
    'margin',
    'marginOut',
    'langBtn',
    'themeBtn',
    'toast',
    'dropOverlay',
  ].map((id) => [id, $(id)]),
);

// ---------------------------------------------------------------- storage --
const store = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode / quota: preferences just won't persist */
    }
  },
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(store.get('ba-settings') || '{}');
  } catch {
    /* ignore corrupt value */
  }
  const num = (v, lo, hi, fallback) => (Number.isFinite(v) ? clamp(v, lo, hi) : fallback);
  return {
    model: saved.model in MODELS ? saved.model : DEFAULTS.model,
    style: STYLES.includes(saved.style) ? saved.style : DEFAULTS.style,
    strength: num(saved.strength, 0, 100, DEFAULTS.strength),
    margin: num(saved.margin, 0, 100, DEFAULTS.margin),
    conf: num(saved.conf, LIMITS.minConf, 0.9, DEFAULTS.conf),
  };
}
const settings = loadSettings();
const saveSettings = () => store.set('ba-settings', JSON.stringify(settings));

// ------------------------------------------------------------------ state --
const state = {
  image: null, // { name, mime, width, height, full, preview, downscaled }
  lb: null, // letterbox info of the last detection run
  detections: [], // object regions from the model
  manual: [], // user-drawn regions
  selected: new Set(), // ids of regions to blur
  tool: null, // null | 'rect' | 'ellipse'
  hoverId: null,
  comparing: false,
  token: 0, // identifies the current analysis run (stale results are dropped)
  phase: 'idle', // idle | loading | detecting | ready | error
  info: null, // { ms }
  error: null,
};
let manualSeq = 0;
const maskCache = new Map();
const detector = new Detector();

const MANUAL_HUE = 258;
const hsl = (h, s, l) => `hsl(${Math.round(h)} ${s}% ${l}%)`;

// ----------------------------------------------------------------- helpers --
let toastTimer = 0;
function toast(message, isError = false) {
  el.toast.textContent = message;
  el.toast.classList.toggle('error', isError);
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.toast.hidden = true), isError ? 6000 : 3200);
}

let busyTimer = 0;
function setBusy(on, text = '', progress = null, immediate = false) {
  clearTimeout(busyTimer);
  const apply = () => {
    el.busy.hidden = false;
    el.busyText.textContent = text;
    el.busyProgress.hidden = progress == null;
    if (progress != null) el.busyBar.style.width = `${Math.round(clamp(progress, 0, 1) * 100)}%`;
  };
  if (!on) {
    el.busy.hidden = true;
    return;
  }
  if (immediate || !el.busy.hidden) apply();
  else busyTimer = setTimeout(apply, 120); // avoids a flash when the model is already cached
}

/** Regions currently shown (confidence filter applied), with display labels. */
function visibleRegions() {
  const dets = state.detections.filter((r) => r.score >= settings.conf).sort((a, b) => a.box[0] - b.box[0]);
  const counts = new Map();
  const seen = new Map();
  for (const r of dets) counts.set(r.classId, (counts.get(r.classId) || 0) + 1);
  for (const r of dets) {
    const n = (seen.get(r.classId) || 0) + 1;
    seen.set(r.classId, n);
    const base = className(r.classId);
    r.label = counts.get(r.classId) > 1 ? `${base} ${n}` : base;
  }
  state.manual.forEach((r, i) => (r.label = `${t('region.custom')} ${i + 1}`));
  return [...dets, ...state.manual];
}
const isSelected = (r) => state.selected.has(r.id);

// -------------------------------------------------------------- rendering --
function fitStage() {
  const img = state.image;
  if (!img || el.editor.hidden) return;
  const cs = getComputedStyle(el.stage);
  const availW = el.stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const maxH = Math.max(240, Math.min(window.innerHeight * 0.74, 920));
  const scale = Math.min(availW / img.width, maxH / img.height, 2);
  el.stageInner.style.width = `${Math.max(1, Math.round(img.width * scale))}px`;
  el.stageInner.style.height = `${Math.max(1, Math.round(img.height * scale))}px`;
}

function masksFor(regions, tw, th, { cache }) {
  const img = state.image;
  const targets = [];
  for (const r of regions) {
    const key = `${r.id}|${settings.margin}|${tw}`;
    let mask = cache ? maskCache.get(key) : undefined;
    if (mask === undefined) {
      mask =
        r.kind === 'object'
          ? rasterizeObjectMask(r.logits, state.lb, r.box, tw, th, { margin: settings.margin / 100 })
          : rasterizeShapeMask(r.shape, r.box, img.width, img.height, tw, th);
      if (cache) maskCache.set(key, mask);
    }
    if (mask) targets.push({ mask, size: boxSize(r.box) * (tw / img.width) });
  }
  return targets;
}

let rafId = 0;
function scheduleRender() {
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = 0;
    renderPreview();
  });
}

function renderPreview() {
  const img = state.image;
  if (!img) return;
  const { preview } = img;
  const ctx = el.view.getContext('2d');
  const chosen = visibleRegions().filter(isSelected);
  if (state.comparing || !chosen.length) {
    ctx.putImageData(preview.base, 0, 0);
    return;
  }
  const work = new ImageData(new Uint8ClampedArray(preview.base.data), preview.width, preview.height);
  const targets = masksFor(chosen, preview.width, preview.height, { cache: true });
  applyEffects(work, targets, { style: settings.style, strength: settings.strength });
  ctx.putImageData(work, 0, 0);
}

function renderOverlay() {
  const img = state.image;
  el.overlay.replaceChildren();
  if (!img) return;
  const regions = visibleRegions();
  const frag = document.createDocumentFragment();
  for (const r of regions) {
    const [x1, y1, x2, y2] = r.box;
    const hue = r.kind === 'object' ? r.hue : MANUAL_HUE;
    const b = document.createElement('button');
    b.type = 'button';
    b.tabIndex = -1;
    b.setAttribute('aria-hidden', 'true');
    b.dataset.id = r.id;
    b.className = `region${r.shape === 'ellipse' ? ' ellipse' : ''}${isSelected(r) ? ' is-selected' : ''}${
      state.hoverId === r.id ? ' is-hover' : ''
    }`;
    const area = ((x2 - x1) * (y2 - y1)) / (img.width * img.height);
    b.style.cssText =
      `left:${(x1 / img.width) * 100}%;top:${(y1 / img.height) * 100}%;` +
      `width:${((x2 - x1) / img.width) * 100}%;height:${((y2 - y1) / img.height) * 100}%;` +
      `--c:${hsl(hue, 85, 55)};--c-dark:${hsl(hue, 65, 34)};z-index:${10 + Math.round((1 - Math.min(1, area)) * 100)}`;
    const label = document.createElement('span');
    label.className = 'region-label';
    label.textContent = r.label;
    b.append(label);
    frag.append(b);
  }
  el.overlay.append(frag);
}

function renderPanel() {
  const regions = visibleRegions();
  const objects = regions.filter((r) => r.kind === 'object');

  // counts / empty state
  el.objectCount.textContent = state.phase === 'ready' || objects.length ? tn('objects.count', objects.length) : '';
  el.emptyObjects.hidden = !(state.phase === 'ready' && regions.length === 0);

  // class chips
  const groups = new Map();
  for (const r of objects) {
    const g = groups.get(r.classId) || { classId: r.classId, hue: r.hue, ids: [] };
    g.ids.push(r.id);
    groups.set(r.classId, g);
  }
  el.chips.replaceChildren(
    ...[...groups.values()].map((g) => {
      const allOn = g.ids.every((id) => state.selected.has(id));
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.setAttribute('aria-pressed', String(allOn));
      chip.dataset.ids = g.ids.join(',');
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.setProperty('--c', hsl(g.hue, 85, 55));
      chip.append(dot, `${className(g.classId)}${g.ids.length > 1 ? ` ×${g.ids.length}` : ''}`);
      return chip;
    }),
  );

  // list
  el.regionList.replaceChildren(
    ...regions.map((r) => {
      const li = document.createElement('li');
      const row = document.createElement('label');
      row.className = `row${state.hoverId === r.id ? ' is-hover' : ''}`;
      row.dataset.id = r.id;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = isSelected(r);
      cb.setAttribute(
        'aria-label',
        r.kind === 'object' ? t('a11y.region', { name: r.label, pct: Math.round(r.score * 100) }) : r.label,
      );
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.setProperty('--c', hsl(r.kind === 'object' ? r.hue : MANUAL_HUE, 85, 55));
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = r.label;
      row.append(cb, dot, name);
      if (r.kind === 'object') {
        const meta = document.createElement('span');
        meta.className = 'meta';
        meta.textContent = `${Math.round(r.score * 100)}%`;
        row.append(meta);
      } else {
        const rm = document.createElement('button');
        rm.type = 'button';
        rm.className = 'remove';
        rm.dataset.remove = r.id;
        rm.setAttribute('aria-label', `${t('region.remove')}: ${r.label}`);
        rm.title = t('region.remove');
        rm.innerHTML = '<svg class="i" aria-hidden="true"><use href="#i-x"/></svg>';
        row.append(rm);
      }
      li.append(row);
      return li;
    }),
  );
}

function renderStatus() {
  const box = el.modelStatus;
  if (state.phase === 'error') {
    const msg = document.createElement('span');
    msg.textContent = state.error;
    msg.style.color = 'var(--danger)';
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn xs';
    retry.textContent = t('action.retry');
    retry.style.marginLeft = '8px';
    retry.addEventListener('click', analyze);
    box.replaceChildren(msg, retry);
  } else if (state.phase === 'loading') box.textContent = t('status.model');
  else if (state.phase === 'detecting') box.textContent = t('status.detecting');
  else if (state.phase === 'ready' && state.info) {
    const n = state.detections.filter((r) => r.score >= settings.conf).length;
    box.textContent = t('status.done', { n, ms: state.info.ms });
  } else box.textContent = '';
}

function renderHint() {
  el.hint.textContent = state.tool ? t('hint.draw') : t('hint.default');
  el.stage.classList.toggle('drawing', Boolean(state.tool));
  el.toolRect.setAttribute('aria-pressed', String(state.tool === 'rect'));
  el.toolEllipse.setAttribute('aria-pressed', String(state.tool === 'ellipse'));
}

function renderControls() {
  el.conf.value = Math.round(settings.conf * 100);
  el.confOut.textContent = `${Math.round(settings.conf * 100)}%`;
  el.strength.value = settings.strength;
  el.strengthOut.textContent = `${Math.round(settings.strength)}%`;
  el.margin.value = settings.margin;
  el.marginOut.textContent = `${Math.round(settings.margin)}%`;
  el.strengthField.hidden = settings.style === 'solid';
  document.querySelectorAll('input[name="style"]').forEach((i) => (i.checked = i.value === settings.style));
  document.querySelectorAll('input[name="model"]').forEach((i) => (i.checked = i.value === settings.model));
  el.modelFastSub.textContent = t('model.fast.sub', { size: formatBytes(MODELS.fast.bytes) });
  el.modelAccurateSub.textContent = t('model.accurate.sub', { size: formatBytes(MODELS.accurate.bytes) });
  el.langBtn.textContent = getLang() === 'fr' ? 'EN' : 'FR';
}

function renderAll() {
  renderControls();
  renderHint();
  renderStatus();
  renderPanel();
  renderOverlay();
  scheduleRender();
}

function setHover(id) {
  if (state.hoverId === id) return;
  state.hoverId = id;
  el.overlay.querySelectorAll('.region').forEach((b) => b.classList.toggle('is-hover', b.dataset.id === id));
  el.regionList.querySelectorAll('.row').forEach((r) => r.classList.toggle('is-hover', r.dataset.id === id));
}

// -------------------------------------------------------------- selection --
// Update checkboxes / chips / overlay in place (rebuilding would drop keyboard focus).
function syncSelection() {
  const sel = state.selected;
  el.regionList.querySelectorAll('.row').forEach((row) => {
    row.querySelector('input').checked = sel.has(row.dataset.id);
  });
  el.chips.querySelectorAll('.chip').forEach((chip) => {
    chip.setAttribute('aria-pressed', String(chip.dataset.ids.split(',').every((id) => sel.has(id))));
  });
  el.overlay.querySelectorAll('.region').forEach((b) => b.classList.toggle('is-selected', sel.has(b.dataset.id)));
  scheduleRender();
}
function setSelected(ids, on) {
  for (const id of ids) {
    if (on) state.selected.add(id);
    else state.selected.delete(id);
  }
  syncSelection();
}
const toggleRegion = (id) => setSelected([id], !state.selected.has(id));

function removeManual(id) {
  state.manual = state.manual.filter((r) => r.id !== id);
  state.selected.delete(id);
  maskCache.forEach((_, key) => key.startsWith(`${id}|`) && maskCache.delete(key));
  renderAll();
}

// ----------------------------------------------------------- analysis flow --
async function analyze() {
  const img = state.image;
  if (!img) return;
  const token = ++state.token;
  state.detections = [];
  state.info = null;
  state.error = null;
  state.phase = 'loading';
  for (const id of [...state.selected]) if (!id.startsWith('m')) state.selected.delete(id);
  maskCache.clear();
  renderAll();

  let stage = 'model';
  try {
    setBusy(true, t('status.model'), 0);
    await detector.load(settings.model, (loaded, total) => {
      if (token !== state.token) return;
      const pct = Math.round((loaded / total) * 100);
      setBusy(true, t('status.modelPct', { pct }), loaded / total);
    });
    if (token !== state.token) return;

    stage = 'detect';
    state.phase = 'detecting';
    setBusy(true, t('status.detecting'));
    renderStatus();
    await nextFrame(); // let the spinner paint before the worker takes over
    const res = await detector.detect(img.full, img.width, img.height, { conf: LIMITS.minConf });
    if (token !== state.token) return;

    state.lb = res.lb;
    state.detections = res.detections.map((d, i) => ({
      id: `d${i + 1}`,
      kind: 'object',
      classId: d.classId,
      score: d.score,
      box: d.box,
      logits: d.logits,
      hue: (d.classId * 137.508) % 360,
    }));
    state.info = { ms: Math.round(res.inferenceMs) };
    state.phase = 'ready';
  } catch (err) {
    if (token !== state.token) return;
    state.phase = 'error';
    state.error = t(stage === 'model' ? 'err.model' : 'err.detect', { msg: String(err?.message || err) });
    toast(state.error, true);
  } finally {
    if (token === state.token) {
      setBusy(false);
      renderAll();
    }
  }
}

// ------------------------------------------------------------ image intake --
function showEditor(on) {
  el.landing.hidden = on;
  el.editor.hidden = !on;
  if (on) fitStage();
}

async function openImage(source, name) {
  let img;
  try {
    img = await loadImage(source, name);
    img.preview = buildPreview(img);
  } catch {
    toast(t('err.decode'), true);
    return;
  }
  state.token++; // drop any run in progress for the previous image
  state.image = img;
  state.lb = null;
  state.detections = [];
  state.manual = [];
  state.selected = new Set();
  state.tool = null;
  state.hoverId = null;
  state.phase = 'idle';
  maskCache.clear();
  el.view.width = img.preview.width;
  el.view.height = img.preview.height;
  el.view.getContext('2d').putImageData(img.preview.base, 0, 0);
  showEditor(true);
  window.scrollTo({ top: 0 });
  if (img.downscaled) toast(t('toast.downscaled', { w: img.width, h: img.height }));
  renderAll();
  analyze();
}

function handleFiles(files) {
  const file = [...(files || [])].find(
    (f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|bmp|avif|heic|heif)$/i.test(f.name),
  );
  if (!file) {
    toast(t('err.notImage'), true);
    return;
  }
  openImage(file, file.name);
}

function newImage() {
  state.token++; // drop any run in progress
  Object.assign(state, {
    image: null,
    lb: null,
    detections: [],
    manual: [],
    tool: null,
    hoverId: null,
    phase: 'idle',
    info: null,
    error: null,
  });
  state.selected = new Set();
  maskCache.clear();
  setBusy(false);
  renderAll();
  showEditor(false);
}

// ------------------------------------------------------------------ export --
async function exportBlob(type = 'image/png', quality = 0.92) {
  const img = state.image;
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img.full, 0, 0);
  const chosen = visibleRegions().filter(isSelected);
  if (chosen.length) {
    const data = ctx.getImageData(0, 0, img.width, img.height);
    applyEffects(data, masksFor(chosen, img.width, img.height, { cache: false }), {
      style: settings.style,
      strength: settings.strength,
    });
    ctx.putImageData(data, 0, 0);
  }
  // toBlob output carries no EXIF/GPS metadata.
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), type, quality),
  );
}

const outputFormat = () =>
  state.image.mime === 'image/jpeg' ? { type: 'image/jpeg', ext: 'jpg' } : { type: 'image/png', ext: 'png' };

async function withExportBusy(fn) {
  setBusy(true, t('status.exporting'), null, true);
  await nextFrame();
  try {
    return await fn();
  } catch (err) {
    toast(t('err.export', { msg: String(err?.message || err) }), true);
  } finally {
    setBusy(false);
  }
}

async function download() {
  if (!state.image) return;
  await withExportBusy(async () => {
    const { type, ext } = outputFormat();
    const blob = await exportBlob(type);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${state.image.name}-blurred.${ext}`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 15000);
    toast(t('toast.saved'));
  });
}

async function copyImage() {
  if (!state.image) return;
  try {
    // Passing a promise keeps the user-activation valid in Safari.
    const item = new ClipboardItem({ 'image/png': exportBlob('image/png') });
    await navigator.clipboard.write([item]);
    toast(t('toast.copied'));
  } catch {
    toast(t('toast.copyFailed'), true);
  }
}

async function shareImage() {
  if (!state.image) return;
  await withExportBusy(async () => {
    const { type, ext } = outputFormat();
    const blob = await exportBlob(type);
    const file = new File([blob], `${state.image.name}-blurred.${ext}`, { type });
    try {
      await navigator.share({ files: [file] });
    } catch (err) {
      if (err?.name !== 'AbortError') throw err;
    }
  });
}

// ----------------------------------------------------------- drawing tools --
function setTool(tool) {
  state.tool = state.tool === tool ? null : tool;
  renderHint();
}

let drag = null;
const toImagePoint = (e) => {
  const rect = el.stageInner.getBoundingClientRect();
  return {
    x: clamp((e.clientX - rect.left) / rect.width, 0, 1) * state.image.width,
    y: clamp((e.clientY - rect.top) / rect.height, 0, 1) * state.image.height,
    scale: rect.width / state.image.width,
  };
};

function placeBox(node, box) {
  const img = state.image;
  node.style.left = `${(box[0] / img.width) * 100}%`;
  node.style.top = `${(box[1] / img.height) * 100}%`;
  node.style.width = `${((box[2] - box[0]) / img.width) * 100}%`;
  node.style.height = `${((box[3] - box[1]) / img.height) * 100}%`;
}
const dragBox = () => [
  Math.min(drag.x0, drag.x1),
  Math.min(drag.y0, drag.y1),
  Math.max(drag.x0, drag.x1),
  Math.max(drag.y0, drag.y1),
];

el.overlay.addEventListener('pointerdown', (e) => {
  if (!state.tool || !state.image || e.button !== 0) return;
  e.preventDefault();
  const p = toImagePoint(e);
  drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, pointerId: e.pointerId };
  el.overlay.setPointerCapture(e.pointerId);
  el.draft.className = `draft${state.tool === 'ellipse' ? ' ellipse' : ''}`;
  el.draft.hidden = false;
  placeBox(el.draft, dragBox());
});
el.overlay.addEventListener('pointermove', (e) => {
  if (!drag || e.pointerId !== drag.pointerId) return;
  const p = toImagePoint(e);
  drag.x1 = p.x;
  drag.y1 = p.y;
  placeBox(el.draft, dragBox());
});
function endDrag(e, commit) {
  if (!drag || e.pointerId !== drag.pointerId) return;
  const box = dragBox();
  const scale = toImagePoint(e).scale;
  drag = null;
  el.draft.hidden = true;
  if (!commit) return;
  if ((box[2] - box[0]) * scale < 10 || (box[3] - box[1]) * scale < 10) return; // accidental click
  const region = { id: `m${++manualSeq}`, kind: 'manual', shape: state.tool, box };
  state.manual.push(region);
  state.selected.add(region.id);
  renderPanel();
  renderOverlay();
  scheduleRender();
}
el.overlay.addEventListener('pointerup', (e) => endDrag(e, true));
el.overlay.addEventListener('pointercancel', (e) => endDrag(e, false));

// ----------------------------------------------------------- event wiring --
el.overlay.addEventListener('click', (e) => {
  if (state.tool) return;
  const b = e.target.closest('.region');
  if (b) toggleRegion(b.dataset.id);
});
el.overlay.addEventListener('pointerover', (e) => setHover(e.target.closest('.region')?.dataset.id ?? null));
el.overlay.addEventListener('pointerleave', () => setHover(null));

el.regionList.addEventListener('change', (e) => {
  const row = e.target.closest('.row');
  if (row && e.target.matches('input[type="checkbox"]')) setSelected([row.dataset.id], e.target.checked);
});
el.regionList.addEventListener('click', (e) => {
  const rm = e.target.closest('[data-remove]');
  if (rm) {
    e.preventDefault();
    removeManual(rm.dataset.remove);
  }
});
el.regionList.addEventListener('pointerover', (e) => setHover(e.target.closest('.row')?.dataset.id ?? null));
el.regionList.addEventListener('pointerleave', () => setHover(null));

el.chips.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  const ids = chip.dataset.ids.split(',');
  setSelected(ids, !ids.every((id) => state.selected.has(id)));
});
el.selectAll.addEventListener('click', () =>
  setSelected(
    visibleRegions().map((r) => r.id),
    true,
  ),
);
el.selectNone.addEventListener('click', () =>
  setSelected(
    visibleRegions().map((r) => r.id),
    false,
  ),
);

el.conf.addEventListener('input', () => {
  settings.conf = Number(el.conf.value) / 100;
  saveSettings();
  el.confOut.textContent = `${el.conf.value}%`;
  renderStatus();
  renderPanel();
  renderOverlay();
  scheduleRender();
});
el.strength.addEventListener('input', () => {
  settings.strength = Number(el.strength.value);
  saveSettings();
  el.strengthOut.textContent = `${el.strength.value}%`;
  scheduleRender();
});
el.margin.addEventListener('input', () => {
  settings.margin = Number(el.margin.value);
  saveSettings();
  el.marginOut.textContent = `${el.margin.value}%`;
  scheduleRender();
});
document.querySelectorAll('input[name="style"]').forEach((input) =>
  input.addEventListener('change', () => {
    settings.style = input.value;
    saveSettings();
    renderControls();
    scheduleRender();
  }),
);
document.querySelectorAll('input[name="model"]').forEach((input) =>
  input.addEventListener('change', () => {
    if (settings.model === input.value) return;
    settings.model = input.value;
    saveSettings();
    analyze();
  }),
);

el.toolRect.addEventListener('click', () => setTool('rect'));
el.toolEllipse.addEventListener('click', () => setTool('ellipse'));
el.newBtn.addEventListener('click', newImage);
el.downloadBtn.addEventListener('click', download);
el.copyBtn.addEventListener('click', copyImage);
el.shareBtn.addEventListener('click', shareImage);

// hold-to-compare
const setComparing = (on) => {
  if (state.comparing === on) return;
  state.comparing = on;
  el.stage.classList.toggle('comparing', on);
  scheduleRender();
};
el.compareBtn.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  setComparing(true);
});
['pointerup', 'pointerleave', 'pointercancel', 'blur'].forEach((ev) =>
  el.compareBtn.addEventListener(ev, () => setComparing(false)),
);
el.compareBtn.addEventListener('keydown', (e) => {
  if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
    e.preventDefault();
    setComparing(true);
  }
});
el.compareBtn.addEventListener('keyup', () => setComparing(false));

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.tool) {
    state.tool = null;
    renderHint();
  }
});

// landing: dropzone, file input, samples, paste, drag & drop
el.dropzone.addEventListener('click', () => el.fileInput.click());
el.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    el.fileInput.click();
  }
});
el.fileInput.addEventListener('change', () => {
  handleFiles(el.fileInput.files);
  el.fileInput.value = '';
});
document.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'));
  if (files.length) {
    e.preventDefault();
    openImage(files[0], files[0].name || 'pasted-image');
  }
});

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  el.dropOverlay.hidden = false;
});
window.addEventListener('dragover', (e) => hasFiles(e) && e.preventDefault());
window.addEventListener('dragleave', (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) el.dropOverlay.hidden = true;
});
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  el.dropOverlay.hidden = true;
  handleFiles(e.dataTransfer.files);
});

function buildSamples() {
  el.sampleList.replaceChildren(
    ...SAMPLES.map((s) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'sample';
      b.dataset.sample = s.id;
      const img = document.createElement('img');
      img.src = s.url;
      img.alt = '';
      img.loading = 'lazy';
      const label = document.createElement('span');
      label.textContent = t(`sample.${s.id}`);
      b.append(img, label);
      b.addEventListener('click', () => openImage(s.url, `${s.id}.jpg`));
      return b;
    }),
  );
}

// language & theme
function switchLang(next) {
  setLang(next);
  store.set('ba-lang', next);
  applyI18n();
  buildSamples();
  renderAll();
}
el.langBtn.addEventListener('click', () => switchLang(getLang() === 'fr' ? 'en' : 'fr'));

const systemDark = window.matchMedia?.('(prefers-color-scheme: dark)');
el.themeBtn.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  store.set('ba-theme', next);
});
systemDark?.addEventListener?.('change', (e) => {
  if (!store.get('ba-theme')) document.documentElement.dataset.theme = e.matches ? 'dark' : 'light';
});

// layout
new ResizeObserver(fitStage).observe(el.stage);
window.addEventListener('resize', fitStage);

// ------------------------------------------------------------------- boot --
function boot() {
  setLang(detectLang(store.get('ba-lang'), navigator.language));
  applyI18n();
  buildSamples();
  renderControls();
  renderHint();

  if (!Detector.isSupported()) {
    el.dropzone.setAttribute('aria-disabled', 'true');
    toast(t('err.unsupported'), true);
    return;
  }
  const canShare = navigator.canShare?.({ files: [new File([''], 'a.png', { type: 'image/png' })] });
  el.shareBtn.hidden = !canShare;
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') el.copyBtn.hidden = true;

  // Offline support (installable PWA). Failure is harmless: the app works without it.
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
  }

  // Warm up the model while the visitor picks an image (unless data saving is on).
  const conn = navigator.connection;
  if (!conn?.saveData && !/^(slow-2g|2g|3g)$/.test(conn?.effectiveType || '')) {
    const warm = () => detector.load(settings.model).catch(() => {});
    if ('requestIdleCallback' in window) requestIdleCallback(warm, { timeout: 2500 });
    else setTimeout(warm, 800);
  }
}
boot();
