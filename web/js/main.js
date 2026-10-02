import { MODELS, SAMPLES, LIMITS, DEFAULTS } from './config.js';
import { t, tn, setLang, getLang, detectLang, applyI18n, className, formatBytes } from './i18n.js';
import { Detector } from './detector.js';
import { rasterizeObjectMask, rasterizeShapeMask, boxSize } from './masks.js';
import { applyEffects, STYLES } from './effects.js';
import { loadImage, buildPreview } from './image.js';

const $ = (id) => document.getElementById(id);
const on = (id, type, handler) => $(id).addEventListener(type, handler);
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

const view = $('view');
const stage = $('stage');
const stageInner = $('stageInner');
const overlay = $('overlay');
const regionList = $('regionList');
const chips = $('chips');

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

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
      // storage blocked, settings just won't stick
    }
  },
};

function readSettings() {
  const num = (v, min, max, fallback) => (Number.isFinite(v) ? clamp(v, min, max) : fallback);
  try {
    const saved = JSON.parse(store.get('ba-settings'));
    return {
      model: saved.model in MODELS ? saved.model : DEFAULTS.model,
      style: STYLES.includes(saved.style) ? saved.style : DEFAULTS.style,
      strength: num(saved.strength, 0, 100, DEFAULTS.strength),
      margin: num(saved.margin, 0, 100, DEFAULTS.margin),
      conf: num(saved.conf, LIMITS.minConf, 0.9, DEFAULTS.conf),
    };
  } catch {
    return { ...DEFAULTS };
  }
}
const settings = readSettings();
const saveSettings = () => store.set('ba-settings', JSON.stringify(settings));

const detector = new Detector();
const state = {
  image: null, // { name, mime, width, height, full, preview, downscaled }
  lb: null, // letterbox of the last detection, needed to place the masks
  detections: [],
  manual: [], // areas drawn by the user
  selected: new Set(), // ids of the regions to blur
  tool: null, // 'rect' | 'ellipse' | null
  hoverId: null,
  comparing: false,
  run: 0, // id of the current analysis, results of older ones are dropped
  phase: 'idle', // idle | loading | detecting | ready | error
  ms: 0,
  error: '',
};
let manualCount = 0;

const hue = (r) => (r.kind === 'object' ? Math.round(r.hue) : 258);
const dotColor = (r) => `hsl(${hue(r)} 85% 55%)`;

// detections above the confidence slider, then the manual areas
function visibleRegions() {
  const found = state.detections.filter((r) => r.score >= settings.conf).sort((a, b) => a.box[0] - b.box[0]);
  const total = {};
  const seen = {};
  for (const r of found) total[r.classId] = (total[r.classId] || 0) + 1;
  for (const r of found) {
    seen[r.classId] = (seen[r.classId] || 0) + 1;
    r.label = total[r.classId] > 1 ? `${className(r.classId)} ${seen[r.classId]}` : className(r.classId);
  }
  state.manual.forEach((r, i) => (r.label = `${t('region.custom')} ${i + 1}`));
  return [...found, ...state.manual];
}
const chosen = () => visibleRegions().filter((r) => state.selected.has(r.id));

function toast(message, isError = false) {
  const box = $('toast');
  box.textContent = message;
  box.classList.toggle('error', isError);
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (box.hidden = true), isError ? 6000 : 3200);
}

function showBusy(text, progress) {
  $('busy').hidden = false;
  $('busyText').textContent = text;
  $('busyProgress').hidden = progress === undefined;
  if (progress !== undefined) $('busyBar').style.width = `${Math.round(progress * 100)}%`;
}
const hideBusy = () => ($('busy').hidden = true);

// ---- drawing the image

function fitStage() {
  const img = state.image;
  if (!img || $('editor').hidden) return;
  const style = getComputedStyle(stage);
  const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const maxHeight = Math.max(240, Math.min(window.innerHeight * 0.74, 920));
  const scale = Math.min((stage.clientWidth - padding) / img.width, maxHeight / img.height, 2);
  stageInner.style.width = `${Math.round(img.width * scale)}px`;
  stageInner.style.height = `${Math.round(img.height * scale)}px`;
}

// mask + size of every region, rasterized for a width x height picture
function targetsFor(regions, width, height) {
  const img = state.image;
  const targets = [];
  for (const r of regions) {
    const mask =
      r.kind === 'object'
        ? rasterizeObjectMask(r.logits, state.lb, r.box, width, height, { margin: settings.margin / 100 })
        : rasterizeShapeMask(r.shape, r.box, img.width, img.height, width, height);
    if (mask) targets.push({ mask, size: boxSize(r.box) * (width / img.width) });
  }
  return targets;
}

function scheduleRender() {
  if (scheduleRender.frame) return;
  scheduleRender.frame = requestAnimationFrame(() => {
    scheduleRender.frame = 0;
    renderPreview();
  });
}

function renderPreview() {
  if (!state.image) return;
  const { base, width, height } = state.image.preview;
  const ctx = view.getContext('2d');
  const regions = chosen();
  if (state.comparing || regions.length === 0) return ctx.putImageData(base, 0, 0);

  const pixels = new ImageData(new Uint8ClampedArray(base.data), width, height);
  applyEffects(pixels, targetsFor(regions, width, height), settings);
  ctx.putImageData(pixels, 0, 0);
}

// ---- boxes, list and chips

function placeBox(node, [x1, y1, x2, y2]) {
  const { width, height } = state.image;
  node.style.left = `${(x1 / width) * 100}%`;
  node.style.top = `${(y1 / height) * 100}%`;
  node.style.width = `${((x2 - x1) / width) * 100}%`;
  node.style.height = `${((y2 - y1) / height) * 100}%`;
}

function renderOverlay() {
  overlay.replaceChildren();
  const img = state.image;
  if (!img) return;
  for (const r of visibleRegions()) {
    const box = make('button', r.shape === 'ellipse' ? 'region ellipse' : 'region');
    box.type = 'button';
    box.tabIndex = -1; // the list on the right is the keyboard way in
    box.setAttribute('aria-hidden', 'true');
    box.dataset.id = r.id;
    box.classList.toggle('is-selected', state.selected.has(r.id));
    box.classList.toggle('is-hover', state.hoverId === r.id);
    placeBox(box, r.box);
    box.style.setProperty('--c', dotColor(r));
    box.style.setProperty('--c-dark', `hsl(${hue(r)} 65% 34%)`);
    // small boxes on top of big ones, so they stay clickable
    const area = ((r.box[2] - r.box[0]) * (r.box[3] - r.box[1])) / (img.width * img.height);
    box.style.zIndex = 10 + Math.round((1 - area) * 100);
    box.append(make('span', 'region-label', r.label));
    overlay.append(box);
  }
}

function rowFor(r) {
  const row = make('label', 'row');
  row.dataset.id = r.id;
  row.classList.toggle('is-hover', state.hoverId === r.id);

  const check = make('input');
  check.type = 'checkbox';
  check.checked = state.selected.has(r.id);
  const percent = r.kind === 'object' ? Math.round(r.score * 100) : 0;
  check.setAttribute('aria-label', r.kind === 'object' ? t('a11y.region', { name: r.label, pct: percent }) : r.label);

  const dot = make('span', 'dot');
  dot.style.setProperty('--c', dotColor(r));
  row.append(check, dot, make('span', 'name', r.label));

  if (r.kind === 'object') {
    row.append(make('span', 'meta', `${percent}%`));
  } else {
    const remove = make('button', 'remove');
    remove.type = 'button';
    remove.dataset.remove = r.id;
    remove.title = t('region.remove');
    remove.setAttribute('aria-label', `${t('region.remove')}: ${r.label}`);
    remove.innerHTML = '<svg class="i" aria-hidden="true"><use href="#i-x"/></svg>';
    row.append(remove);
  }
  const item = make('li');
  item.append(row);
  return item;
}

function renderPanel() {
  const regions = visibleRegions();
  const objects = regions.filter((r) => r.kind === 'object');
  $('objectCount').textContent = state.phase === 'ready' || objects.length ? tn('objects.count', objects.length) : '';
  $('emptyObjects').hidden = !(state.phase === 'ready' && regions.length === 0);

  // one chip per class: click to (un)select all of them
  const byClass = new Map();
  for (const r of objects) byClass.set(r.classId, [...(byClass.get(r.classId) || []), r]);
  chips.replaceChildren(
    ...[...byClass.values()].map((group) => {
      const chip = make('button', 'chip');
      chip.type = 'button';
      chip.dataset.ids = group.map((r) => r.id).join(',');
      chip.setAttribute('aria-pressed', String(group.every((r) => state.selected.has(r.id))));
      const dot = make('span', 'dot');
      dot.style.setProperty('--c', dotColor(group[0]));
      chip.append(dot, className(group[0].classId) + (group.length > 1 ? ` ×${group.length}` : ''));
      return chip;
    }),
  );

  regionList.replaceChildren(...regions.map(rowFor));
}

// Patch checkboxes, chips and boxes in place: rebuilding them would drop the keyboard focus.
function refreshSelection() {
  const { selected } = state;
  for (const row of regionList.querySelectorAll('.row'))
    row.querySelector('input').checked = selected.has(row.dataset.id);
  for (const chip of chips.querySelectorAll('.chip')) {
    chip.setAttribute('aria-pressed', String(chip.dataset.ids.split(',').every((id) => selected.has(id))));
  }
  for (const box of overlay.querySelectorAll('.region'))
    box.classList.toggle('is-selected', selected.has(box.dataset.id));
  scheduleRender();
}

function select(ids, on) {
  for (const id of ids) {
    if (on) state.selected.add(id);
    else state.selected.delete(id);
  }
  refreshSelection();
}

function setHover(id) {
  if (state.hoverId === id) return;
  state.hoverId = id;
  for (const box of overlay.querySelectorAll('.region')) box.classList.toggle('is-hover', box.dataset.id === id);
  for (const row of regionList.querySelectorAll('.row')) row.classList.toggle('is-hover', row.dataset.id === id);
}

function removeManual(id) {
  state.manual = state.manual.filter((r) => r.id !== id);
  state.selected.delete(id);
  renderAll();
}

// ---- status, hint, controls

function renderStatus() {
  const box = $('modelStatus');
  if (state.phase === 'error') {
    const retry = make('button', 'btn xs', t('action.retry'));
    retry.type = 'button';
    retry.addEventListener('click', analyze);
    box.replaceChildren(make('span', 'error', state.error), retry);
    return;
  }
  const count = visibleRegions().filter((r) => r.kind === 'object').length;
  const text = {
    loading: t('status.model'),
    detecting: t('status.detecting'),
    ready: t('status.done', { n: count, ms: state.ms }),
  };
  box.textContent = text[state.phase] || '';
}

function renderHint() {
  $('hint').textContent = t(state.tool ? 'hint.draw' : 'hint.default');
  stage.classList.toggle('drawing', Boolean(state.tool));
  $('toolRect').setAttribute('aria-pressed', String(state.tool === 'rect'));
  $('toolEllipse').setAttribute('aria-pressed', String(state.tool === 'ellipse'));
}

function renderControls() {
  $('conf').value = Math.round(settings.conf * 100);
  $('strength').value = settings.strength;
  $('margin').value = settings.margin;
  $('confOut').textContent = `${Math.round(settings.conf * 100)}%`;
  $('strengthOut').textContent = `${Math.round(settings.strength)}%`;
  $('marginOut').textContent = `${Math.round(settings.margin)}%`;
  $('strengthField').hidden = settings.style === 'solid';
  document.querySelector(`input[name="style"][value="${settings.style}"]`).checked = true;
  document.querySelector(`input[name="model"][value="${settings.model}"]`).checked = true;
  $('modelFastSub').textContent = t('model.fast.sub', { size: formatBytes(MODELS.fast.bytes) });
  $('modelAccurateSub').textContent = t('model.accurate.sub', { size: formatBytes(MODELS.accurate.bytes) });
  $('langBtn').textContent = getLang() === 'fr' ? 'EN' : 'FR';
}

function renderAll() {
  renderControls();
  renderHint();
  renderStatus();
  renderPanel();
  renderOverlay();
  scheduleRender();
}

// ---- detection

async function analyze() {
  const img = state.image;
  const run = ++state.run;
  state.detections = [];
  state.phase = 'loading';
  state.error = '';
  for (const id of state.selected) if (id.startsWith('d')) state.selected.delete(id);
  renderAll();

  let failure = 'err.model';
  try {
    showBusy(t('status.model'), 0);
    detector.onProgress = (loaded, total) => {
      showBusy(t('status.modelPct', { pct: Math.round((loaded / total) * 100) }), loaded / total);
    };
    await detector.load(settings.model);
    if (run !== state.run) return;

    failure = 'err.detect';
    state.phase = 'detecting';
    showBusy(t('status.detecting'));
    renderStatus();
    await nextFrame(); // let the spinner show up before the worker gets busy
    const result = await detector.detect(img.full, img.width, img.height);
    if (run !== state.run) return;

    state.lb = result.lb;
    state.detections = result.detections.map((d, i) => ({
      ...d,
      id: `d${i + 1}`,
      kind: 'object',
      hue: (d.classId * 137.508) % 360, // golden angle: neighbours get very different colors
    }));
    state.ms = Math.round(result.ms);
    state.phase = 'ready';
  } catch (err) {
    if (run !== state.run) return;
    state.phase = 'error';
    state.error = t(failure, { msg: err.message });
    toast(state.error, true);
  }
  hideBusy();
  renderAll();
}

// ---- opening images

function reset(image) {
  state.run++; // forget the analysis in progress
  Object.assign(state, {
    image,
    lb: null,
    detections: [],
    manual: [],
    selected: new Set(),
    tool: null,
    hoverId: null,
    phase: 'idle',
  });
}

function showEditor(visible) {
  $('landing').hidden = visible;
  $('editor').hidden = !visible;
  fitStage();
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
  reset(img);
  view.width = img.preview.width;
  view.height = img.preview.height;
  view.getContext('2d').putImageData(img.preview.base, 0, 0);
  showEditor(true);
  window.scrollTo({ top: 0 });
  if (img.downscaled) toast(t('toast.downscaled', { w: img.width, h: img.height }));
  renderAll();
  analyze();
}

function openFiles(files) {
  const file = [...files].find(
    (f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|bmp|avif|heic|heif)$/i.test(f.name),
  );
  if (file) openImage(file, file.name);
  else toast(t('err.notImage'), true);
}

function newImage() {
  reset(null);
  hideBusy();
  renderAll();
  showEditor(false);
}

// ---- export

function exportBlob(type) {
  const img = state.image;
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img.full, 0, 0);

  const regions = chosen();
  if (regions.length) {
    const pixels = ctx.getImageData(0, 0, img.width, img.height);
    applyEffects(pixels, targetsFor(regions, img.width, img.height), settings);
    ctx.putImageData(pixels, 0, 0);
  }
  // a canvas has no metadata, so EXIF and GPS are gone
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('toBlob failed'))), type, 0.92);
  });
}

async function download() {
  const { image } = state;
  const jpeg = image.mime === 'image/jpeg';
  showBusy(t('status.exporting'));
  await nextFrame();
  try {
    const blob = await exportBlob(jpeg ? 'image/jpeg' : 'image/png');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${image.name}-blurred.${jpeg ? 'jpg' : 'png'}`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 15000);
    toast(t('toast.saved'));
  } catch (err) {
    toast(t('err.export', { msg: err.message }), true);
  }
  hideBusy();
}

async function copyImage() {
  try {
    // giving the promise (not the blob) keeps Safari happy: the write has to start inside the click
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': exportBlob('image/png') })]);
    toast(t('toast.copied'));
  } catch {
    toast(t('toast.copyFailed'), true);
  }
}

// ---- drawing areas

let drag = null;

function pointOnImage(e) {
  const rect = stageInner.getBoundingClientRect();
  const { width, height } = state.image;
  return [
    clamp((e.clientX - rect.left) / rect.width, 0, 1) * width,
    clamp((e.clientY - rect.top) / rect.height, 0, 1) * height,
  ];
}

function dragBox() {
  const [ax, ay] = drag.start;
  const [bx, by] = drag.end;
  return [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
}

function endDrag(e, keep) {
  if (drag?.id !== e.pointerId) return;
  const box = dragBox();
  drag = null;
  $('draft').hidden = true;

  const screenSize = Math.min(box[2] - box[0], box[3] - box[1]) * (stageInner.clientWidth / state.image.width);
  if (!keep || screenSize < 10) return; // a click, not a drag

  const area = { id: `m${++manualCount}`, kind: 'manual', shape: state.tool, box };
  state.manual.push(area);
  state.selected.add(area.id);
  renderPanel();
  renderOverlay();
  scheduleRender();
}

overlay.addEventListener('pointerdown', (e) => {
  if (!state.tool || e.button !== 0) return;
  e.preventDefault();
  overlay.setPointerCapture(e.pointerId);
  const start = pointOnImage(e);
  drag = { id: e.pointerId, start, end: start };
  const draft = $('draft');
  draft.className = state.tool === 'ellipse' ? 'draft ellipse' : 'draft';
  draft.hidden = false;
  placeBox(draft, dragBox());
});
overlay.addEventListener('pointermove', (e) => {
  if (drag?.id !== e.pointerId) return;
  drag.end = pointOnImage(e);
  placeBox($('draft'), dragBox());
});
overlay.addEventListener('pointerup', (e) => endDrag(e, true));
overlay.addEventListener('pointercancel', (e) => endDrag(e, false));

function setTool(tool) {
  state.tool = state.tool === tool ? null : tool;
  renderHint();
}

// ---- events

overlay.addEventListener('click', (e) => {
  const box = e.target.closest('.region');
  if (box && !state.tool) select([box.dataset.id], !state.selected.has(box.dataset.id));
});
overlay.addEventListener('pointerover', (e) => setHover(e.target.closest('.region')?.dataset.id ?? null));
overlay.addEventListener('pointerleave', () => setHover(null));

regionList.addEventListener('change', (e) => {
  select([e.target.closest('.row').dataset.id], e.target.checked);
});
regionList.addEventListener('click', (e) => {
  const remove = e.target.closest('[data-remove]');
  if (!remove) return;
  e.preventDefault();
  removeManual(remove.dataset.remove);
});
regionList.addEventListener('pointerover', (e) => setHover(e.target.closest('.row')?.dataset.id ?? null));
regionList.addEventListener('pointerleave', () => setHover(null));

chips.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  const ids = chip.dataset.ids.split(',');
  select(ids, !ids.every((id) => state.selected.has(id)));
});
on('selectAll', 'click', () =>
  select(
    visibleRegions().map((r) => r.id),
    true,
  ),
);
on('selectNone', 'click', () =>
  select(
    visibleRegions().map((r) => r.id),
    false,
  ),
);

function onSlider(id, key, divisor, after) {
  on(id, 'input', (e) => {
    settings[key] = Number(e.target.value) / divisor;
    saveSettings();
    $(`${id}Out`).textContent = `${e.target.value}%`;
    after();
  });
}
onSlider('conf', 'conf', 100, () => {
  renderStatus();
  renderPanel();
  renderOverlay();
  scheduleRender();
});
onSlider('strength', 'strength', 1, scheduleRender);
onSlider('margin', 'margin', 1, scheduleRender);

for (const input of document.querySelectorAll('input[name="style"]')) {
  input.addEventListener('change', () => {
    settings.style = input.value;
    saveSettings();
    renderControls();
    scheduleRender();
  });
}
for (const input of document.querySelectorAll('input[name="model"]')) {
  input.addEventListener('change', () => {
    if (settings.model === input.value) return;
    settings.model = input.value;
    saveSettings();
    analyze();
  });
}

on('toolRect', 'click', () => setTool('rect'));
on('toolEllipse', 'click', () => setTool('ellipse'));
on('newBtn', 'click', newImage);
on('downloadBtn', 'click', download);
on('copyBtn', 'click', copyImage);

// hold the button to see the original
function compare(show) {
  state.comparing = show;
  stage.classList.toggle('comparing', show);
  scheduleRender();
}
const compareBtn = $('compareBtn');
compareBtn.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  compare(true);
});
compareBtn.addEventListener('keydown', (e) => {
  if (e.key !== ' ' && e.key !== 'Enter') return;
  e.preventDefault();
  compare(true);
});
for (const type of ['pointerup', 'pointerleave', 'pointercancel', 'blur', 'keyup']) {
  compareBtn.addEventListener(type, () => compare(false));
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.tool) setTool(state.tool);
});

// landing page: click, paste, drag and drop
on('dropzone', 'click', () => $('fileInput').click());
on('dropzone', 'keydown', (e) => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  e.preventDefault();
  $('fileInput').click();
});
on('fileInput', 'change', (e) => {
  openFiles(e.target.files);
  e.target.value = '';
});
document.addEventListener('paste', (e) => {
  const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
  if (!file) return;
  e.preventDefault();
  openImage(file, file.name || 'pasted-image');
});

// a counter, because dragenter/dragleave also fire for every child element
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $('dropOverlay').hidden = false;
});
window.addEventListener('dragover', (e) => hasFiles(e) && e.preventDefault());
window.addEventListener('dragleave', (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  $('dropOverlay').hidden = dragDepth > 0;
});
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $('dropOverlay').hidden = true;
  openFiles(e.dataTransfer.files);
});

function buildSamples() {
  $('sampleList').replaceChildren(
    ...SAMPLES.map((sample) => {
      const button = make('button', 'sample');
      button.type = 'button';
      button.dataset.sample = sample.id;
      const img = make('img');
      img.src = sample.url;
      img.alt = '';
      img.loading = 'lazy';
      button.append(img, make('span', '', t(`sample.${sample.id}`)));
      button.addEventListener('click', () => openImage(sample.url, `${sample.id}.jpg`));
      return button;
    }),
  );
}

on('langBtn', 'click', () => {
  const next = getLang() === 'fr' ? 'en' : 'fr';
  setLang(next);
  store.set('ba-lang', next);
  applyI18n();
  buildSamples();
  renderAll();
});

const root = document.documentElement;
on('themeBtn', 'click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  store.set('ba-theme', root.dataset.theme);
});
window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
  if (!store.get('ba-theme')) root.dataset.theme = e.matches ? 'dark' : 'light';
});

new ResizeObserver(fitStage).observe(stage);
window.addEventListener('resize', fitStage);

// ---- start

setLang(detectLang(store.get('ba-lang'), navigator.language));
applyI18n();
buildSamples();
renderControls();
renderHint();

if (typeof Worker === 'undefined' || typeof WebAssembly === 'undefined') {
  toast(t('err.unsupported'), true);
} else {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') $('copyBtn').hidden = true;

  // offline support, the app works fine without it
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
  }

  // start downloading the model while the visitor picks a picture (not on data saver)
  if (!navigator.connection?.saveData) {
    const warmUp = () => detector.load(settings.model).catch(() => {});
    if ('requestIdleCallback' in window) requestIdleCallback(warmUp, { timeout: 2500 });
    else setTimeout(warmUp, 800);
  }
}
