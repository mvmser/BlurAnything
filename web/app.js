// BlurAnything web: detection with YOLOv8n (ONNX Runtime Web) and blurring on a canvas.
// Everything runs in the browser, nothing is uploaded.
/* global ort */

import { gaussianBlurRGBA } from "./blur.js";
import { CLASS_FR, STRINGS } from "./i18n.js";
import { INPUT_SIZE, decode, letterbox, padBox, toTensorData } from "./yolo.js";

const MODEL_URL = "models/yolov8n.onnx";
const MODEL_BYTES = 12851144;
const MAX_SIDE = 2400; // larger images are downscaled to keep the canvas fast

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const state = {
  lang: document.documentElement.lang === "en" ? "en" : "fr",
  theme: document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  image: null, // canvas holding the (possibly downscaled) source
  name: "image",
  type: "image/jpeg",
  raw: null, // model output, kept to re-threshold without re-running the model
  detections: [],
  selected: new Set(),
  manual: [],
  tool: "pick",
  hover: -1,
  drag: null,
  compare: false,
  layer: null, // cached blurred / pixelated copy of the whole image
  status: { kind: "idle" },
};

// ---------- i18n, theme ----------
function t(key, ...args) {
  const value = STRINGS[state.lang][key] ?? STRINGS.fr[key];
  return typeof value === "function" ? value(...args) : value;
}

const className = (name) => (state.lang === "fr" ? CLASS_FR[name] || name : name);

function applyLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;
  document.title = t("title");
  document.querySelector("meta[name=description]").content = t("description");
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", t(el.dataset.i18nAria));
  $("langTxt").textContent = lang === "fr" ? "EN" : "FR";
  $("list").setAttribute("aria-label", t("objects"));
  try { localStorage.setItem("lang", lang); } catch { /* storage unavailable */ }
  renderPanel();
  renderStatus();
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  document.querySelector("meta[name=theme-color]").content = theme === "dark" ? "#121212" : "#fbfaf7";
  try { localStorage.setItem("theme", theme); } catch { /* storage unavailable */ }
  draw();
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------- status line ----------
function setStatus(status) {
  state.status = status;
  renderStatus();
}

function renderStatus() {
  const el = $("status");
  const s = state.status;
  const blurred = regionsToBlur().length;
  let text = "";
  let cls = "status";

  if (s.kind === "loading") { text = t("loadingModel", s.progress); cls += " busy"; }
  else if (s.kind === "detecting") { text = t("detecting"); cls += " busy"; }
  else if (s.kind === "error") { text = t(s.key); cls += " error"; }
  else if (s.kind === "ready") {
    const n = state.detections.length;
    if (state.tool === "draw") text = t("drawHint");
    else if (n === 0 && state.manual.length === 0) { text = t("noDetections"); cls += " warn"; }
    else if (n === 0) text = t("onlyManual", blurred);
    else if (blurred === 0) { text = t("nothingBlurred", n); cls += " warn"; }
    else text = t("summary", n, blurred);
  }

  el.className = cls;
  el.textContent = text;
  const progress = $("progress");
  progress.hidden = s.kind !== "loading" || !s.progress;
  progress.value = s.progress || 0;
  $("retry").hidden = !(s.kind === "error" && s.key === "modelError");

  const label = t("downloadN", blurred);
  $("download").textContent = label;
  $("downloadDock").textContent = label;
  $("download").disabled = $("downloadDock").disabled = !state.image || s.kind === "loading" || s.kind === "detecting";
}

// ---------- model ----------
let sessionPromise = null;
let modelLoaded = false;
let modelProgress = 0;

async function fetchModel(onProgress) {
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  if (!response.body) return new Uint8Array(await response.arrayBuffer());

  const total = Number(response.headers.get("content-length")) || MODEL_BYTES;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(99, Math.round((received / total) * 100)));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const bytes = await fetchModel((progress) => {
        modelProgress = progress;
        if (state.status.kind === "loading") setStatus({ kind: "loading", progress });
      });
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      modelLoaded = true;
      return session;
    })();
    sessionPromise.catch(() => { sessionPromise = null; });
  }
  return sessionPromise;
}

async function runModel(source) {
  const geometry = letterbox(source.width, source.height);
  const input = document.createElement("canvas");
  input.width = input.height = INPUT_SIZE;
  const ictx = input.getContext("2d", { willReadFrequently: true });
  ictx.fillStyle = "rgb(114,114,114)";
  ictx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ictx.drawImage(source, geometry.padX, geometry.padY, geometry.w, geometry.h);
  const pixels = ictx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const session = await getSession();
  const tensor = new ort.Tensor("float32", toTensorData(pixels), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const output = outputs[session.outputNames[0]];
  return { data: output.data, numBoxes: output.dims[2], geometry };
}

function applyThreshold() {
  const { data, numBoxes, geometry } = state.raw;
  const confidence = Number($("confidence").value) / 100;
  state.detections = decode(data, numBoxes, geometry, state.image.width, state.image.height, confidence);
  // People first. When there is nobody on the photo, select everything so the result is never empty.
  const people = state.detections.flatMap((d, i) => (d.object === "person" ? [i] : []));
  state.selected = new Set(people.length ? people : state.detections.map((_, i) => i));
}

// ---------- image loading ----------
// <img> applies the EXIF orientation in every browser, createImageBitmap is the fallback.
async function decodeToCanvas(file) {
  let source;
  let width;
  let height;
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    source = img;
    width = img.naturalWidth;
    height = img.naturalHeight;
  } catch {
    const bitmap = await createImageBitmap(file);
    source = bitmap;
    width = bitmap.width;
    height = bitmap.height;
  } finally {
    URL.revokeObjectURL(url);
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(width * scale));
  out.height = Math.max(1, Math.round(height * scale));
  out.getContext("2d").drawImage(source, 0, 0, out.width, out.height);
  return out;
}

async function loadFile(file) {
  if (!file || !(file.type || "").startsWith("image/")) return;
  let image;
  try {
    image = await decodeToCanvas(file);
  } catch {
    showEditor();
    setStatus({ kind: "error", key: "badFile" });
    return;
  }

  state.image = image;
  state.name = (file.name || "image").replace(/\.[^.]+$/, "") || "image";
  state.type = file.type === "image/png" ? "image/png" : "image/jpeg";
  state.manual = [];
  state.detections = [];
  state.selected = new Set();
  state.raw = null;
  state.layer = null;
  state.hover = -1;
  showEditor();
  draw();
  renderPanel();
  await detect();
}

async function detect() {
  setStatus(modelLoaded ? { kind: "detecting" } : { kind: "loading", progress: modelProgress });
  try {
    await getSession();
    setStatus({ kind: "detecting" });
    await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0))); // let the status paint
    state.raw = await runModel(state.image);
    applyThreshold();
    setStatus({ kind: "ready" });
  } catch (err) {
    console.error(err);
    setStatus({ kind: "error", key: "modelError" });
  }
  renderPanel();
  draw();
}

function showEditor() {
  $("drop").hidden = true;
  $("editor").hidden = false;
  $("dock").hidden = false;
  document.body.classList.add("editing");
  window.scrollTo({ top: 0 });
}

function reset() {
  state.image = null;
  $("file").value = "";
  $("editor").hidden = true;
  $("dock").hidden = true;
  $("drop").hidden = false;
  document.body.classList.remove("editing");
  setStatus({ kind: "idle" });
}

// ---------- effects ----------
function settings() {
  const image = state.image;
  const scale = Math.max(1, Math.max(image.width, image.height) / 900); // same look whatever the resolution
  const strength = Number($("strength").value);
  return {
    method: document.querySelector("input[name=method]:checked").value,
    sigma: Math.min(150, strength * scale),
    block: Math.max(4, Math.round(strength * scale * 0.6)),
    padding: Number($("padding").value) / 100,
  };
}

// Blur the whole image once (at reduced size for big radii) and reuse it for every region.
function blurredLayer(source, sigma) {
  const factor = Math.max(1, Math.floor(sigma / 3));
  const small = document.createElement("canvas");
  small.width = Math.max(1, Math.round(source.width / factor));
  small.height = Math.max(1, Math.round(source.height / factor));
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(source, 0, 0, small.width, small.height);
  const pixels = sctx.getImageData(0, 0, small.width, small.height);
  gaussianBlurRGBA(pixels, sigma / factor);
  sctx.putImageData(pixels, 0, 0);

  const out = document.createElement("canvas");
  out.width = source.width;
  out.height = source.height;
  const octx = out.getContext("2d");
  octx.imageSmoothingQuality = "high";
  octx.drawImage(small, 0, 0, out.width, out.height);
  return out;
}

function pixelatedLayer(source, block) {
  const small = document.createElement("canvas");
  small.width = Math.max(1, Math.ceil(source.width / block));
  small.height = Math.max(1, Math.ceil(source.height / block));
  small.getContext("2d").drawImage(source, 0, 0, small.width, small.height);

  const out = document.createElement("canvas");
  out.width = source.width;
  out.height = source.height;
  const octx = out.getContext("2d");
  octx.imageSmoothingEnabled = false;
  octx.drawImage(small, 0, 0, small.width * block, small.height * block, 0, 0, small.width * block, small.height * block);
  return out;
}

function effectLayer(s) {
  const key = s.method === "gaussian" ? `g${s.sigma.toFixed(1)}` : `p${s.block}`;
  if (!state.layer || state.layer.key !== key) {
    state.layer = {
      key,
      canvas: s.method === "gaussian" ? blurredLayer(state.image, s.sigma) : pixelatedLayer(state.image, s.block),
    };
  }
  return state.layer.canvas;
}

function regionsToBlur() {
  if (!state.image) return [];
  const picked = state.detections.filter((_, i) => state.selected.has(i));
  return [...picked, ...state.manual];
}

// Paint the photo with the effect applied. Used for the preview and for the exported file.
function paint(target, withEffect = true) {
  const image = state.image;
  if (target.canvas.width !== image.width || target.canvas.height !== image.height) {
    target.canvas.width = image.width;
    target.canvas.height = image.height;
  }
  target.drawImage(image, 0, 0);

  const regions = withEffect ? regionsToBlur() : [];
  if (!regions.length) return;
  const s = settings();
  const layer = s.method === "solid" ? null : effectLayer(s);
  for (const region of regions) {
    const r = padBox(region, s.padding, image.width, image.height);
    if (!r) continue;
    if (layer) target.drawImage(layer, r.x, r.y, r.w, r.h, r.x, r.y, r.w, r.h);
    else {
      target.fillStyle = "#000";
      target.fillRect(r.x, r.y, r.w, r.h);
    }
  }
}

function drawOverlay() {
  if (!$("boxes").checked) return;
  const lw = Math.max(2, Math.round(state.image.width / 380));
  const accent = cssVar("--acc");
  ctx.font = `700 ${Math.max(13, lw * 6)}px "Bricolage Grotesque", system-ui, sans-serif`;
  ctx.textBaseline = "middle";

  state.detections.forEach((d, i) => {
    const on = state.selected.has(i);
    const hot = i === state.hover;
    const w = d.x_max - d.x_min;
    const h = d.y_max - d.y_min;
    ctx.save();
    if (on || hot) {
      ctx.lineWidth = lw * (hot ? 1.8 : 1.4);
      ctx.strokeStyle = accent;
      ctx.strokeRect(d.x_min, d.y_min, w, h);
      const label = `${className(d.object)} ${Math.round(d.confidence * 100)}%`;
      const pad = lw * 3;
      const th = lw * 8;
      const tw = ctx.measureText(label).width + pad * 2;
      const ty = d.y_min - th >= 0 ? d.y_min - th : d.y_min;
      ctx.fillStyle = accent;
      ctx.fillRect(d.x_min, ty, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, d.x_min + pad, ty + th / 2 + 1);
    } else {
      ctx.lineWidth = lw;
      ctx.setLineDash([lw * 4, lw * 3]);
      ctx.strokeStyle = "rgba(0,0,0,.55)";
      ctx.strokeRect(d.x_min, d.y_min, w, h);
      ctx.lineDashOffset = lw * 4;
      ctx.strokeStyle = "rgba(255,255,255,.95)";
      ctx.strokeRect(d.x_min, d.y_min, w, h);
    }
    ctx.restore();
  });

  const dragRect = state.drag && dragBox(state.drag);
  if (dragRect) {
    ctx.save();
    ctx.lineWidth = lw * 1.4;
    ctx.setLineDash([lw * 4, lw * 3]);
    ctx.strokeStyle = accent;
    ctx.strokeRect(dragRect.x_min, dragRect.y_min, dragRect.x_max - dragRect.x_min, dragRect.y_max - dragRect.y_min);
    ctx.restore();
  }
}

function draw() {
  if (!state.image) return;
  paint(ctx, !state.compare);
  if (!state.compare) drawOverlay();
}

let drawQueued = false;
function scheduleDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

// ---------- side panel ----------
function setSelection(indices) {
  state.selected = new Set(indices);
  renderPanel();
  draw();
}

function toggleIndex(i) {
  const next = new Set(state.selected);
  next.has(i) ? next.delete(i) : next.add(i);
  setSelection(next);
}

function renderPanel() {
  const groups = new Map();
  state.detections.forEach((d, i) => groups.set(d.object, [...(groups.get(d.object) || []), i]));

  const chips = $("chips");
  chips.replaceChildren();
  for (const [name, indices] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
    const count = indices.filter((i) => state.selected.has(i)).length;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.setAttribute("aria-pressed", String(count === indices.length));
    if (count > 0 && count < indices.length) chip.dataset.partial = "true";
    const label = document.createElement("span");
    label.textContent = className(name);
    const small = document.createElement("small");
    small.textContent = String(indices.length);
    chip.append(label, small);
    chip.onclick = () => {
      const next = new Set(state.selected);
      for (const i of indices) count === indices.length ? next.delete(i) : next.add(i);
      setSelection(next);
    };
    chips.append(chip);
  }

  const hasObjects = state.detections.length > 0;
  $("whatHint").hidden = !hasObjects;
  $("selAll").parentElement.hidden = !hasObjects;

  const list = $("list");
  list.replaceChildren();
  state.detections.forEach((d, i) => {
    const li = document.createElement("li");
    li.className = i === state.hover ? "hover" : "";
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.selected.has(i);
    box.onchange = () => toggleIndex(i);
    const name = document.createElement("span");
    name.textContent = className(d.object);
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `${Math.round(d.confidence * 100)}%`;
    label.append(box, name, score);
    li.append(label);
    li.onpointerenter = () => { state.hover = i; scheduleDraw(); };
    li.onpointerleave = () => { state.hover = -1; scheduleDraw(); };
    list.append(li);
  });
  state.manual.forEach((_, i) => {
    const li = document.createElement("li");
    const label = document.createElement("label");
    label.textContent = `${t("manualArea")} ${i + 1}`;
    label.style.cursor = "default";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "rm";
    remove.textContent = "✕";
    remove.setAttribute("aria-label", `${t("remove")} ${t("manualArea")} ${i + 1}`);
    remove.onclick = () => { state.manual.splice(i, 1); renderPanel(); draw(); };
    li.append(label, remove);
    list.append(li);
  });

  $("undo").hidden = state.manual.length === 0;
  renderStatus();
}

// ---------- canvas pointer input ----------
function toImageCoords(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function dragBox({ x0, y0, x1, y1 }) {
  return { x_min: Math.min(x0, x1), y_min: Math.min(y0, y1), x_max: Math.max(x0, x1), y_max: Math.max(y0, y1) };
}

// The smallest box under the point wins, so a person in front of a bus is picked before the bus.
function hitTest({ x, y }) {
  let best = -1;
  let bestArea = Infinity;
  state.detections.forEach((d, i) => {
    if (x < d.x_min || x > d.x_max || y < d.y_min || y > d.y_max) return;
    const area = (d.x_max - d.x_min) * (d.y_max - d.y_min);
    if (area < bestArea) { best = i; bestArea = area; }
  });
  return best;
}

let press = null;
canvas.addEventListener("pointerdown", (e) => {
  if (!state.image || e.button > 0) return;
  const p = toImageCoords(e);
  press = { id: e.pointerId, x0: p.x, y0: p.y, cx: e.clientX, cy: e.clientY, moved: false, touch: e.pointerType !== "mouse" };
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  const p = toImageCoords(e);
  if (!press) {
    if (e.pointerType === "mouse" && state.tool === "pick") {
      const hit = hitTest(p);
      canvas.classList.toggle("hit", hit >= 0);
      if (hit !== state.hover) { state.hover = hit; scheduleDraw(); }
    }
    return;
  }
  if (!press.moved && Math.hypot(e.clientX - press.cx, e.clientY - press.cy) > 6) press.moved = true;
  const canDraw = state.tool === "draw" || !press.touch; // on touch screens, dragging scrolls unless draw mode is on
  if (press.moved && canDraw) {
    state.drag = { x0: press.x0, y0: press.y0, x1: p.x, y1: p.y };
    scheduleDraw();
  }
});

function endPress(e, cancelled) {
  if (!press) return;
  const { moved } = press;
  press = null;
  const rect = state.drag && dragBox(state.drag);
  state.drag = null;
  if (cancelled) { draw(); return; }
  if (rect && rect.x_max - rect.x_min > 6 && rect.y_max - rect.y_min > 6) {
    state.manual.push(rect);
    renderPanel();
  } else if (!moved && state.tool === "pick") {
    const hit = hitTest(toImageCoords(e));
    if (hit >= 0) toggleIndex(hit);
  }
  draw();
}
canvas.addEventListener("pointerup", (e) => endPress(e, false));
canvas.addEventListener("pointercancel", (e) => endPress(e, true));
canvas.addEventListener("pointerleave", () => {
  canvas.classList.remove("hit");
  if (state.hover >= 0 && !press) { state.hover = -1; scheduleDraw(); }
});

// ---------- export ----------
async function download() {
  if (!state.image) return;
  const out = document.createElement("canvas").getContext("2d");
  paint(out); // no overlay; canvas export drops EXIF/GPS metadata
  const blob = await new Promise((resolve) => out.canvas.toBlob(resolve, state.type, 0.95));
  if (!blob) return;
  const ext = state.type === "image/png" ? "png" : "jpg";
  const filename = `${state.name}-blurred.${ext}`;

  // On phones the share sheet offers "Save Image", which is easier than a file download.
  const file = new File([blob], filename, { type: state.type });
  if (matchMedia("(pointer: coarse)").matches && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
      return;
    } catch (err) {
      if (err.name === "AbortError") return;
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- wiring ----------
const drop = $("drop");
drop.addEventListener("click", (e) => { if (!e.target.closest("button")) $("file").click(); });
$("pickBtn").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
$("sample").addEventListener("click", async () => {
  try {
    const blob = await (await fetch("samples/street.jpg")).blob();
    await loadFile(new File([blob], "street.jpg", { type: "image/jpeg" }));
  } catch {
    showEditor();
    setStatus({ kind: "error", key: "badFile" });
  }
});

document.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) drop.classList.remove("over"); });
document.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  loadFile(e.dataTransfer?.files?.[0]);
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});

for (const radio of document.querySelectorAll("input[name=tool]")) {
  radio.addEventListener("change", () => {
    state.tool = radio.value;
    canvas.classList.toggle("drawing", state.tool === "draw");
    canvas.classList.remove("hit");
    renderStatus();
  });
}
$("undo").addEventListener("click", () => { state.manual.pop(); renderPanel(); draw(); });

for (const el of document.querySelectorAll("input[name=method], #padding, #boxes")) {
  el.addEventListener("input", () => { renderStatus(); scheduleDraw(); });
}
$("strength").addEventListener("input", () => { $("strengthOut").textContent = $("strength").value; scheduleDraw(); });
$("confidence").addEventListener("input", () => {
  $("confidenceOut").textContent = `${$("confidence").value} %`;
  if (!state.raw) return;
  applyThreshold();
  renderPanel();
  scheduleDraw();
});
$("selAll").addEventListener("click", () => setSelection(state.detections.map((_, i) => i)));
$("selNone").addEventListener("click", () => setSelection([]));
$("retry").addEventListener("click", detect);

// Hold to compare with the original (mouse, touch and keyboard).
for (const id of ["compare", "compareDock"]) {
  const button = $(id);
  const set = (on) => { state.compare = on; button.classList.toggle("on", on); scheduleDraw(); };
  button.addEventListener("pointerdown", (e) => { e.preventDefault(); set(true); });
  for (const type of ["pointerup", "pointerleave", "pointercancel", "blur"]) button.addEventListener(type, () => set(false));
  button.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); set(true); } });
  button.addEventListener("keyup", () => set(false));
  button.addEventListener("contextmenu", (e) => e.preventDefault());
}

$("download").addEventListener("click", download);
$("downloadDock").addEventListener("click", download);
$("reset").addEventListener("click", reset);
$("langBtn").addEventListener("click", () => applyLanguage(state.lang === "fr" ? "en" : "fr"));
$("themeBtn").addEventListener("click", () => applyTheme(state.theme === "dark" ? "light" : "dark"));

applyLanguage(state.lang);
$("strengthOut").textContent = $("strength").value;
$("confidenceOut").textContent = `${$("confidence").value} %`;

// Start downloading the model while the visitor picks a photo, so the first detection feels instant.
if ("requestIdleCallback" in window) requestIdleCallback(() => getSession().catch(() => {}));
else setTimeout(() => getSession().catch(() => {}), 1500);
