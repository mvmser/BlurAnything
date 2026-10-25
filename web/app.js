// BlurAnything web: detection with YOLOv8n (ONNX Runtime Web) and blurring on a canvas.
/* global ort */

import { INPUT_SIZE, decode, letterbox, padBox, toTensorData } from "./yolo.js";

const MODEL_URL = "models/yolov8n.onnx";
const MAX_SIDE = 2400; // larger images are downscaled to keep the canvas fast

const I18N = {
  fr: {
    title: "Floute n'importe quoi sur une photo.",
    lead: "Personnes, voitures, écrans et 77 autres types d'objets sont détectés par YOLOv8, puis floutés en un clic. L'IA tourne dans ton navigateur : tes images ne quittent jamais ton appareil.",
    drop: "Dépose une image ici",
    "drop-sub": "ou clique pour choisir un fichier. Tu peux aussi la coller.",
    sample: "Essayer avec une photo d'exemple",
    auto: "Flouter tous les…",
    objects: "Objets",
    draw: "Astuce : glisse sur l'image pour flouter une zone à la main.",
    style: "Style",
    gaussian: "Flou",
    pixelate: "Pixels",
    solid: "Masque",
    strength: "Intensité",
    padding: "Marge",
    confidence: "Confiance min.",
    "show-boxes": "Afficher les cadres",
    download: "Télécharger",
    new: "Nouvelle image",
    "how1-t": "Privé par conception",
    how1: "Le modèle YOLOv8n (13 Mo) est téléchargé une fois et tourne en local avec ONNX Runtime Web.",
    "how2-t": "80 classes d'objets",
    how2: "Entraîné sur COCO : personnes, véhicules, animaux, écrans, téléphones, sacs, etc.",
    "how3-t": "Aussi une API",
    how3: "Le même pipeline existe en service FastAPI (/detect et /blur) et en app Streamlit.",
    loadingModel: "Chargement du modèle…",
    detecting: "Détection en cours…",
    found: (n, ms) => `${n} objet${n > 1 ? "s" : ""} détecté${n > 1 ? "s" : ""} en ${ms} ms`,
    none: "Aucun objet détecté. Baisse la confiance ou dessine une zone.",
    manual: "zone manuelle",
    badFile: "Ce fichier n'est pas une image lisible.",
    modelError: "Impossible de charger le modèle.",
  },
  en: {
    loadingModel: "Loading model…",
    detecting: "Detecting objects…",
    found: (n, ms) => `${n} object${n > 1 ? "s" : ""} detected in ${ms} ms`,
    none: "No objects detected. Lower the confidence or draw an area.",
    manual: "manual area",
    badFile: "This file is not a readable image.",
    modelError: "Could not load the model.",
  },
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const englishText = new Map();

const state = {
  lang: "en",
  image: null, // HTMLCanvasElement holding the (possibly downscaled) source
  name: "image",
  rawOutput: null, // { data, numBoxes, geometry } kept to re-threshold without re-running the model
  detections: [],
  selected: new Set(),
  autoClasses: new Set(["person"]),
  manual: [],
  drag: null,
};

let sessionPromise = null;

// ---------- i18n ----------
function t(key, ...args) {
  const value = I18N[state.lang][key] ?? I18N.en[key] ?? englishText.get(key);
  return typeof value === "function" ? value(...args) : value;
}

function applyLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll("[data-i18n]")) {
    const key = el.dataset.i18n;
    if (!englishText.has(key)) englishText.set(key, el.textContent);
    el.textContent = lang === "en" ? englishText.get(key) : t(key);
  }
  $("lang").textContent = lang === "en" ? "FR" : "EN";
  try { localStorage.setItem("lang", lang); } catch { /* storage unavailable */ }
  if (state.image) renderPanel();
}

function setStatus(text, isError = false) {
  $("status").textContent = text || "";
  $("status").classList.toggle("error", isError);
}

// ---------- model ----------
function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
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
  const { data, numBoxes, geometry } = state.rawOutput;
  const confidence = Number($("confidence").value) / 100;
  state.detections = decode(data, numBoxes, geometry, state.image.width, state.image.height, confidence);
  state.selected = new Set();
}

// ---------- image loading ----------
async function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    showEditor();
    setStatus(t("badFile"), true);
    return;
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const source = document.createElement("canvas");
  source.width = Math.round(bitmap.width * scale);
  source.height = Math.round(bitmap.height * scale);
  source.getContext("2d").drawImage(bitmap, 0, 0, source.width, source.height);

  state.image = source;
  state.name = (file.name || "image").replace(/\.[^.]+$/, "");
  state.manual = [];
  state.detections = [];
  state.rawOutput = null;
  showEditor();
  render();
  await detect();
}

async function detect() {
  setStatus(sessionPromise ? t("detecting") : t("loadingModel"));
  try {
    await getSession();
    setStatus(t("detecting"));
    const start = performance.now();
    state.rawOutput = await runModel(state.image);
    applyThreshold();
    const ms = Math.round(performance.now() - start);
    setStatus(state.detections.length ? t("found", state.detections.length, ms) : t("none"));
  } catch (err) {
    console.error(err);
    setStatus(t("modelError"), true);
  }
  renderPanel();
  render();
}

function showEditor() {
  $("drop").hidden = true;
  $("editor").hidden = false;
}

// ---------- blurring ----------
function regionsToBlur() {
  const fromDetections = state.detections.filter(
    (d, i) => state.autoClasses.has(d.object) || state.selected.has(i),
  );
  return [...fromDetections, ...state.manual];
}

function blurInto(target, source, rect, method, strength) {
  const { x, y, w, h } = rect;
  if (method === "solid") {
    target.fillStyle = "#000";
    target.fillRect(x, y, w, h);
    return;
  }
  if (method === "pixelate") {
    const small = document.createElement("canvas");
    small.width = Math.max(1, Math.round(w / strength));
    small.height = Math.max(1, Math.round(h / strength));
    small.getContext("2d").drawImage(source, x, y, w, h, 0, 0, small.width, small.height);
    target.imageSmoothingEnabled = false;
    target.drawImage(small, 0, 0, small.width, small.height, x, y, w, h);
    target.imageSmoothingEnabled = true;
    return;
  }
  // Gaussian: blur a slightly larger crop so edges are not darkened, then clip back.
  const m = strength * 2;
  const sx = Math.max(0, x - m);
  const sy = Math.max(0, y - m);
  const sw = Math.min(source.width, x + w + m) - sx;
  const sh = Math.min(source.height, y + h + m) - sy;
  const tmp = document.createElement("canvas");
  tmp.width = sw;
  tmp.height = sh;
  const tctx = tmp.getContext("2d");
  if (supportsCanvasFilter()) {
    tctx.filter = `blur(${strength}px)`;
    tctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
  } else {
    // Safari fallback: repeated down/up sampling approximates a blur.
    const factor = Math.max(2, strength / 2);
    const small = document.createElement("canvas");
    small.width = Math.max(1, Math.round(sw / factor));
    small.height = Math.max(1, Math.round(sh / factor));
    const sctx = small.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(source, sx, sy, sw, sh, 0, 0, small.width, small.height);
    tctx.imageSmoothingQuality = "high";
    tctx.drawImage(small, 0, 0, sw, sh);
  }
  target.drawImage(tmp, x - sx, y - sy, w, h, x, y, w, h);
}

let canvasFilterSupport;
function supportsCanvasFilter() {
  if (canvasFilterSupport === undefined) {
    const test = document.createElement("canvas").getContext("2d");
    test.filter = "blur(2px)";
    canvasFilterSupport = test.filter === "blur(2px)";
  }
  return canvasFilterSupport;
}

function renderBlurred(target, withOverlay) {
  const img = state.image;
  const method = document.querySelector("input[name=method]:checked").value;
  const strength = Number($("strength").value);
  const padding = Number($("padding").value) / 100;

  target.canvas.width = img.width;
  target.canvas.height = img.height;
  target.drawImage(img, 0, 0);
  for (const region of regionsToBlur()) {
    const rect = padBox(region, padding, img.width, img.height);
    if (rect) blurInto(target, img, rect, method, strength);
  }
  if (withOverlay) drawOverlay(target);
}

function drawOverlay(target) {
  const line = Math.max(2, Math.round(state.image.width / 400));
  target.lineWidth = line;
  target.font = `${Math.max(12, line * 6)}px system-ui, sans-serif`;
  target.textBaseline = "top";
  if ($("boxes").checked) {
    state.detections.forEach((d, i) => {
      const active = state.autoClasses.has(d.object) || state.selected.has(i);
      target.strokeStyle = active ? "#c5f135" : "rgba(255,255,255,.75)";
      target.strokeRect(d.x_min, d.y_min, d.x_max - d.x_min, d.y_max - d.y_min);
      const label = `${d.object} ${Math.round(d.confidence * 100)}%`;
      const tw = target.measureText(label).width + line * 4;
      const th = line * 8;
      target.fillStyle = active ? "#c5f135" : "rgba(0,0,0,.65)";
      target.fillRect(d.x_min, Math.max(0, d.y_min - th), tw, th);
      target.fillStyle = active ? "#0d0f0c" : "#fff";
      target.fillText(label, d.x_min + line * 2, Math.max(0, d.y_min - th) + line);
    });
  }
  if (state.drag) {
    const { x0, y0, x1, y1 } = state.drag;
    target.setLineDash([line * 3, line * 2]);
    target.strokeStyle = "#c5f135";
    target.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    target.setLineDash([]);
  }
}

function render() {
  if (state.image) renderBlurred(ctx, true);
}

// ---------- side panel ----------
function renderPanel() {
  const counts = {};
  for (const d of state.detections) counts[d.object] = (counts[d.object] || 0) + 1;

  const chips = $("chips");
  chips.replaceChildren();
  for (const [name, n] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = `${name} · ${n}`;
    chip.setAttribute("aria-pressed", String(state.autoClasses.has(name)));
    chip.onclick = () => {
      state.autoClasses.has(name) ? state.autoClasses.delete(name) : state.autoClasses.add(name);
      renderPanel();
      render();
    };
    chips.append(chip);
  }

  const list = $("list");
  list.replaceChildren();
  state.detections.forEach((d, i) => {
    const li = document.createElement("li");
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.autoClasses.has(d.object) || state.selected.has(i);
    box.disabled = state.autoClasses.has(d.object);
    box.onchange = () => {
      box.checked ? state.selected.add(i) : state.selected.delete(i);
      render();
    };
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `${Math.round(d.confidence * 100)}%`;
    label.append(box, ` ${d.object}`, score);
    li.append(label);
    list.append(li);
  });
  state.manual.forEach((m, i) => {
    const li = document.createElement("li");
    const label = document.createElement("label");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.textContent = "✕";
    remove.setAttribute("aria-label", "Remove");
    remove.onclick = () => {
      state.manual.splice(i, 1);
      renderPanel();
      render();
    };
    label.append(`✎ ${t("manual")} ${i + 1}`, remove);
    li.append(label);
    list.append(li);
  });
  $("count").textContent = `(${state.detections.length + state.manual.length})`;
}

// ---------- manual areas ----------
function toImageCoords(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

canvas.addEventListener("pointerdown", (e) => {
  if (!state.image) return;
  const p = toImageCoords(e);
  state.drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!state.drag) return;
  const p = toImageCoords(e);
  Object.assign(state.drag, { x1: p.x, y1: p.y });
  render();
});
canvas.addEventListener("pointerup", () => {
  const d = state.drag;
  state.drag = null;
  if (d && Math.abs(d.x1 - d.x0) > 4 && Math.abs(d.y1 - d.y0) > 4) {
    state.manual.push({
      x_min: Math.min(d.x0, d.x1), y_min: Math.min(d.y0, d.y1),
      x_max: Math.max(d.x0, d.x1), y_max: Math.max(d.y0, d.y1),
    });
    renderPanel();
  }
  render();
});

// ---------- wiring ----------
const drop = $("drop");
drop.addEventListener("click", (e) => { if (e.target.id !== "sample") $("file").click(); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  loadFile(e.dataTransfer.files[0]);
});
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});
$("sample").addEventListener("click", async () => {
  const blob = await (await fetch("samples/street.jpg")).blob();
  loadFile(new File([blob], "street.jpg", { type: blob.type || "image/jpeg" }));
});

for (const el of document.querySelectorAll("input[name=method], #strength, #padding, #boxes")) {
  el.addEventListener("input", render);
}
$("confidence").addEventListener("input", () => {
  if (!state.rawOutput) return;
  applyThreshold();
  renderPanel();
  render();
});

$("download").addEventListener("click", () => {
  const out = document.createElement("canvas").getContext("2d");
  renderBlurred(out, false);
  out.canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.name}_blurred.jpg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/jpeg", 0.95);
});

$("reset").addEventListener("click", () => {
  state.image = null;
  $("file").value = "";
  $("editor").hidden = true;
  $("drop").hidden = false;
  setStatus("");
});

$("lang").addEventListener("click", () => applyLanguage(state.lang === "en" ? "fr" : "en"));

let savedLang = null;
try { savedLang = localStorage.getItem("lang"); } catch { /* storage unavailable */ }
applyLanguage(savedLang || (navigator.language.startsWith("fr") ? "fr" : "en"));

// Start downloading the model early so the first detection feels instant.
if ("requestIdleCallback" in window) requestIdleCallback(() => getSession().catch(() => {}));
