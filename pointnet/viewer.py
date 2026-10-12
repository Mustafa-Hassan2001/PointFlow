"""Build a self-contained interactive HTML viewer for segmentation results.

The page embeds every room's points (quantized), colours, ground-truth and
predicted labels, and renders them with three.js (loaded from a CDN). Open the
file in any browser: orbit / pan / zoom, switch between prediction, ground
truth, error map and original colour, and toggle classes on and off.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from .data.s3dis import CLASS_COLORS, CLASSES, NUM_CLASSES
from .metrics import ConfusionMatrix


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _metrics(cm: ConfusionMatrix) -> dict:
    nan_to_none = lambda a: [None if np.isnan(v) else round(float(v), 4) for v in a]  # noqa: E731
    return {"oa": cm.overall_accuracy, "macc": cm.mean_accuracy, "miou": cm.mean_iou,
            "iou": nan_to_none(cm.iou), "acc": nan_to_none(cm.class_accuracy)}


def encode_room(name: str, xyz: np.ndarray, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                max_points: int | None = 200_000, seed: int = 0) -> dict:
    """Pack one room for the viewer. Metrics use all points; display may be sub-sampled."""
    cm = ConfusionMatrix(NUM_CLASSES)
    cm.update(pred, gt)
    if max_points and len(xyz) > max_points:
        keep = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb, gt, pred = xyz[keep], rgb[keep], gt[keep], pred[keep]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    q = np.round((xyz - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype("<u2")
    return {
        "name": name, "n": int(len(xyz)), "total": int(cm.matrix.sum()),
        "min": lo.round(4).tolist(), "max": hi.round(4).tolist(),
        "xyz": _b64(q), "rgb": _b64(np.clip(rgb, 0, 255).astype(np.uint8)),
        "gt": _b64(gt.astype(np.uint8)), "pred": _b64(pred.astype(np.uint8)),
        "counts": np.bincount(gt, minlength=NUM_CLASSES).tolist(),
        "metrics": _metrics(cm), "_cm": cm,
    }


def build_viewer_html(rooms: list[dict], title: str = "PointNet S3DIS Viewer", subtitle: str = "",
                      notice: str = "", standalone: bool = True) -> str:
    """Render the viewer page. ``standalone`` adds the doctype/meta needed to open the file directly."""
    total = ConfusionMatrix(NUM_CLASSES)
    for r in rooms:
        total.matrix += r["_cm"].matrix
    payload = {
        "title": title, "subtitle": subtitle, "notice": notice,
        "classes": CLASSES, "colors": CLASS_COLORS.tolist(),
        "overall": _metrics(total),
        "rooms": [{k: v for k, v in r.items() if k != "_cm"} for r in rooms],
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    if standalone:
        page = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + page + "\n</html>\n")
    return page


_TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: instrument console - 3D viewport on the left, readout panel on the right; stacks on phones. */
:root {
  --bg: #eef1f5; --panel: #ffffff; --fg: #18202b; --muted: #5d6b7c; --line: #d5dce5;
  --view: #dfe4eb; --accent: #3550c8; --accent-ink: #ffffff; --bad: #d63a2f; --ok: #9aa6b4;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1319; --panel: #161c24; --fg: #e4e9f0; --muted: #8e9bab; --line: #29323e;
  --view: #0a0d12; --accent: #8fa2ff; --accent-ink: #0f1319; --bad: #ff6b5e; --ok: #4a5462; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1319; --panel: #161c24; --fg: #e4e9f0; --muted: #8e9bab; --line: #29323e;
  --view: #0a0d12; --accent: #8fa2ff; --accent-ink: #0f1319; --bad: #ff6b5e; --ok: #4a5462; color-scheme: dark }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.45; }
.app { display: flex; flex-direction: column; gap: 12px; padding-inline: 16px; padding-block: 14px; min-height: 100%; box-sizing: border-box; }
header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 10px 24px; }
h1 { font-size: 20px; font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
.sub { color: var(--muted); margin: 2px 0 0; font-size: 13px; }
.notice { font-size: 12px; color: var(--muted); border: 1px dashed var(--line); border-radius: 6px; padding: 6px 10px; }
.notice b { color: var(--fg); font-weight: 600; }
.stats { display: flex; gap: 18px; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; }
.stat .k { font-size: 12px; color: var(--muted); }
.stat .v { font-family: var(--mono); font-size: 20px; font-variant-numeric: tabular-nums; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 12px; flex: 1; min-height: 520px; }
.viewport { position: relative; background: var(--view); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; min-height: 420px; touch-action: none; cursor: grab; }
.viewport:active { cursor: grabbing; }
.viewport canvas { display: block; width: 100%; height: 100%; position: absolute; inset: 0; }
.hud { position: absolute; left: 10px; bottom: 8px; font-family: var(--mono); font-size: 11px; color: var(--muted); pointer-events: none; }
.hud-r { left: auto; right: 10px; }
.err { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; padding: 20px; color: var(--muted); }
aside { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.group { display: flex; flex-direction: column; gap: 8px; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
select, .seg button, .btn { font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
select { padding: 6px 8px; width: 100%; }
.seg { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.seg button { padding: 6px 8px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.btn { padding: 6px 10px; cursor: pointer; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.row { display: flex; gap: 8px; align-items: center; }
input[type=range] { flex: 1; accent-color: var(--accent); }
.legend { display: flex; flex-direction: column; gap: 2px; }
.cls { display: grid; grid-template-columns: 12px minmax(0, 1fr) 54px 44px; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 5px; cursor: pointer; border: 0; background: none; color: var(--fg); font: inherit; text-align: left; }
.cls:hover { background: var(--bg); }
.cls[aria-pressed="false"] { opacity: 0.38; }
.sw { width: 12px; height: 12px; border-radius: 3px; }
.bar { height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
.num { font-family: var(--mono); font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.hint { font-size: 12px; color: var(--muted); margin: 0; }
.legend-head { display: grid; grid-template-columns: 12px minmax(0, 1fr) 54px 44px; gap: 8px; padding: 0 6px; font-size: 11px; color: var(--muted); }
.legend-head span:nth-child(n+3) { text-align: right; }
@media (max-width: 820px) {
  .main { grid-template-columns: minmax(0, 1fr); }
  .viewport { min-height: 0; aspect-ratio: 4 / 3; max-width: 100%; }
  .hud-r { display: none; }
}
</style>

<div class="app">
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div class="stats" id="stats"></div>
  </header>
  <div class="notice" id="notice" hidden></div>
  <div class="main">
    <div class="viewport" id="viewport" aria-label="3D point cloud view">
      <div class="hud" id="hud"></div>
      <div class="hud hud-r">drag rotate · shift/right-drag pan · scroll zoom</div>
    </div>
    <aside>
      <div class="group">
        <label class="label" for="room">Room</label>
        <select id="room"></select>
      </div>
      <div class="group">
        <span class="label">Colour by</span>
        <div class="seg" id="mode" role="group" aria-label="Colour mode">
          <button type="button" data-mode="pred" aria-pressed="true">Prediction</button>
          <button type="button" data-mode="gt" aria-pressed="false">Ground truth</button>
          <button type="button" data-mode="err" aria-pressed="false">Errors</button>
          <button type="button" data-mode="rgb" aria-pressed="false">RGB</button>
        </div>
      </div>
      <div class="group">
        <label class="label" for="size">Point size</label>
        <div class="row"><input type="range" id="size" min="1" max="8" step="0.5" value="3"><button class="btn" id="reset" type="button">Reset view</button></div>
      </div>
      <div class="group">
        <span class="label">Classes in this room</span>
        <div class="legend-head"><span></span><span>class</span><span>IoU</span><span>pts</span></div>
        <div class="legend" id="legend"></div>
        <p class="hint">Click a class to hide or show its points (filters by the labels being shown).</p>
      </div>
    </aside>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const fmtPct = (v) => (v == null ? "–" : (v * 100).toFixed(1));
const fmtN = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n);
const decode = (b64, Type) => { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Type(u.buffer); };

$("title").textContent = DATA.title;
$("subtitle").textContent = DATA.subtitle;
if (DATA.notice) { $("notice").innerHTML = DATA.notice; $("notice").hidden = false; }

const rooms = DATA.rooms.map((r) => {
  const q = decode(r.xyz, Uint16Array), n = r.n;
  const pos = new Float32Array(n * 3);
  const cx = (r.min[0] + r.max[0]) / 2, cy = (r.min[1] + r.max[1]) / 2, cz = (r.min[2] + r.max[2]) / 2;
  for (let i = 0; i < n; i++) {
    const x = r.min[0] + q[i * 3] / 65535 * (r.max[0] - r.min[0]);
    const y = r.min[1] + q[i * 3 + 1] / 65535 * (r.max[1] - r.min[1]);
    const z = r.min[2] + q[i * 3 + 2] / 65535 * (r.max[2] - r.min[2]);
    pos[i * 3] = x - cx; pos[i * 3 + 1] = z - cz; pos[i * 3 + 2] = -(y - cy); // S3DIS z-up -> three.js y-up
  }
  const ext = Math.hypot(r.max[0] - r.min[0], r.max[1] - r.min[1], r.max[2] - r.min[2]);
  return { ...r, pos, rgb: decode(r.rgb, Uint8Array), gt: decode(r.gt, Uint8Array), pred: decode(r.pred, Uint8Array), ext };
});

const state = { room: 0, mode: "pred", hidden: new Set(), size: 3 };
try { const s = JSON.parse(localStorage.getItem("pn-viewer") || "{}"); if (s.mode) state.mode = s.mode; if (s.size) state.size = s.size; } catch (e) {}
const save = () => { try { localStorage.setItem("pn-viewer", JSON.stringify({ mode: state.mode, size: state.size })); } catch (e) {} };

// Header stats
function renderStats() {
  const r = rooms[state.room], o = DATA.overall;
  const items = [["Points", fmtN(r.total)], ["Room OA", fmtPct(r.metrics.oa) + "%"], ["Room mIoU", fmtPct(r.metrics.miou) + "%"],
                 ["Test-area mIoU", fmtPct(o.miou) + "%"], ["Test-area OA", fmtPct(o.oa) + "%"]];
  $("stats").innerHTML = items.map(([k, v]) => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}

// Room selector
$("room").innerHTML = rooms.map((r, i) => `<option value="${i}">${r.name} · mIoU ${fmtPct(r.metrics.miou)}%</option>`).join("");
$("room").addEventListener("change", (e) => { state.room = +e.target.value; state.hidden.clear(); rebuild(); resetView(); });

// Mode buttons
document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; save(); rebuild(); }));

// Legend
function renderLegend() {
  const r = rooms[state.room];
  $("legend").innerHTML = DATA.classes.map((c, i) => {
    const iou = r.metrics.iou[i], col = DATA.colors[i];
    if (!r.counts[i] && !r.pred.includes(i)) return "";
    return `<button type="button" class="cls" data-c="${i}" aria-pressed="${!state.hidden.has(i)}">
      <span class="sw" style="background:rgb(${col})"></span>
      <span><span>${c}</span><div class="bar"><i style="width:${iou == null ? 0 : iou * 100}%"></i></div></span>
      <span class="num">${fmtPct(iou)}</span><span class="num">${fmtN(r.counts[i])}</span></button>`;
  }).join("");
  $("legend").querySelectorAll(".cls").forEach((el) => el.addEventListener("click", () => {
    const c = +el.dataset.c; state.hidden.has(c) ? state.hidden.delete(c) : state.hidden.add(c); rebuild();
  }));
}

// three.js scene
const vp = $("viewport");
let renderer, scene, camera, points, geom, mat;
const orbit = { theta: -0.8, phi: 1.05, radius: 10, target: null };

function cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function errColors() { return [new THREE.Color(cssColor("--ok")), new THREE.Color(cssColor("--bad"))]; }

function init() {
  if (typeof THREE === "undefined") {
    vp.insertAdjacentHTML("beforeend", '<div class="err">The 3D library could not load. Check your internet connection and reload the page.</div>');
    return false;
  }
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  vp.prepend(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  orbit.target = new THREE.Vector3();
  geom = new THREE.BufferGeometry();
  mat = new THREE.PointsMaterial({ size: state.size, sizeAttenuation: false, vertexColors: true });
  points = new THREE.Points(geom, mat);
  scene.add(points);
  new ResizeObserver(resize).observe(vp);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyBg(); rebuild(); });
  new MutationObserver(() => { applyBg(); rebuild(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  bindControls();
  return true;
}
function applyBg() { if (renderer) renderer.setClearColor(new THREE.Color(cssColor("--view"))); }
function resize() {
  const w = vp.clientWidth, h = vp.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); draw();
}

function rebuild() {
  const r = rooms[state.room];
  document.querySelectorAll("#mode button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  renderStats(); renderLegend();
  if (!renderer) return;
  const filterLabels = state.mode === "gt" ? r.gt : r.pred;
  let m = 0;
  for (let i = 0; i < r.n; i++) if (!state.hidden.has(filterLabels[i])) m++;
  const pos = new Float32Array(m * 3), col = new Float32Array(m * 3);
  const [ok, bad] = errColors();
  let errCount = 0, j = 0;
  for (let i = 0; i < r.n; i++) {
    if (state.hidden.has(filterLabels[i])) continue;
    pos[j * 3] = r.pos[i * 3]; pos[j * 3 + 1] = r.pos[i * 3 + 1]; pos[j * 3 + 2] = r.pos[i * 3 + 2];
    let c;
    if (state.mode === "rgb") c = [r.rgb[i * 3] / 255, r.rgb[i * 3 + 1] / 255, r.rgb[i * 3 + 2] / 255];
    else if (state.mode === "err") { const wrong = r.gt[i] !== r.pred[i]; if (wrong) errCount++; const e = wrong ? bad : ok; c = [e.r, e.g, e.b]; }
    else { const k = DATA.colors[state.mode === "gt" ? r.gt[i] : r.pred[i]]; c = [k[0] / 255, k[1] / 255, k[2] / 255]; }
    col[j * 3] = c[0]; col[j * 3 + 1] = c[1]; col[j * 3 + 2] = c[2];
    j++;
  }
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geom.computeBoundingSphere();
  $("hud").textContent = state.mode === "err"
    ? `${fmtN(m)} points shown · ${fmtN(errCount)} misclassified (red)`
    : `${fmtN(m)} of ${fmtN(r.n)} points shown`;
  draw();
}

function resetView() { const r = rooms[state.room]; orbit.theta = -0.8; orbit.phi = 1.05; orbit.radius = r.ext * 1.15; if (orbit.target) orbit.target.set(0, 0, 0); draw(); }
function draw() {
  if (!renderer) return;
  const { theta, phi, radius, target } = orbit;
  camera.position.set(target.x + radius * Math.sin(phi) * Math.sin(theta), target.y + radius * Math.cos(phi), target.z + radius * Math.sin(phi) * Math.cos(theta));
  camera.lookAt(target);
  renderer.render(scene, camera);
}

function bindControls() {
  const ptrs = new Map(); let last = null, pinch = null;
  const el = renderer.domElement;
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  el.addEventListener("pointerdown", (e) => { el.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, e); last = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; });
  el.addEventListener("pointerup", (e) => { ptrs.delete(e.pointerId); pinch = null; last = null; });
  el.addEventListener("pointercancel", (e) => { ptrs.delete(e.pointerId); pinch = null; last = null; });
  el.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.set(e.pointerId, e);
    if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()]; const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pinch) { orbit.radius = Math.min(Math.max(orbit.radius * pinch / d, 0.5), 200); draw(); }
      pinch = d; return;
    }
    if (!last) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y; last.x = e.clientX; last.y = e.clientY;
    if (last.pan) {
      const s = orbit.radius * 0.0015;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    } else {
      orbit.theta -= dx * 0.006; orbit.phi = Math.min(Math.max(orbit.phi - dy * 0.006, 0.05), Math.PI - 0.05);
    }
    draw();
  });
  el.addEventListener("wheel", (e) => { e.preventDefault(); orbit.radius = Math.min(Math.max(orbit.radius * Math.exp(e.deltaY * 0.001), 0.5), 200); draw(); }, { passive: false });
}

$("size").value = state.size;
$("size").addEventListener("input", (e) => { state.size = +e.target.value; if (mat) mat.size = state.size; save(); draw(); });
$("reset").addEventListener("click", resetView);

if (init()) applyBg();
rebuild(); resetView(); if (renderer) resize();
</script>
"""
