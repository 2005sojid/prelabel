/**
 * Full-resolution review of one project image.
 *
 * The image comes from the server rather than a local `File`, so it is fetched
 * by URL and the browser's own cache does the work. Navigation follows the
 * gallery's current order, which is what makes "least confident first" useful:
 * pressing → walks you through the images most worth checking.
 */

import * as api from "../api.js";
import { $, setText, show } from "../dom.js";
import { drawClassification, drawDetections, fitTransform } from "../detections.js";
import { confidence } from "../state.js";
import { visibleDetections } from "../detections.js";

const ZOOM_IN = 1.1;
const ZOOM_OUT = 0.9;

let project = null;
let items = [];
let index = -1;
let image = null;

const view = { scale: 1, offsetX: 0, offsetY: 0 };
const pan = { active: false, startX: 0, startY: 0 };

const ratio = () => window.devicePixelRatio || 1;

export const isOpen = () => index >= 0;

export function open(currentProject, currentItems, position) {
  project = currentProject;
  items = currentItems;
  index = position;
  show($("lightbox"), true);
  render();
}

export function close() {
  index = -1;
  image = null;
  show($("lightbox"), false);
}

export function navigate(direction) {
  if (index < 0 || !items.length) return;
  index = (index + direction + items.length) % items.length;
  render();
}

export function refresh() {
  if (index < 0) return;
  // The counts are threshold-dependent too, so repainting the boxes without
  // them leaves the header claiming a number that is no longer on screen.
  renderCounts();
  paint();
}

function current() {
  return items[index];
}

function render() {
  const item = current();
  if (!item || !project) return;

  setText($("lb-name"), item.rel_path);
  setText(
    $("lb-meta"),
    `${item.inference_ms ? `${item.inference_ms.toFixed(1)} ms · ` : ""}${item.width}×${item.height}` +
      (item.review_priority ? ` · priority ${item.review_priority.toFixed(2)}` : ""),
  );
  renderCounts();

  image = new Image();
  image.onload = () => {
    resetView();
    paint();
  };
  image.onerror = () => paint();
  image.src = api.itemImageUrl(project.id, item.id);
}

/**
 * The counts in the header, which all move with the confidence slider.
 *
 * Both sides are filtered: a full baseline count beside a filtered detection
 * count reads as a disagreement that is really just the threshold. The verdict
 * is the exception — it is the stored diff over everything the run produced —
 * so the tooltip says which number the slider moves and which it does not.
 */
function renderCounts() {
  const item = current();
  if (!item) return;
  const shown = visibleDetections(item.detections, confidence()).length;
  const baseline = visibleDetections(item.baseline || [], confidence()).length;
  const compared = item.baseline?.length
    ? ` · baseline ${baseline}${item.disputed ? `, ${item.disputed} differ` : ", identical"}`
    : "";
  setText($("lb-count"), `${shown} det${compared}`);
  $("lb-count").title = item.baseline?.length
    ? "Counts follow the confidence slider. The differ count is the stored diff, over every detection."
    : "";
}

function resetView() {
  const item = current();
  const wrap = $("lb-canvas-wrap");
  if (!item || !image?.width) return;
  const fit = fitTransform(
    wrap.clientWidth * ratio(), wrap.clientHeight * ratio(),
    image.width, image.height,
  );
  view.scale = fit.scale;
  view.offsetX = fit.offsetX;
  view.offsetY = fit.offsetY;
}

function paint() {
  const item = current();
  if (!item) return;

  const canvas = $("lb-canvas");
  const wrap = $("lb-canvas-wrap");
  const ctx = canvas.getContext("2d");

  canvas.width = wrap.clientWidth * ratio();
  canvas.height = wrap.clientHeight * ratio();
  canvas.style.width = `${wrap.clientWidth}px`;
  canvas.style.height = `${wrap.clientHeight}px`;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!image?.width) return;

  ctx.save();
  ctx.translate(view.offsetX, view.offsetY);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(image, 0, 0);

  const visible = visibleDetections(item.detections, confidence());
  if (item.task === "classify") {
    ctx.restore();
    drawClassification(ctx, visible);
    return;
  }
  drawDetections(ctx, visible, view.scale);
  if (showingBaseline && item.baseline?.length) drawBaseline(ctx, item.baseline, view.scale);
  ctx.restore();
}

/**
 * Draw the comparison set over the current one, dashed and in white.
 *
 * Two sets of coloured boxes are unreadable, so the baseline gets a single
 * neutral outline: what you are looking for is *where the two disagree*, and a
 * dashed box with no solid box under it is exactly that, at a glance.
 */
function drawBaseline(ctx, detections, scale) {
  ctx.save();
  ctx.setLineDash([8 / scale, 5 / scale]);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2 / scale;
  for (const detection of visibleDetections(detections, confidence())) {
    if (!detection.box) continue;
    const [x1, y1, x2, y2] = detection.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.restore();
}

let showingBaseline = true;

export function toggleBaseline(visible) {
  showingBaseline = visible;
  if (index >= 0) paint();
}

export function initProjectLightbox() {
  const wrap = $("lb-canvas-wrap");

  wrap.addEventListener("wheel", (event) => {
    if (index < 0) return;
    event.preventDefault();
    const bounds = wrap.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * ratio();
    const y = (event.clientY - bounds.top) * ratio();
    const factor = event.deltaY > 0 ? ZOOM_OUT : ZOOM_IN;
    view.offsetX = x - (x - view.offsetX) * factor;
    view.offsetY = y - (y - view.offsetY) * factor;
    view.scale *= factor;
    paint();
  }, { passive: false });

  wrap.addEventListener("mousedown", (event) => {
    if (index < 0) return;
    pan.active = true;
    pan.startX = event.clientX * ratio() - view.offsetX;
    pan.startY = event.clientY * ratio() - view.offsetY;
  });
  window.addEventListener("mousemove", (event) => {
    if (!pan.active || index < 0) return;
    view.offsetX = event.clientX * ratio() - pan.startX;
    view.offsetY = event.clientY * ratio() - pan.startY;
    paint();
  });
  window.addEventListener("mouseup", () => { pan.active = false; });
  wrap.addEventListener("dblclick", () => {
    if (index < 0) return;
    resetView();
    paint();
  });
}
