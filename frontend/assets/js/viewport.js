/**
 * The main canvas: one zoomable, pannable image with its detections on top.
 *
 * Owns the view transform (scale + offset) so no other module has to think about
 * coordinate spaces. The webcam path uses {@link renderSource}, which fits the
 * frame to the canvas instead — live video has nothing to pan around.
 */

import { $ } from "./dom.js";
import { drawClassification, drawDetections, fitTransform } from "./detections.js";

const ZOOM_IN = 1.1;
const ZOOM_OUT = 0.9;
const FIT_MARGIN = 0.9;

const view = { scale: 1, offsetX: 0, offsetY: 0 };
const pan = { active: false, startX: 0, startY: 0 };

let image = null;
let detections = [];
let task = "detect";
let interactive = true;

let canvas;
let ctx;
let container;
let tooltip;

// --- public -----------------------------------------------------------------

export function initViewport() {
  canvas = $("canvas");
  ctx = canvas.getContext("2d");
  container = $("viewport");
  tooltip = $("tooltip");

  container.addEventListener("wheel", onWheel, { passive: false });
  container.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", () => (pan.active = false));
}

/** Show a decoded image and fit it to the viewport. */
export function showImage(nextImage) {
  image = nextImage;
  interactive = true;
  canvas.hidden = false;
  reset();
}

/** Replace the results drawn over the current image. */
export function setResults(nextDetections, nextTask) {
  detections = nextDetections || [];
  if (nextTask) task = nextTask;
}

export function clearImage() {
  image = null;
  detections = [];
  hideTooltip();
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/** Re-fit the image to the viewport. */
export function reset() {
  if (!image) return;
  const scale = Math.min(container.clientWidth / image.width, container.clientHeight / image.height) * FIT_MARGIN;
  view.scale = scale;
  view.offsetX = (container.clientWidth - image.width * scale) / 2;
  view.offsetY = (container.clientHeight - image.height * scale) / 2;
  render();
}

export function render() {
  if (!image || !ctx) return;
  resizeToContainer();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(view.offsetX, view.offsetY);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(image, 0, 0);

  if (task === "classify") {
    ctx.restore();
    drawClassification(ctx, detections);
    return;
  }
  drawDetections(ctx, detections, view.scale);
  ctx.restore();
}

/**
 * Draw an arbitrary source (a webcam capture canvas) fitted to the viewport.
 * Zoom and pan do not apply, so the transform is recomputed each frame.
 */
export function renderSource(source, nextDetections, nextTask) {
  if (!ctx) return;
  detections = nextDetections || [];
  if (nextTask) task = nextTask;
  image = null;
  interactive = false;
  canvas.hidden = false;

  resizeToContainer();
  const fit = fitTransform(canvas.width, canvas.height, source.width, source.height);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(fit.offsetX, fit.offsetY);
  ctx.scale(fit.scale, fit.scale);
  ctx.drawImage(source, 0, 0);

  if (task === "classify") {
    ctx.restore();
    drawClassification(ctx, detections);
    return;
  }
  drawDetections(ctx, detections, fit.scale);
  ctx.restore();
}

export const hasImage = () => image !== null;

// --- interaction ------------------------------------------------------------

function resizeToContainer() {
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
}

function onWheel(event) {
  if (!interactive || !image) return;
  event.preventDefault();
  const factor = event.deltaY > 0 ? ZOOM_OUT : ZOOM_IN;
  const bounds = container.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  view.offsetX = x - (x - view.offsetX) * factor;
  view.offsetY = y - (y - view.offsetY) * factor;
  view.scale *= factor;
  render();
}

function onMouseDown(event) {
  if (!interactive || !image) return;
  pan.active = true;
  pan.startX = event.clientX - view.offsetX;
  pan.startY = event.clientY - view.offsetY;
}

function onMouseMove(event) {
  if (pan.active) {
    view.offsetX = event.clientX - pan.startX;
    view.offsetY = event.clientY - pan.startY;
    render();
  }
  updateTooltip(event);
}

function updateTooltip(event) {
  if (!interactive || !image || task === "classify" || !detections.length) {
    hideTooltip();
    return;
  }
  const bounds = container.getBoundingClientRect();
  const x = (event.clientX - bounds.left - view.offsetX) / view.scale;
  const y = (event.clientY - bounds.top - view.offsetY) / view.scale;

  const hit = detections.find(
    (d) => d.box && x > d.box[0] && x < d.box[2] && y > d.box[1] && y < d.box[3],
  );
  if (!hit) {
    hideTooltip();
    return;
  }
  tooltip.hidden = false;
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.textContent = `${hit.class_name} · ${(hit.confidence * 100).toFixed(1)}%`;
}

function hideTooltip() {
  if (tooltip) tooltip.hidden = true;
}
