/**
 * Full-resolution review of one batch item, with zoom and pan.
 *
 * Navigation follows the gallery's current filter and sort order, so pressing →
 * moves to the next image the user can actually see.
 */

import { $, setText, show } from "../dom.js";
import { drawClassification, drawDetections, fitTransform } from "../detections.js";
import { confidence } from "../state.js";
import { items, visibleFor } from "./items.js";
import { visibleOrder } from "./gallery.js";

const ZOOM_IN = 1.1;
const ZOOM_OUT = 0.9;

let index = -1;
let image = null;
let imageFor = -1;
let objectUrl = null;

const view = { scale: 1, offsetX: 0, offsetY: 0 };
const pan = { active: false, startX: 0, startY: 0 };

export const isOpen = () => index >= 0;

const ratio = () => window.devicePixelRatio || 1;

export function open(position) {
  const item = items[position];
  if (!item || item.status === "pending") return;
  index = position;
  show($("lightbox"), true);
  render();
}

export function close() {
  index = -1;
  show($("lightbox"), false);
  releaseImage();
}

function releaseImage() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  image = null;
  imageFor = -1;
}

export function navigate(direction) {
  if (index < 0 || !visibleOrder.length) return;
  let position = visibleOrder.indexOf(index);
  if (position < 0) position = 0;
  position = (position + direction + visibleOrder.length) % visibleOrder.length;
  index = visibleOrder[position];
  render();
}

/** Re-read the current item — used when the confidence threshold changes. */
export function refresh() {
  if (index >= 0) render();
}

function render() {
  const item = items[index];
  if (!item) return;

  setText($("lb-name"), item.name);
  setText($("lb-meta"), `${item.inferMs ? `${item.inferMs.toFixed(1)} ms · ` : ""}${item.width}×${item.height}`);
  setText($("lb-count"), `${visibleFor(item, confidence()).length} det`);

  if (imageFor !== index) {
    imageFor = index;
    releaseImageKeepingIndex();
    objectUrl = URL.createObjectURL(item.file);
    image = new Image();
    image.onload = () => {
      resetView();
      paint();
    };
    image.onerror = () => paint();
    image.src = objectUrl;
    return;
  }
  paint(); // same image (e.g. threshold changed) — keep the current zoom
}

function releaseImageKeepingIndex() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  image = null;
}

function resetView() {
  const item = items[index];
  const wrap = $("lb-canvas-wrap");
  if (!item || !image) return;
  const fit = fitTransform(wrap.clientWidth * ratio(), wrap.clientHeight * ratio(), item.width, item.height);
  view.scale = fit.scale;
  view.offsetX = fit.offsetX;
  view.offsetY = fit.offsetY;
}

function paint() {
  const item = items[index];
  if (!item) return;

  const canvas = $("lb-canvas");
  const wrap = $("lb-canvas-wrap");
  const ctx = canvas.getContext("2d");

  const width = (canvas.width = wrap.clientWidth * ratio());
  const height = (canvas.height = wrap.clientHeight * ratio());
  canvas.style.width = `${wrap.clientWidth}px`;
  canvas.style.height = `${wrap.clientHeight}px`;
  ctx.clearRect(0, 0, width, height);
  if (!image || !image.width) return;

  ctx.save();
  ctx.translate(view.offsetX, view.offsetY);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(image, 0, 0);

  const visible = visibleFor(item, confidence());
  if (item.task === "classify") {
    ctx.restore();
    drawClassification(ctx, visible);
    return;
  }
  drawDetections(ctx, visible, view.scale);
  ctx.restore();
}

export function initLightbox() {
  // The close and navigation buttons are bound in main.js, which routes them to
  // whichever lightbox is open — this one, or the project one.
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
    pan.active = true;
    pan.startX = event.clientX * ratio() - view.offsetX;
    pan.startY = event.clientY * ratio() - view.offsetY;
    wrap.style.cursor = "grabbing";
  });
  window.addEventListener("mousemove", (event) => {
    if (!pan.active || index < 0) return;
    view.offsetX = event.clientX * ratio() - pan.startX;
    view.offsetY = event.clientY * ratio() - pan.startY;
    paint();
  });
  window.addEventListener("mouseup", () => {
    pan.active = false;
    wrap.style.cursor = "";
  });
  wrap.addEventListener("dblclick", () => {
    resetView();
    paint();
  });
}
