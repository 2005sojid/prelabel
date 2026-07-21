/**
 * The batch working set: one entry per image the user dropped in.
 *
 * Pure data plus the operations that keep it consistent. Nothing here touches
 * the DOM — {@link module:batch/gallery} owns that.
 */

import { visibleDetections } from "../detections.js";

/**
 * @typedef {object} BatchItem
 * @property {File}   file
 * @property {string} name       display name
 * @property {string} path       unique key: the folder-relative path when available
 * @property {"pending"|"done"|"error"} status
 * @property {Array}  detections
 * @property {string} task
 * @property {number} inferMs
 * @property {number} width
 * @property {number} height
 * @property {ImageBitmap|null} thumb
 * @property {number} visibleCount  detections above the current threshold
 */

/** @type {BatchItem[]} */
export let items = [];

export function createItems(files) {
  releaseAll();
  items = files.map((file) => ({
    file,
    name: file.name,
    // webkitRelativePath keeps sibling folders with same-named files distinct,
    // which matters when the batch is exported as a dataset.
    path: file.webkitRelativePath || file.name,
    status: "pending",
    detections: [],
    task: "detect",
    inferMs: 0,
    width: 0,
    height: 0,
    thumb: null,
    thumbLoading: false,
    visibleCount: 0,
    card: null,
    canvas: null,
    countEl: null,
  }));
  return items;
}

export function resetResults() {
  for (const item of items) {
    item.status = "pending";
    item.detections = [];
    item.inferMs = 0;
    item.visibleCount = 0;
  }
}

/**
 * Release decoded thumbnails.
 *
 * `ImageBitmap` holds memory outside the JS heap until it is closed, so a large
 * folder that is simply dropped on the floor keeps that memory reserved until
 * the GC eventually gets to it — if it ever does.
 */
export function releaseAll() {
  for (const item of items) {
    item.thumb?.close?.();
    item.thumb = null;
  }
  items = [];
}

export const doneItems = () => items.filter((item) => item.status === "done");
export const hasItems = () => items.length > 0;

/** The task to export as: whatever the completed results say they are. */
export function batchTask(fallback = "detect") {
  return doneItems()[0]?.task || fallback;
}

export const visibleFor = (item, threshold) => visibleDetections(item.detections, threshold);

/** Aggregate counts for the toolbar and the sidebar. */
export function summarise(threshold) {
  const perClass = new Map();
  let total = 0;
  let imagesWithDetections = 0;

  for (const item of items) {
    const visible = visibleFor(item, threshold);
    if (visible.length) imagesWithDetections += 1;
    total += visible.length;
    for (const detection of visible) {
      perClass.set(detection.class_name, (perClass.get(detection.class_name) || 0) + 1);
    }
  }

  const timings = doneItems().map((item) => item.inferMs).filter(Boolean);
  return {
    total,
    imagesWithDetections,
    perClass: [...perClass.entries()].sort((a, b) => b[1] - a[1]),
    processed: items.filter((item) => item.status !== "pending").length,
    count: items.length,
    averageMs: timings.length ? timings.reduce((a, b) => a + b, 0) / timings.length : 0,
  };
}
