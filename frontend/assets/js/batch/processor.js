/**
 * The batch inference loop.
 *
 * Inference-first: each chunk is one batched request — a single forward pass on
 * the GPU — and image dimensions come back with the results, so progress is
 * never blocked on decoding thumbnails in the browser.
 *
 * Inference runs once at a low confidence floor and the slider filters the
 * results client-side. Moving the slider is therefore instant and never re-runs
 * the model.
 */

import * as api from "./../api.js";
import { $, show } from "../dom.js";
import { imageOptions } from "../options.js";
import { items } from "./items.js";
import { applyAspect, drawCard, markCardError, renderStats, setProgress, updateCardCount, filterSort } from "./gallery.js";

/** Images per request. Matches the batch size the OpenVINO throughput model uses. */
export const CHUNK_SIZE = 16;

/** Infer once at this floor; the slider filters everything above it. */
export const CONFIDENCE_FLOOR = 0.01;

/** Bumped to cancel an in-flight run; a loop whose token is stale stops. */
let token = 0;
let running = false;

export const isRunning = () => running;

export function cancel() {
  token += 1;
  running = false;
  show($("batch-cancel"), false);
}

/** Wait for a cancelled run to actually leave its loop. */
async function settle() {
  while (running) await new Promise((resolve) => setTimeout(resolve, 20));
}

export async function process() {
  if (running) return;
  running = true;
  const myToken = ++token;
  show($("batch-cancel"), true);

  const pending = items.filter((item) => item.status !== "done");
  let processed = items.length - pending.length;
  setProgress(processed, items.length);

  try {
    for (let offset = 0; offset < pending.length; offset += CHUNK_SIZE) {
      if (myToken !== token) break;

      const chunk = pending.slice(offset, offset + CHUNK_SIZE);
      await runChunk(chunk, myToken);
      if (myToken !== token) break;

      processed += chunk.length;
      setProgress(processed, items.length);
      renderStats();
      filterSort({ redraw: false });
    }
  } finally {
    running = false;
    if (myToken === token) show($("batch-cancel"), false);
  }
}

async function runChunk(chunk, myToken) {
  try {
    const response = await api.predictBatch(chunk.map((item) => item.file), CONFIDENCE_FLOOR, imageOptions());
    if (myToken !== token) return;
    (response.results || []).forEach((result, index) => applyResult(chunk[index], result));
  } catch (error) {
    for (const item of chunk) markCardError(item, error.message);
  }

  for (const item of chunk) {
    item.card?.classList.remove("pending");
    updateCardCount(item);
    if (item.thumb) drawCard(item);
  }
}

function applyResult(item, result) {
  if (!item) return;
  if (!result || result.status !== "ok") {
    markCardError(item, result?.detail);
    return;
  }
  item.detections = result.detections || [];
  item.task = result.task || "detect";
  item.inferMs = result.timings?.inference_ms || result.speed_ms || 0;

  // Dimensions from the server let the cell take its final aspect ratio before
  // the thumbnail has been decoded, so the grid does not reflow later.
  if (result.image_shape) {
    [item.height, item.width] = result.image_shape;
    applyAspect(item);
  }
  item.status = "done";
}

/**
 * Re-run everything. Used when the model or the device changes, since existing
 * results no longer reflect what is loaded.
 */
export async function reprocess() {
  cancel();
  await settle();
  for (const item of items) {
    item.status = "pending";
    item.detections = [];
    item.inferMs = 0;
    item.visibleCount = 0;
    item.card?.classList.add("pending");
    item.card?.querySelector(".err")?.remove();
  }
  setProgress(0, items.length);
  renderStats();
  await process();
}
