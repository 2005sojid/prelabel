/**
 * Batch mode: wiring between the item list, the gallery, the processor and the
 * lightbox. Everything specific to one of those lives in its own module; this
 * file only decides when they talk to each other.
 */

import { $, show } from "../dom.js";
import { on, setMode, store } from "../state.js";
import { initExportMenu } from "../export/menu.js";
import * as gallery from "./gallery.js";
import * as lightbox from "./lightbox.js";
import * as processor from "./processor.js";
import { createItems, hasItems, items, releaseAll } from "./items.js";

let confFrame = 0;

export const isActive = () => store.mode === "batch";

export async function enterBatch(files) {
  setMode("batch");
  show($("batch-view"), true);
  show($("hint"), false);
  show($("badge"), false);
  $("canvas").hidden = true;
  show($("video"), false);

  createItems(files);
  gallery.buildGallery();
  gallery.renderStats();
  gallery.setProgress(0, items.length);

  if (store.model) await processor.process();
  else gallery.statusMessage(`Load a model to process ${items.length} images`);
}

export function clearBatch() {
  processor.cancel();
  lightbox.close();
  releaseAll();
  gallery.destroyGallery();
  show($("batch-view"), false);
  setMode("single");
  show($("hint"), true);
  $("canvas").hidden = false;
}

/**
 * Dragging the slider only recounts (cheap); releasing it repaints the
 * thumbnails and the lightbox (not cheap, but by then the user has stopped).
 */
function applyConfidence(redraw) {
  cancelAnimationFrame(confFrame);
  confFrame = requestAnimationFrame(() => {
    for (const item of items) {
      if (item.status === "done") gallery.updateCardCount(item);
    }
    gallery.renderStats();
    gallery.filterSort({ redraw });
    if (redraw) lightbox.refresh();
  });
}

export function initBatch() {
  gallery.setCardHandler(lightbox.open);
  lightbox.initLightbox();
  initExportMenu();

  const refresh = () => gallery.filterSort({ redraw: true });
  $("batch-filter").addEventListener("change", refresh);
  $("batch-sort").addEventListener("change", refresh);

  let searchFrame = 0;
  $("batch-search").addEventListener("input", () => {
    cancelAnimationFrame(searchFrame);
    searchFrame = requestAnimationFrame(refresh);
  });

  $("batch-size").addEventListener("change", () => {
    $("gallery").className = $("batch-size").value;
    requestAnimationFrame(gallery.redrawVisible);
  });

  $("batch-cancel").addEventListener("click", processor.cancel);
  $("batch-clear").addEventListener("click", clearBatch);

  // A new model or device makes existing results stale — re-run everything.
  on("model:loaded", () => {
    if (isActive() && hasItems()) processor.reprocess();
  });

  on("conf:preview", () => isActive() && applyConfidence(false));
  on("conf:commit", () => isActive() && applyConfidence(true));
}

export { gallery, lightbox, processor };
