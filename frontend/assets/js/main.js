/**
 * Bootstrap: initialise each module, then wire the interactions that cross
 * module boundaries — global drag & drop, the confidence slider, keyboard
 * shortcuts and window resize.
 */

import * as api from "./api.js";
import { $, isTyping, pickFiles, setText, show, wireDropZone } from "./dom.js";
import { emit, on, setMode, store } from "./state.js";
import * as batch from "./batch/index.js";
import * as batchLightbox from "./batch/lightbox.js";
import * as modelPanel from "./model-panel.js";
import { refreshAvailability } from "./options.js";
import * as projects from "./projects/index.js";
import * as projectLightbox from "./projects/lightbox.js";
import * as single from "./single-view.js";
import * as viewport from "./viewport.js";
import * as webcam from "./webcam.js";

/**
 * The lightbox currently on screen, if any. Two views can open one — the
 * drag-and-drop batch and a server project — and they share the same markup, so
 * the buttons and keyboard route to whichever is actually open.
 */
function openLightbox() {
  if (projectLightbox.isOpen()) return projectLightbox;
  if (batchLightbox.isOpen()) return batchLightbox;
  return null;
}

// --- routing ----------------------------------------------------------------

/**
 * Stand down whatever view currently owns the viewport.
 *
 * The three full-screen modes — webcam, batch gallery, project — all draw over
 * the same area, so a new input has to close the others or it renders behind
 * one of them. Closing a project view does not stop its run: that lives on the
 * server and keeps going.
 */
function leaveCurrentMode() {
  if (webcam.isStreaming()) webcam.stop();
  if (batch.isActive()) batch.clearBatch();
  if (projects.isActive()) projects.closeProject();
}

/**
 * Decide what a set of dropped files means.
 * Several images open the batch gallery; a single image or a video does not.
 */
function routeMedia(files) {
  const images = files.filter((file) => single.isMediaFile(file) && !single.isVideoFile(file));
  const video = files.find(single.isVideoFile);

  leaveCurrentMode();
  if (images.length > 1) return batch.enterBatch(images);
  setMode("single");

  if (video) return single.loadMediaFile(video);
  if (images.length === 1) return single.loadMediaFile(images[0]);
  return undefined;
}

/** Route an arbitrary drop by file type: models vs media. */
function routeAny(files) {
  const models = files.filter(modelPanel.isModelFile);
  const media = files.filter(single.isMediaFile);

  if (models.length) modelPanel.loadModelFiles(models);
  if (media.length) routeMedia(media);
  // Nothing recognised — try it as a model and let the server explain why not.
  if (!models.length && !media.length) modelPanel.loadModelFiles(files);
}

// --- global drag & drop -----------------------------------------------------

function initGlobalDrop() {
  const overlay = $("drop-overlay");
  let depth = 0;

  const carriesFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  const hide = () => {
    depth = 0;
    show(overlay, false);
  };

  window.addEventListener("dragenter", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth += 1;
    show(overlay, true);
  });
  window.addEventListener("dragover", (event) => {
    if (carriesFiles(event)) event.preventDefault();
  });
  window.addEventListener("dragleave", (event) => {
    if (!carriesFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) show(overlay, false);
  });
  window.addEventListener("drop", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    hide();
    const files = [...(event.dataTransfer.files || [])];
    if (files.length) routeAny(files);
  });

  // A drop landing on a zone must also clear the page-wide overlay.
  window.addEventListener("dragend", hide);
}

// --- controls ---------------------------------------------------------------

function initMediaControls() {
  wireDropZone($("media-drop"), routeMedia, { multiple: true, accept: "image/*,video/*" });

  $("folder-btn").addEventListener("click", () => {
    pickFiles({ directory: true }, (files) => {
      const images = files.filter((file) => file.type.startsWith("image"));
      if (!images.length) return;
      leaveCurrentMode();
      batch.enterBatch(images);
    });
  });

  $("webcam-btn").addEventListener("click", async () => {
    if (webcam.isStreaming()) {
      webcam.stop();
      return;
    }
    leaveCurrentMode();
    setMode("single");
    await webcam.start(single.showResults);
  });
}

function initConfidenceSlider() {
  const slider = $("conf");
  slider.addEventListener("input", () => {
    setText($("conf-val"), `${slider.value}%`);
    emit("conf:preview", { conf: Number(slider.value) / 100 });
  });
  slider.addEventListener("change", () => {
    emit("conf:commit", { conf: Number(slider.value) / 100 });
    // Batch filters client-side; single mode has to ask the server again.
    if (!batch.isActive() && !webcam.isStreaming()) single.run();
  });
}

function initLightboxControls() {
  $("lb-close").addEventListener("click", () => openLightbox()?.close());
  $("lb-prev").addEventListener("click", () => openLightbox()?.navigate(-1));
  $("lb-next").addEventListener("click", () => openLightbox()?.navigate(1));
}

function initKeyboard() {
  window.addEventListener("keydown", (event) => {
    const lightbox = openLightbox();
    if (lightbox) {
      if (event.key === "Escape") lightbox.close();
      else if (event.key === "ArrowLeft") lightbox.navigate(-1);
      else if (event.key === "ArrowRight") lightbox.navigate(1);
      return;
    }
    // Single-key shortcuts must not fire while the user is typing a filter.
    if (isTyping() || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key.toLowerCase() === "r") viewport.reset();
  });
}

function initResize() {
  let frame = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const lightbox = openLightbox();
      if (lightbox) lightbox.refresh();
      else if (projects.isActive()) projects.gallery?.repaint?.();
      else if (batch.isActive()) batch.gallery.redrawVisible();
      else if (!single.isShowingVideo()) viewport.render();
    });
  });
}

// --- start ------------------------------------------------------------------

async function start() {
  viewport.initViewport();
  modelPanel.initModelPanel();
  batch.initBatch();
  initMediaControls();
  initGlobalDrop();
  initConfidenceSlider();
  initLightboxControls();
  initKeyboard();
  initResize();
  await projects.initProjects();

  // A newly loaded model invalidates whatever is on screen in single mode.
  on("model:loaded", () => {
    refreshAvailability();
    if (!batch.isActive() && single.hasMedia()) single.run();
  });
  on("model:cleared", () => {
    refreshAvailability();
    if (batch.isActive()) batch.gallery.statusMessage("Load a model to process this batch");
  });

  await modelPanel.loadCapabilities();
  refreshAvailability();

  // Reflect a model the server already had loaded, so a page refresh does not
  // make a working model look absent.
  try {
    const health = await api.getHealth();
    if (health.model) modelPanel.adoptExistingModel(health.model);
  } catch {
    // Already reported by loadCapabilities.
  }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
else start();

export { store };
