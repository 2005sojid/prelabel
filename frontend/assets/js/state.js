/**
 * Shared application state and a minimal event bus.
 *
 * Modules read `store` and react to events rather than calling into each other,
 * so adding a view means subscribing to events — not editing the modules that
 * already exist.
 *
 * Events:
 *   `model:loaded`   {model}   a model finished loading (or the device changed)
 *   `model:cleared`  {}        the model was unloaded
 *   `mode:changed`   {mode}    "single" | "batch"
 *   `conf:preview`   {conf}    slider is being dragged — cheap updates only
 *   `conf:commit`    {conf}    slider released — full redraw / re-run
 */

export const store = {
  /** "single" | "batch" */
  mode: "single",
  /** Model info from the server, or null. */
  model: null,
  /** Task of the active model — mirrors whatever is currently on screen. */
  task: "detect",
  /** Server limits and capabilities, from /api/formats. */
  capabilities: null,
  /** Device id currently selected in the UI ("cpu" | "cuda"). */
  device: null,
};

export const isModelReady = () => store.model !== null;

/** Confidence threshold as a 0–1 fraction, read from the slider. */
export function confidence() {
  const slider = document.getElementById("conf");
  return slider ? Number(slider.value) / 100 : 0.5;
}

const listeners = new Map();

export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
  return () => listeners.get(event).delete(handler);
}

export function emit(event, payload = {}) {
  for (const handler of listeners.get(event) || []) {
    try {
      handler(payload);
    } catch (error) {
      // One bad subscriber must not stop the others, and must not vanish.
      console.error(`Handler for "${event}" failed:`, error);
    }
  }
}

export function setModel(model) {
  store.model = model;
  store.task = model?.task || "detect";
  emit("model:loaded", { model });
}

export function clearModel() {
  store.model = null;
  emit("model:cleared", {});
}

export function setMode(mode) {
  if (store.mode === mode) return;
  store.mode = mode;
  emit("mode:changed", { mode });
}
