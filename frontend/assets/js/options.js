/**
 * Inference options shared by the single-image, batch and video paths.
 *
 * These exist as controls rather than API-only parameters because a feature the
 * UI cannot reach is a feature nobody uses: sliced inference and class filtering
 * were both implemented and both invisible until they had a checkbox.
 */

import { $, setText } from "./dom.js";
import { store } from "./state.js";

/** Class ids to keep, as the server wants them: "0,2" or null for everything. */
export function classFilter() {
  const raw = $("opt-classes")?.value ?? "";
  const ids = raw
    .split(/[,;\s]+/)
    .map((part) => Number.parseInt(part, 10))
    .filter((value) => Number.isInteger(value) && value >= 0);
  return ids.length ? [...new Set(ids)].sort((a, b) => a - b).join(",") : null;
}

export const isTiled = () => Boolean($("opt-tiled")?.checked);
export const isTracking = () => Boolean($("opt-track")?.checked);

/** Everything the image endpoints accept, ready to drop into a form. */
export function imageOptions() {
  return { classes: classFilter(), tiled: isTiled() ? "true" : null };
}

export function videoOptions() {
  return { classes: classFilter(), track: isTracking() ? "true" : null };
}

/**
 * Tracking depends on the backend and on an optional package, so the checkbox
 * has to reflect what the loaded model can actually do rather than offer
 * something that will quietly do nothing.
 */
export function refreshAvailability() {
  const track = $("opt-track");
  const note = $("opt-track-note");
  if (!track) return;

  const capable = store.capabilities?.features?.trackers?.length > 0;
  const model = store.model;
  const usable = capable && model && model.task !== "classify";

  track.disabled = !usable;
  if (!model) setText(note, "load a model first");
  else if (!usable) setText(note, "not available for this model");
  else setText(note, "ids are drawn on the rendered clip");
}
