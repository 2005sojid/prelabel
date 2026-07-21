/**
 * The retrain panel: fine-tune the loaded model on this project's labels.
 *
 * The far end of the loop. Everything else here gets a model's guesses in front
 * of a person to correct; this teaches the model from the corrections and hands
 * back a new checkpoint to adopt, re-run, and diff against the old one.
 *
 * Like {@link module:projects/comparison}, this module owns the *status* it
 * renders into ``#train-panel`` and the API calls behind the buttons; the
 * buttons themselves live in ``#train-bar`` and are wired from the project view.
 */

import * as api from "../api.js";
import { $, el, replace, setText, show } from "../dom.js";
import * as toast from "../toast.js";

let state = null;
let projectId = null;

/** Whether a fine-tune is in flight, so the view knows to keep polling. */
export const isRunning = () => Boolean(state && (state.active || state.status === "running"));

/** Whether there is a finished checkpoint ready to adopt. */
export const canAdopt = () => Boolean(state && state.status === "done" && state.weights);

export const status = () => state?.status || "";

export async function load(id) {
  projectId = id;
  try {
    state = await api.getTraining(id);
  } catch {
    state = null;
  }
  render();
  return state;
}

export function reset() {
  state = null;
  projectId = null;
  render();
}

// --- actions ----------------------------------------------------------------

export async function start(id, settings) {
  try {
    const response = await api.startTraining(id, settings);
    state = { ...response.training, active: true };
    render();
    toast.info("Fine-tuning started", "Training on this project's labels. This can take a while.");
    return true;
  } catch (error) {
    toast.error("Could not start training", error.message);
    return false;
  }
}

export async function cancel(id) {
  try {
    await api.cancelTraining(id);
    toast.info("Stopping training", "It will halt at the end of the current epoch.");
    return true;
  } catch (error) {
    toast.error("Could not stop training", error.message);
    return false;
  }
}

/** Load the retrained weights as the active model. Returns the model info, or null. */
export async function adopt(id) {
  try {
    const response = await api.adoptRetrained(id);
    toast.info("Retrained model adopted", `${response.model?.name} is now loaded. Re-run to see the difference.`);
    return response.model || null;
  } catch (error) {
    toast.error("Could not adopt the model", error.message);
    return null;
  }
}

// --- rendering --------------------------------------------------------------

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

const METRICS = [
  { key: "map50", label: "mAP@50" },
  { key: "map", label: "mAP@50-95" },
  { key: "precision", label: "precision" },
  { key: "recall", label: "recall" },
];

function render() {
  const panel = $("train-panel");
  const badge = $("train-badge");
  if (!panel) return;

  const s = state;
  if (!s || !s.status) {
    show(panel, false);
    show(badge, false);
    return;
  }

  show(badge, true);
  setText(badge, badgeText(s));
  badge.className = `compare-badge ${s.status === "failed" ? "conflicted" : ""}`;

  show(panel, true);
  replace(panel, blocks(s).filter(Boolean));
}

function badgeText(s) {
  if (s.status === "running") return `training · epoch ${s.epoch || 0}/${s.epochs || "?"}`;
  if (s.status === "done") {
    const map = s.metrics?.map50;
    return map != null ? `trained · mAP@50 ${percent(map)}` : "trained";
  }
  if (s.status === "failed") return "training failed";
  if (s.status === "cancelled") return "training stopped";
  return s.status;
}

function blocks(s) {
  if (s.status === "failed") {
    return [el("div", { className: "compare-headline", text: s.detail || "Training failed." })];
  }
  return [runningLine(s), datasetLine(s), metrics(s), sourceNote(s)];
}

function runningLine(s) {
  if (s.status !== "running") return null;
  const ratio = s.epochs ? (s.epoch || 0) / s.epochs : 0;
  return el("div", { className: "train-progress-wrap" }, [
    el("div", { className: "compare-headline", text: s.detail || `Epoch ${s.epoch || 0} of ${s.epochs}` }),
    el("div", { className: "train-bar-track" }, [
      el("div", { className: "train-bar-fill", style: { width: percent(ratio) } }),
    ]),
  ]);
}

function datasetLine(s) {
  const d = s.dataset;
  if (!d) return null;
  return el("div", { className: "compare-counts" }, [
    chip(`${d.images}`, "images"),
    chip(`${d.boxes}`, "boxes"),
    chip(`${d.train}/${d.val}`, "train/val"),
    chip(`${d.num_classes}`, d.num_classes === 1 ? "class" : "classes"),
  ]);
}

function chip(value, label) {
  return el("span", { className: "compare-chip" }, [el("b", { text: String(value) }), ` ${label}`]);
}

function metrics(s) {
  const values = s.metrics || {};
  const present = METRICS.filter((m) => values[m.key] != null);
  if (!present.length) return null;
  return el("div", { className: "compare-block" }, [
    el("div", { className: "compare-block-title", text: s.status === "done" ? "On the held-out split" : "So far" }),
    el("div", { className: "compare-rows" }, present.map((m) =>
      el("div", { className: "compare-row" }, [
        el("span", { className: "compare-swap", text: m.label }),
        el("span", { className: "compare-count", text: percent(values[m.key]) }),
      ]),
    )),
  ]);
}

function sourceNote(s) {
  if (s.status !== "done") return null;
  const from = s.source === "baseline" ? "the baseline labels" : "the current labels";
  return el("div", { className: "compare-block-title", text: `Fine-tuned from ${from}. Adopt it, then re-run to compare.` });
}
