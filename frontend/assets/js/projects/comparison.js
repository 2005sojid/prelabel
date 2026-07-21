/**
 * The comparison panel: two annotation sets, and where they disagree.
 *
 * Deliberately neutral about which set is right. "Baseline" is whatever you
 * captured or imported; "current" is what the model says now. Whether a
 * disagreement means the model is wrong or the labels are depends on why you
 * are looking, and the tool should not pretend to know.
 */

import * as api from "../api.js";
import { $, el, replace, setText, show } from "../dom.js";
import { colorFor } from "../colors.js";
import * as toast from "../toast.js";

/** How each kind of difference is described and coloured. */
export const KINDS = {
  agreed: { label: "agreed", css: "kind-agreed", hint: "same object, same class" },
  reclassified: { label: "reclassified", css: "kind-reclassified", hint: "same object, different class" },
  missing: { label: "only in baseline", css: "kind-missing", hint: "gone from the current set" },
  added: { label: "only in current", css: "kind-added", hint: "not in the baseline" },
};

let summary = null;

export const hasComparison = () => Boolean(summary?.available);

export async function load(projectId) {
  try {
    summary = await api.getComparison(projectId);
  } catch {
    summary = null;
  }
  render();
  return summary;
}

export function reset() {
  summary = null;
  render();
}

// --- rendering --------------------------------------------------------------

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function render() {
  const panel = $("compare-panel");
  const badge = $("compare-badge");
  if (!panel) return;

  if (!summary?.available) {
    show(panel, false);
    setText(badge, summary?.baseline_items ? "baseline ready" : "");
    show(badge, Boolean(summary?.baseline_items));
    return;
  }

  show(panel, true);
  show(badge, true);
  setText(badge, `${percent(summary.agreement)} agreement`);
  badge.className = `compare-badge ${summary.disputed ? "conflicted" : "clean"}`;

  replace(panel, [
    headline(),
    counts(),
    swaps(),
    classes(),
  ].filter(Boolean));
}

function headline() {
  return el("div", { className: "compare-headline" }, [
    el("b", { text: `${summary.items_disputed}` }),
    ` of ${summary.items} images differ · `,
    el("b", { text: percent(summary.agreement) }),
    " of objects agree",
  ]);
}

function counts() {
  const row = el("div", { className: "compare-counts" });
  for (const [kind, meta] of Object.entries(KINDS)) {
    const value = summary.counts[kind] ?? 0;
    row.append(
      el("span", { className: `compare-chip ${meta.css}`, attrs: { title: meta.hint } }, [
        el("b", { text: String(value) }),
        ` ${meta.label}`,
      ]),
    );
  }
  return row;
}

/**
 * A titled block of name→count rows.
 *
 * The rows wrap into columns rather than filling the width. On a wide screen a
 * full-width row puts the count a thousand pixels from the name it belongs to,
 * which is unreadable — and eight of them push the gallery off the fold.
 */
function block(title, rows) {
  if (!rows.length) return null;
  return el("div", { className: "compare-block" }, [
    el("div", { className: "compare-block-title", text: title }),
    el("div", { className: "compare-rows" }, rows),
  ]);
}

function swaps() {
  return block("Most common swaps", (summary.reclassifications || []).slice(0, 6).map((entry) =>
    el("div", { className: "compare-row" }, [
      el("span", { className: "compare-swap", text: entry.swap }),
      el("span", { className: "compare-count", text: String(entry.count) }),
    ]),
  ));
}

function classes() {
  const disputed = (summary.by_class || []).filter((entry) => entry.disputed > 0).slice(0, 8);
  return block("Where they disagree", disputed.map((entry) =>
    el("div", { className: "compare-row", attrs: { title: `${entry.agreed} agreed` } }, [
      el("span", { className: "swatch", style: { background: colorFor(entry.class_name) } }),
      el("span", { className: "compare-swap", text: entry.class_name }),
      el("span", { className: "compare-count", text: String(entry.disputed) }),
    ]),
  ));
}

// --- actions ----------------------------------------------------------------

export async function capture(projectId) {
  try {
    const result = await api.captureBaseline(projectId);
    toast.info(
      "Baseline captured",
      `${result.captured} images frozen. Load another model and re-run to see the difference.`,
    );
    summary = { ...result.comparison, available: true, baseline_items: result.captured };
    render();
    return true;
  } catch (error) {
    toast.error("Could not capture a baseline", error.message);
    return false;
  }
}

export async function clear(projectId) {
  try {
    await api.clearBaseline(projectId);
    toast.info("Baseline cleared", "The current annotations were not touched.");
    reset();
    return true;
  } catch (error) {
    toast.error("Could not clear the baseline", error.message);
    return false;
  }
}

/** Load corrected labels as the set to compare against, rather than over the model's. */
export async function importAsBaseline(projectId, file) {
  const result = await api.importAnnotations(projectId, file, { into: "baseline" });
  summary = { ...result.comparison, available: true, baseline_items: result.matched };
  render();
  return result;
}
