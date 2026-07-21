/**
 * The batch gallery: cards, lazy thumbnails, filtering, sorting and the stats bar.
 *
 * Thumbnails are decoded only as their card approaches the viewport. Decoding
 * every image up front would make a folder of a few thousand photos stall before
 * a single inference had run — and inference, not decoding, is the point.
 */

import { $, clear, el, replace, setText, showEmpty } from "../dom.js";
import { colorFor } from "../colors.js";
import { confidence } from "../state.js";
import { doneItems, items, summarise, visibleFor } from "./items.js";

const THUMB_MAX_EDGE = 360;
const OBSERVER_MARGIN = "400px";

/** Indices of the items currently on screen, in display order. */
export let visibleOrder = [];

let observer = null;

// --- construction -----------------------------------------------------------

export function buildGallery() {
  setupObserver();
  const gallery = clear($("gallery"));
  gallery.className = $("batch-size").value;

  for (const [index, item] of items.entries()) {
    const canvas = el("canvas");
    const count = el("span", { className: "cnt", text: "…" });
    const card = el("div", { className: "gcard pending" }, [
      canvas,
      el("div", { className: "meta" }, [
        el("span", { className: "fname", text: item.name, attrs: { title: item.path } }),
        count,
      ]),
    ]);
    card.addEventListener("click", () => onCardActivate?.(index));

    item.card = card;
    item.canvas = canvas;
    item.countEl = count;
    card._item = item;

    gallery.append(card);
    observer.observe(card);
  }
  visibleOrder = items.map((_, index) => index);
}

let onCardActivate = null;
export const setCardHandler = (handler) => { onCardActivate = handler; };

export function destroyGallery() {
  observer?.disconnect();
  observer = null;
  visibleOrder = [];
  clear($("gallery"));
}

// --- thumbnails -------------------------------------------------------------

function setupObserver() {
  observer?.disconnect();
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && entry.target._item) loadThumbnail(entry.target._item);
      }
    },
    { root: $("gallery"), rootMargin: OBSERVER_MARGIN },
  );
}

/**
 * Decode and downscale off the main thread. `createImageBitmap` is markedly
 * faster than an `Image` plus `canvas.toDataURL`, which serialises and blocks.
 */
async function makeThumbnail(file) {
  const full = await createImageBitmap(file);
  const scale = Math.min(1, THUMB_MAX_EDGE / Math.max(full.width, full.height));
  const thumb = await createImageBitmap(full, {
    resizeWidth: Math.max(1, Math.round(full.width * scale)),
    resizeHeight: Math.max(1, Math.round(full.height * scale)),
    resizeQuality: "medium",
  });
  const size = { width: full.width, height: full.height };
  full.close();
  return { thumb, ...size };
}

async function loadThumbnail(item) {
  if (item.thumb || item.thumbLoading) return;
  item.thumbLoading = true;
  try {
    const { thumb, width, height } = await makeThumbnail(item.file);
    item.thumb = thumb;
    if (!item.width) {
      item.width = width;
      item.height = height;
      applyAspect(item);
    }
    drawCard(item);
  } catch {
    // Unreadable file: the card stays blank and inference reports the error.
  } finally {
    item.thumbLoading = false;
  }
}

export function applyAspect(item) {
  if (item.card && item.width && item.height) {
    item.card.style.setProperty("--ar", (item.width / item.height).toFixed(3));
  }
}

// --- painting ---------------------------------------------------------------

const devicePixelRatio = () => window.devicePixelRatio || 1;

export function drawCard(item) {
  const canvas = item.canvas;
  if (!item.thumb || !canvas || !canvas.clientWidth) return; // not laid out yet

  const ratio = devicePixelRatio();
  const width = (canvas.width = canvas.clientWidth * ratio);
  const height = (canvas.height = canvas.clientHeight * ratio);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  // Cover-fit: the cell already matches the image aspect ratio, so cropping is
  // negligible and we avoid letterbox bars.
  const scale = Math.max(width / item.width, height / item.height);
  const offsetX = (width - item.width * scale) / 2;
  const offsetY = (height - item.height * scale) / 2;
  ctx.drawImage(item.thumb, offsetX, offsetY, item.width * scale, item.height * scale);

  for (const detection of visibleFor(item, confidence())) {
    if (!detection.box) continue;
    const [x1, y1, x2, y2] = detection.box;
    ctx.strokeStyle = colorFor(detection.class_name);
    ctx.lineWidth = Math.max(1, 1.5 * ratio);
    ctx.strokeRect(offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
  }
}

export function markCardError(item, detail) {
  item.status = "error";
  if (!item.card || item.card.querySelector(".err")) return;
  item.card.append(el("span", { className: "err", text: "⚠", attrs: { title: detail || "failed" } }));
}

export function clearCardError(item) {
  item.card?.querySelector(".err")?.remove();
}

export function updateCardCount(item) {
  const count = visibleFor(item, confidence()).length;
  item.visibleCount = count;
  if (!item.countEl) return;
  setText(item.countEl, item.status === "error" ? "err" : count);
  item.countEl.classList.toggle("zero", count === 0);
}

export function setProgress(done, total) {
  $("batch-progress").firstElementChild.style.width = `${(100 * done) / Math.max(1, total)}%`;
}

// --- filter / sort ----------------------------------------------------------

export function filterSort({ redraw = true } = {}) {
  const filter = $("batch-filter").value;
  const sort = $("batch-sort").value;
  const query = $("batch-search").value.trim().toLowerCase();

  const order = items
    .map((_, index) => index)
    .filter((index) => {
      const item = items[index];
      const count = item.visibleCount || 0;
      const passesFilter =
        filter === "all" || (filter === "with" && count > 0) || (filter === "without" && count === 0);
      const passesSearch = !query || item.name.toLowerCase().includes(query);
      return passesFilter && passesSearch;
    });

  order.sort((a, b) => {
    if (sort === "name") return items[a].name.localeCompare(items[b].name);
    if (sort === "most") return (items[b].visibleCount || 0) - (items[a].visibleCount || 0);
    return (items[a].visibleCount || 0) - (items[b].visibleCount || 0);
  });

  visibleOrder = order;
  const gallery = $("gallery");
  for (const item of items) if (item.card) item.card.hidden = true;
  for (const index of order) {
    const item = items[index];
    if (!item.card) continue;
    item.card.hidden = false;
    gallery.append(item.card); // reorder in place
    if (redraw && item.status === "done") drawCard(item);
  }
}

export function redrawVisible() {
  for (const index of visibleOrder) {
    const item = items[index];
    if (item?.status === "done") drawCard(item);
  }
}

// --- stats ------------------------------------------------------------------

export function renderStats() {
  const threshold = confidence();
  const stats = summarise(threshold);

  const parts = [
    el("b", { text: String(stats.count) }),
    " images · ",
    el("b", { text: String(stats.total) }),
    ` detections · ${stats.perClass.length} classes`,
  ];
  if (stats.averageMs) {
    parts.push(
      el("span", {
        attrs: { title: "average model inference per image" },
        text: ` · ~${stats.averageMs.toFixed(1)} ms/img`,
      }),
    );
  }
  if (stats.processed < stats.count) {
    parts.push(el("span", { className: "sub", text: ` · processing ${stats.processed}/${stats.count}` }));
  }
  replace($("batch-stat"), parts);

  renderClassBreakdown(stats);
}

function renderClassBreakdown(stats) {
  const list = $("det-list");
  if (!stats.total) {
    showEmpty(list, stats.processed ? "No detections at this confidence" : "Processing…");
    return;
  }

  replace(list, [
    el("div", { className: "det summary" }, [
      el("span", {
        className: "name",
        text: `${stats.imagesWithDetections}/${stats.count} images have detections`,
      }),
    ]),
    ...stats.perClass.map(([name, count]) =>
      el("div", { className: "det" }, [
        el("span", { className: "swatch", style: { background: colorFor(name) } }),
        el("span", { className: "name", text: name }),
        el("span", { className: "conf", text: String(count) }),
      ]),
    ),
  ]);
}

export function statusMessage(message) {
  replace($("batch-stat"), message);
}

export const completedCount = () => doneItems().length;
