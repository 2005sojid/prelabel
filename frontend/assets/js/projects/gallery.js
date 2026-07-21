/**
 * The project gallery.
 *
 * Unlike the drag-and-drop batch view, the images live on the server and the
 * result set can be enormous, so this pages: the server sorts and filters, and
 * the browser holds one page at a time. Thumbnails are ordinary `<img>` tags
 * pointed at the cached thumbnail endpoint, which lets the browser handle
 * decoding, caching and eviction instead of us keeping bitmaps alive.
 */

import * as api from "../api.js";
import { $, clear, el, replace, setText, showEmpty } from "../dom.js";
import { colorFor } from "../colors.js";
import { confidence } from "../state.js";
import { visibleDetections } from "../detections.js";

const PAGE_SIZE = 120;

let projectId = null;
let items = [];
let offset = 0;
let total = 0;
let loading = false;
let onActivate = null;

export const currentItems = () => items;
export const setActivateHandler = (handler) => { onActivate = handler; };

export function reset(id) {
  projectId = id;
  items = [];
  offset = 0;
  total = 0;
  clear($("project-gallery"));
}

export function query() {
  return {
    order: $("project-sort").value,
    only: $("project-filter").value,
    search: $("project-search").value.trim(),
  };
}

/** Load the first page, replacing whatever is shown. */
export async function load(id = projectId) {
  reset(id);
  await loadMore();
}

/** Append the next page. Safe to call repeatedly; ignores overlapping calls. */
export async function loadMore() {
  if (loading || !projectId) return;
  if (total && offset >= total) return;
  loading = true;

  try {
    const body = await api.listItems(projectId, { offset, limit: PAGE_SIZE, ...query() });
    total = body.stats.total;
    if (!body.items.length && offset === 0) {
      showEmpty($("project-gallery"), "Nothing matches this filter");
      return;
    }
    offset += body.items.length;
    for (const item of body.items) {
      items.push(item);
      $("project-gallery").append(card(item, items.length - 1));
    }
  } catch (error) {
    if (offset === 0) showEmpty($("project-gallery"), `❌ ${error.message}`);
  } finally {
    loading = false;
  }
}

const visibleCount = (item) => visibleDetections(item.detections, confidence()).length;

function card(item, index) {
  const canvas = el("canvas");
  const image = el("img", {
    className: "gcard-img",
    attrs: { loading: "lazy", decoding: "async", alt: item.name },
    src: api.itemImageUrl(projectId, item.id, { thumb: true }),
  });

  const node = el("div", { className: `gcard ${item.status === "pending" ? "pending" : ""}` }, [
    image,
    canvas,
    el("div", { className: "meta" }, [
      el("span", { className: "fname", text: item.name, attrs: { title: item.rel_path } }),
      // Count what the *current* threshold shows, not what was stored at the
      // inference floor — otherwise a card claims 9 while opening it shows 4.
      el("span", {
        className: `cnt ${visibleCount(item) ? "" : "zero"}`,
        text: item.status === "error" ? "err" : String(visibleCount(item)),
      }),
    ]),
  ]);

  if (item.width && item.height) node.style.setProperty("--ar", (item.width / item.height).toFixed(3));
  if (item.status === "error") {
    node.append(el("span", { className: "err", text: "⚠", attrs: { title: item.detail || "failed" } }));
  }

  node.addEventListener("click", () => onActivate?.(index));
  // Boxes are drawn once the browser has the thumbnail's real dimensions.
  image.addEventListener("load", () => drawOverlay(node, item, image));
  return node;
}

/**
 * Draw the boxes on the canvas layered over the thumbnail.
 *
 * The overlay is a separate canvas rather than a redraw of the image, so the
 * confidence slider can repaint annotations without re-decoding anything.
 */
function drawOverlay(node, item, image) {
  const canvas = node.querySelector("canvas");
  const width = image.clientWidth;
  const height = image.clientHeight;
  if (!width || !height || !item.width) return;

  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // The thumbnail is drawn with `object-fit: cover`, which scales to fill the
  // cell and crops the overflow. The overlay has to reproduce exactly that —
  // a plain width ratio would leave every box offset by half the cropped edge.
  const scale = Math.max(canvas.width / item.width, canvas.height / item.height);
  const offsetX = (canvas.width - item.width * scale) / 2;
  const offsetY = (canvas.height - item.height * scale) / 2;

  ctx.lineWidth = Math.max(1, 1.5 * ratio);
  for (const detection of visibleDetections(item.detections, confidence())) {
    if (!detection.box) continue;
    const [x1, y1, x2, y2] = detection.box;
    ctx.strokeStyle = colorFor(detection.class_name);
    ctx.strokeRect(
      offsetX + x1 * scale,
      offsetY + y1 * scale,
      (x2 - x1) * scale,
      (y2 - y1) * scale,
    );
  }
}

/** Repaint every visible card — used when the confidence threshold moves. */
export function repaint() {
  for (const [index, node] of [...$("project-gallery").children].entries()) {
    const image = node.querySelector("img");
    const item = items[index];
    if (item && image?.complete) drawOverlay(node, item, image);
    if (item && item.status !== "error") {
      const counter = node.querySelector(".cnt");
      if (counter) {
        const count = visibleCount(item);
        setText(counter, String(count));
        counter.classList.toggle("zero", count === 0);
      }
    }
  }
}

export function renderStats(project) {
  const stats = project.stats;
  const parts = [
    el("b", { text: String(stats.total) }),
    " images · ",
    el("b", { text: String(stats.done) }),
    " done",
  ];
  if (stats.failed) parts.push(el("span", { className: "sub", text: ` · ${stats.failed} failed` }));
  // "found", not "shown": this is the project-wide total at the inference floor,
  // while the cards show what survives the current threshold.
  if (stats.detections) parts.push(` · ${stats.detections} found`);
  if (stats.average_ms) parts.push(el("span", { className: "sub", text: ` · ~${stats.average_ms} ms/img` }));
  replace($("project-stat"), parts);

  const done = stats.total ? (stats.done + stats.failed) / stats.total : 0;
  $("project-progress").firstElementChild.style.width = `${Math.round(done * 100)}%`;
}

export function renderClasses(classes) {
  const list = $("det-list");
  if (!classes?.length) {
    showEmpty(list, "No detections yet");
    return;
  }
  replace(
    list,
    classes.map((entry) =>
      el("div", { className: "det" }, [
        el("span", { className: "swatch", style: { background: colorFor(entry.class_name) } }),
        el("span", { className: "name", text: entry.class_name }),
        el("span", { className: "conf", text: String(entry.count) }),
      ]),
    ),
  );
}

/** Load another page when the user scrolls near the end. */
export function initInfiniteScroll() {
  const gallery = $("project-gallery");
  gallery.addEventListener("scroll", () => {
    const remaining = gallery.scrollHeight - gallery.scrollTop - gallery.clientHeight;
    if (remaining < 600) loadMore();
  });
}
