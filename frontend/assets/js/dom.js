/**
 * DOM helpers.
 *
 * Every element here is built with `createElement` and `textContent`. Nothing in
 * this application interpolates data into `innerHTML` — filenames, class names
 * and server error strings all reach the page as text, so a folder containing
 * `<img src=x onerror=…>.jpg` renders as a filename rather than executing.
 */

/** Element by id. */
export const $ = (id) => document.getElementById(id);

/**
 * Create an element.
 *
 * @param {string} tag
 * @param {object} [props]   className, textContent, dataset, attrs, style, on
 * @param {Array}  [children]
 */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  const { className, text, dataset, attrs, style, on, ...rest } = props;

  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (dataset) Object.assign(node.dataset, dataset);
  if (style) Object.assign(node.style, style);
  if (attrs) for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (on) for (const [event, handler] of Object.entries(on)) node.addEventListener(event, handler);
  Object.assign(node, rest);

  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Remove every child of `node`. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** Replace a node's children in one step. */
export function replace(node, children) {
  clear(node);
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Show a single centred message inside a container. */
export function showEmpty(node, message) {
  return replace(node, el("p", { className: "empty", text: message }));
}

export const show = (node, visible = true) => { node.hidden = !visible; };
export const setText = (node, value) => { node.textContent = value === null || value === undefined ? "" : String(value); };

/**
 * True when the user is typing, so global single-key shortcuts should stand down.
 */
export function isTyping() {
  const active = document.activeElement;
  if (!active) return false;
  return active.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName);
}

/** Open a file picker and hand the selection to `onPick`. */
export function pickFiles({ multiple = false, accept = "", directory = false } = {}, onPick) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = multiple || directory;
  if (accept) input.accept = accept;
  if (directory) input.webkitdirectory = true;
  input.addEventListener("change", (event) => {
    const files = [...event.target.files];
    if (files.length) onPick(files);
  });
  input.click();
}

/**
 * Wire a drop zone: click to browse, drag to highlight, drop to accept.
 * Returns a teardown function.
 */
export function wireDropZone(node, onFiles, { multiple = true, accept = "" } = {}) {
  const open = () => pickFiles({ multiple, accept }, onFiles);

  const onClick = () => open();
  const onKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
  };
  const onDragOver = (event) => { event.preventDefault(); event.stopPropagation(); node.classList.add("over"); };
  const onDragLeave = () => node.classList.remove("over");
  const onDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    node.classList.remove("over");
    const files = [...(event.dataTransfer?.files || [])];
    if (files.length) onFiles(files);
  };

  node.addEventListener("click", onClick);
  node.addEventListener("keydown", onKeyDown);
  node.addEventListener("dragover", onDragOver);
  node.addEventListener("dragleave", onDragLeave);
  node.addEventListener("drop", onDrop);

  return () => {
    node.removeEventListener("click", onClick);
    node.removeEventListener("keydown", onKeyDown);
    node.removeEventListener("dragover", onDragOver);
    node.removeEventListener("dragleave", onDragLeave);
    node.removeEventListener("drop", onDrop);
  };
}
