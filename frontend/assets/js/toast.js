/**
 * Non-blocking notifications.
 *
 * Replaces `alert()`, which blocks the page and cannot show more than one line —
 * a poor fit for messages like "the video rendered, but your browser can't play
 * this codec", where the user needs the detail *and* the app underneath.
 */

import { $, el } from "./dom.js";

const DEFAULT_TIMEOUT = 6000;
const STICKY_LEVELS = new Set(["error"]);

function stack() {
  return $("toasts");
}

/**
 * @param {"info"|"warn"|"error"} level
 * @param {string} title
 * @param {string} [message]
 * @param {number} [timeout]  ms; 0 keeps it until dismissed
 */
export function toast(level, title, message = "", timeout) {
  const host = stack();
  if (!host) return () => {};

  const close = el("span", { className: "toast-close", text: "✕", attrs: { role: "button", title: "Dismiss" } });
  const node = el("div", { className: `toast ${level}` }, [
    close,
    el("div", { className: "toast-title", text: title }),
    message ? el("div", { text: message }) : null,
  ]);

  const dismiss = () => node.remove();
  close.addEventListener("click", dismiss);
  host.append(node);

  const delay = timeout ?? (STICKY_LEVELS.has(level) ? 0 : DEFAULT_TIMEOUT);
  if (delay > 0) setTimeout(dismiss, delay);
  return dismiss;
}

export const info = (title, message, timeout) => toast("info", title, message, timeout);
export const warn = (title, message, timeout) => toast("warn", title, message, timeout);
export const error = (title, message, timeout) => toast("error", title, message, timeout);
