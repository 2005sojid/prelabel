/**
 * Picking a folder on the server.
 *
 * The images never leave the machine they are on, so a project is created by
 * naming a directory rather than uploading anything. That directory has to be
 * inside a configured root, and the picker only ever shows what is allowed —
 * so an unreachable path is impossible to choose rather than an error to hit.
 */

import * as api from "../api.js";
import { $, clear, el, replace, setText, show } from "../dom.js";
import * as toast from "../toast.js";

let current = { path: "", imageCount: 0 };
let resolvePick = null;

/**
 * Open the picker. Resolves to `{path, imageCount, name}`, or null if cancelled.
 *
 * The name is asked for here rather than through `window.prompt`: a native
 * prompt blocks the whole page, cannot be styled, and cannot show the folder it
 * is naming. One field in the dialog you are already looking at is better on all
 * three counts.
 */
export function pickFolder() {
  return new Promise((resolve) => {
    resolvePick = resolve;
    $("folder-name").value = "";
    show($("folder-dialog"), true);
    navigate("");
  });
}

function close(result) {
  show($("folder-dialog"), false);
  const resolver = resolvePick;
  resolvePick = null;
  resolver?.(result);
}

async function navigate(path) {
  const list = $("folder-list");
  replace(list, el("p", { className: "empty", text: "Loading…" }));

  let body;
  try {
    body = await api.browseFolder(path);
  } catch (error) {
    replace(list, el("p", { className: "empty", text: `❌ ${error.message}` }));
    return;
  }

  current = { path: body.path, imageCount: body.image_count };
  setText($("folder-path"), body.path || "Allowed roots");
  setText(
    $("folder-count"),
    body.path ? `${body.image_count} image${body.image_count === 1 ? "" : "s"} here` : "",
  );
  $("folder-choose").disabled = !body.path || body.image_count === 0;

  // Suggest the folder's own name, but never overwrite something typed already.
  const nameField = $("folder-name");
  if (body.path && !nameField.dataset.touched) {
    nameField.value = body.path.split(/[\\/]/).filter(Boolean).pop() || "Project";
  }

  clear(list);
  if (body.parent) {
    list.append(row("⬆", "Up one level", () => navigate(body.parent)));
  }
  if (!body.directories.length && !body.path) {
    replace(
      list,
      el("p", {
        className: "empty",
        text: "No dataset roots configured. Set PL_DATA_ROOTS to the folders this server may read.",
      }),
    );
    return;
  }
  for (const entry of body.directories) {
    list.append(row("📁", entry.name, () => navigate(entry.path)));
  }
  if (!body.directories.length) {
    list.append(el("p", { className: "empty", text: "No subfolders — choose this one." }));
  }
}

function row(icon, label, onClick) {
  return el("button", { className: "folder-row", on: { click: onClick } }, [
    el("span", { className: "folder-icon", text: icon }),
    el("span", { className: "folder-name", text: label }),
  ]);
}

export function initFolderBrowser() {
  const nameField = $("folder-name");

  const submit = () => {
    if (!current.path || !current.imageCount) return;
    close({
      path: current.path,
      imageCount: current.imageCount,
      name: nameField.value.trim() || current.path.split(/[\\/]/).filter(Boolean).pop() || "Project",
    });
  };

  nameField.addEventListener("input", () => { nameField.dataset.touched = "1"; });
  nameField.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submit();
  });

  $("folder-cancel").addEventListener("click", () => close(null));
  $("folder-choose").addEventListener("click", submit);
  $("folder-dialog").addEventListener("click", (event) => {
    if (event.target === $("folder-dialog")) close(null);
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("folder-dialog").hidden) close(null);
  });
}

/** Tell the user how to switch the feature on, once, when it is off. */
export async function warnIfUnavailable() {
  try {
    const roots = await api.datasetRoots();
    if (!roots.configured) {
      toast.info(
        "Folder projects are off",
        "Set PL_DATA_ROOTS to the directories this server may read, then restart. " +
          "Drag & drop still works without it.",
        12000,
      );
    }
    return roots.configured;
  } catch {
    return false;
  }
}
