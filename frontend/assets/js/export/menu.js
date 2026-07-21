/**
 * The export dropdown.
 *
 * The menu is built when it opens, from the task the completed results report —
 * so a pose model offers keypoints and a classifier offers ImageNet, without the
 * user choosing a format that cannot represent their results.
 */

import { $, clear, el, setText, show } from "../dom.js";
import { batchTask, doneItems } from "../batch/items.js";
import * as toast from "../toast.js";
import { EXPORTERS } from "./formats.js";
import { ZipLimitError } from "./zip.js";

function buildMenu(task) {
  const menu = clear($("export-menu"));
  const exporters = EXPORTERS[task] || EXPORTERS.detect;

  menu.append(el("div", { className: "grp", text: `Export as · task: ${task}` }));
  for (const exporter of exporters) {
    menu.append(
      el("button", { on: { click: () => runExport(exporter) } }, [
        exporter.label,
        el("small", { text: exporter.sub }),
      ]),
    );
  }
  return menu;
}

async function runExport(exporter) {
  show($("export-menu"), false);
  if (!doneItems().length) {
    toast.warn("Nothing to export yet", "Wait for at least one image to finish processing.");
    return;
  }

  const button = $("batch-export");
  const label = button.textContent;
  button.disabled = true;
  setText(button, "⏳ Exporting…");
  try {
    await exporter.run();
    toast.info("Export ready", `${doneItems().length} images written to a ${exporter.label} dataset.`);
  } catch (error) {
    if (error instanceof ZipLimitError) toast.warn("Export too large", error.message, 0);
    else toast.error("Export failed", error.message);
  } finally {
    button.disabled = false;
    setText(button, label);
  }
}

export function initExportMenu() {
  const button = $("batch-export");
  const menu = $("export-menu");

  button.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!menu.hidden) {
      show(menu, false);
      return;
    }
    buildMenu(batchTask());
    show(menu, true);
  });

  document.addEventListener("click", () => show(menu, false));
  menu.addEventListener("click", (event) => event.stopPropagation());
}
