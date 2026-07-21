/**
 * Project mode: create, run, review, export.
 *
 * The run happens on the server, so this module is mostly a view: it starts
 * work, polls while it is going, and stops polling when it is not. Closing the
 * page does not stop anything — reopening it picks the project back up.
 */

import * as api from "../api.js";
import { $, el, replace, setText, show } from "../dom.js";
import { adoptExistingModel } from "../model-panel.js";
import { confidence, on, setMode, store } from "../state.js";
import * as toast from "../toast.js";
import { initFolderBrowser, pickFolder, warnIfUnavailable } from "./browser.js";
import * as compare from "./comparison.js";
import * as gallery from "./gallery.js";
import * as lightbox from "./lightbox.js";
import * as train from "./training.js";

/** How often to refresh a running project. */
const POLL_MS = 1200;

let active = null;      // the open project, as returned by the API
let pollTimer = 0;
let available = false;

export const isActive = () => store.mode === "project";
export const activeProject = () => active;

export { gallery };

// --- opening ----------------------------------------------------------------

/**
 * The open project lives in the URL fragment.
 *
 * A run outlives the page, so reloading should not drop you back to an empty
 * screen — and a project you can link to is one you can send to a colleague.
 */
const projectFromUrl = () => new URLSearchParams(location.hash.slice(1)).get("project");

function rememberInUrl(id) {
  const hash = id ? `#project=${encodeURIComponent(id)}` : "";
  if (location.hash !== hash) history.replaceState(null, "", hash || location.pathname);
}

export async function openProject(id) {
  disarmDelete();
  rememberInUrl(id);
  setMode("project");
  show($("project-view"), true);
  show($("hint"), false);
  show($("badge"), false);
  $("canvas").hidden = true;
  show($("video"), false);
  show($("batch-view"), false);

  await refresh(id);
  await gallery.load(id);
}

export function closeProject() {
  stopPolling();
  compare.reset();
  train.reset();
  active = null;
  rememberInUrl(null);
  lightbox.close();
  show($("project-view"), false);
  setMode("single");
  show($("hint"), true);
  $("canvas").hidden = false;
  renderProjectList();
}

async function refresh(id = active?.id) {
  if (!id) return;
  try {
    active = await api.getProject(id);
  } catch (error) {
    toast.error("Could not load the project", error.message);
    return;
  }
  setText($("project-name"), active.name);
  setText($("project-source"), active.source_dir);
  gallery.renderStats(active);
  gallery.renderClasses(active.classes);
  renderExportOptions();
  await compare.load(active.id);
  renderCompareControls();
  await train.load(active.id);
  renderTrainControls();
  // Last, so it reads fresh training state: the run button and status line both
  // depend on whether a fine-tune is in flight, and that is only known now.
  renderRunState();
}

function renderTrainControls() {
  const running = train.isRunning();
  show($("train-start"), !running);
  show($("train-cancel"), running);
  show($("train-adopt"), train.canAdopt());
  // The knobs are meaningless mid-run, and re-reading them would fight the server.
  $("train-epochs").disabled = running;
  $("train-source").disabled = running;
}

/** Options that only mean something once there is a second set to compare with. */
const COMPARE_ONLY = { "project-sort": ["disputed"], "project-filter": ["disputed", "agreed"] };

function renderCompareControls() {
  const ready = compare.hasComparison();
  show($("compare-clear"), ready);
  show($("compare-overlay-wrap"), ready);
  setText($("compare-capture"), ready ? "Recapture baseline" : "Capture baseline");

  // Without a baseline these sort and filter by a column that is all zeros —
  // an option that silently does nothing is worse than one you cannot pick.
  for (const [id, values] of Object.entries(COMPARE_ONLY)) {
    const select = $(id);
    for (const value of values) {
      const option = select.querySelector(`option[value="${value}"]`);
      if (option) option.disabled = !ready;
    }
    if (!ready && values.includes(select.value)) {
      select.value = select.options[0].value;
      select.dispatchEvent(new Event("change"));
    }
  }
}

// --- run control ------------------------------------------------------------

/** Poll while either the run or the training is in flight — both progress on the server. */
const shouldPoll = () => Boolean(active?.running) || train.isRunning();

function renderRunState() {
  const running = Boolean(active?.running);
  const button = $("project-run");
  const training = train.isRunning();
  setText(button, running ? "⏹ Stop" : active?.stats.pending ? "▶ Run" : "↻ Re-run");
  button.classList.toggle("primary", !running);
  // Running the model and training on it share the one GPU, so the run button
  // stands down while a fine-tune is going rather than starting a doomed run.
  button.disabled = training;

  setText($("project-status"), training ? "Training…" : statusLabel(active));
  $("project-status").className = `project-status ${training ? "running" : active?.status || ""}`;

  if (shouldPoll()) startPolling();
  else stopPolling();
}

function statusLabel(project) {
  if (!project) return "";
  if (project.running) return `Running · ${project.stats.done}/${project.stats.total}`;
  return {
    new: "Not started",
    done: "Finished",
    cancelled: project.detail || "Stopped",
    failed: `Failed: ${project.detail}`,
  }[project.status] || project.status;
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const before = active?.stats.done ?? 0;
    await refresh();
    // Only reload the grid when new results actually landed, so scrolling a
    // finished part of the gallery is not yanked back to the top every second.
    if ((active?.stats.done ?? 0) !== before) await gallery.load(active.id);
    if (!shouldPoll()) stopPolling();
  }, POLL_MS);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = 0;
}

async function toggleRun() {
  if (!active) return;
  try {
    if (active.running) {
      await api.cancelProject(active.id);
      toast.info("Stopping", "The run will halt after the batch in flight.");
    } else {
      // A run resumes: it processes what is still pending. When nothing is,
      // resuming finishes instantly and changes nothing — so the button that
      // says "Re-run" has to clear the old results first, or it silently lies.
      const restart = !active.stats.pending;
      await api.runProject(active.id, { settings: readSettings(), restart });
    }
  } catch (error) {
    toast.error(active.running ? "Could not stop the run" : "Could not start the run", error.message);
  }
  await refresh();
}

function readSettings() {
  const classes = $("project-classes").value
    .split(",")
    .map((part) => Number.parseInt(part.trim(), 10))
    .filter((value) => Number.isInteger(value));
  return {
    conf: 0.05,          // infer low; the slider filters what is shown
    classes,
    tiled: $("project-tiled").checked,
  };
}

// --- export / import --------------------------------------------------------

function renderExportOptions() {
  const select = $("project-export-format");
  const formats = active?.export_formats || [{ id: "coco", label: "COCO 1.0" }];
  replace(select, formats.map((format) => el("option", { value: format.id, text: format.label })));
}

function downloadExport() {
  if (!active) return;
  const format = $("project-export-format").value;
  // A plain navigation: the archive is built on the server and streamed, so it
  // never has to fit in the page's memory.
  window.location.href = api.exportUrl(active.id, format, confidence());
}

/**
 * Load COCO annotations.
 *
 * @param {"current"|"baseline"} into  Overwrite the model's output, or sit
 *   beside it as the set to compare against. The second is how corrected labels
 *   become a way to find where the model — or the labelling — is wrong.
 */
function importAnnotations(into = "current") {
  if (!active) return;
  const input = el("input", { type: "file", attrs: { accept: ".json,application/json" } });
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const summary = into === "baseline"
        ? await compare.importAsBaseline(active.id, file)
        : await api.importAnnotations(active.id, file);
      toast.info(
        into === "baseline" ? "Baseline imported" : "Annotations imported",
        `${summary.matched} images matched, ${summary.annotations} annotations.` +
          (summary.unmatched ? ` ${summary.unmatched} file(s) had no match here.` : ""),
      );
      await refresh();
      await gallery.load(active.id);
    } catch (error) {
      toast.error("Import failed", error.message);
    }
  });
  input.click();
}

// --- the sidebar list -------------------------------------------------------

export async function renderProjectList() {
  const container = $("project-list");
  if (!available) {
    replace(container, el("p", { className: "empty", text: "Set PL_DATA_ROOTS to use projects" }));
    return;
  }

  let projects = [];
  try {
    projects = (await api.listProjects()).projects;
  } catch {
    replace(container, el("p", { className: "empty", text: "Could not load projects" }));
    return;
  }

  if (!projects.length) {
    replace(container, el("p", { className: "empty", text: "No projects yet" }));
    return;
  }

  replace(
    container,
    projects.map((project) =>
      el("button", {
        className: `project-row ${active?.id === project.id ? "active" : ""}`,
        on: { click: () => openProject(project.id) },
      }, [
        el("span", { className: "project-row-name", text: project.name }),
        el("span", {
          className: "project-row-meta",
          text: project.running
            ? `${project.stats.done}/${project.stats.total}…`
            : `${project.stats.total} img`,
        }),
      ]),
    ),
  );
}

async function createProject() {
  const chosen = await pickFolder();
  if (!chosen) return;

  try {
    const project = await api.createProject({ name: chosen.name, path: chosen.path });
    toast.info("Project created", `${project.stats.total} images registered.`);
    await renderProjectList();
    await openProject(project.id);
  } catch (error) {
    toast.error("Could not create the project", error.message);
  }
}

/** Milliseconds the delete button stays armed before reverting. */
const CONFIRM_WINDOW = 4000;
let disarmTimer = 0;

/**
 * Delete on the second click.
 *
 * A two-step button instead of `window.confirm`: a native confirm blocks the
 * page, cannot say what is and is not destroyed, and reads as an interruption.
 * Arming the button in place makes the consequence visible where the action is.
 */
async function removeProject() {
  if (!active) return;
  const button = $("project-delete");

  if (!button.dataset.armed) {
    button.dataset.armed = "1";
    button.classList.add("armed");
    setText(button, "🗑 Delete? (images kept)");
    disarmTimer = setTimeout(() => disarmDelete(), CONFIRM_WINDOW);
    return;
  }

  disarmDelete();
  try {
    await api.deleteProject(active.id);
    toast.info("Project deleted", "The source folder was left alone.");
    closeProject();
  } catch (error) {
    toast.error("Could not delete the project", error.message);
  }
}

function disarmDelete() {
  clearTimeout(disarmTimer);
  const button = $("project-delete");
  delete button.dataset.armed;
  button.classList.remove("armed");
  setText(button, "🗑");
}

// --- wiring -----------------------------------------------------------------

export async function initProjects() {
  initFolderBrowser();
  gallery.initInfiniteScroll();
  gallery.setActivateHandler((index) => lightbox.open(active, gallery.currentItems(), index));
  lightbox.initProjectLightbox();

  $("project-new").addEventListener("click", createProject);
  $("project-run").addEventListener("click", toggleRun);
  $("project-close").addEventListener("click", closeProject);
  $("project-delete").addEventListener("click", removeProject);
  $("project-export").addEventListener("click", downloadExport);
  $("project-import").addEventListener("click", () => importAnnotations("current"));
  $("compare-import").addEventListener("click", () => importAnnotations("baseline"));

  $("compare-capture").addEventListener("click", async () => {
    if (active && await compare.capture(active.id)) {
      renderCompareControls();
      await gallery.load(active.id);
    }
  });
  $("compare-clear").addEventListener("click", async () => {
    if (active && await compare.clear(active.id)) {
      renderCompareControls();
      await gallery.load(active.id);
    }
  });
  $("compare-overlay").addEventListener("change", (event) => {
    lightbox.toggleBaseline(event.target.checked);
  });

  $("train-start").addEventListener("click", async () => {
    if (!active) return;
    const settings = {
      epochs: Number.parseInt($("train-epochs").value, 10) || undefined,
      source: $("train-source").value,
    };
    if (await train.start(active.id, settings)) {
      renderRunState();
      renderTrainControls();
      startPolling();
    }
  });
  $("train-cancel").addEventListener("click", () => active && train.cancel(active.id));
  $("train-adopt").addEventListener("click", async () => {
    if (!active) return;
    const model = await train.adopt(active.id);
    if (model) {
      // Update the sidebar to the new model and let the "Re-run" nudge fire.
      adoptExistingModel(model);
      await refresh();
    }
  });

  const reload = () => active && gallery.load(active.id);
  $("project-sort").addEventListener("change", reload);
  $("project-filter").addEventListener("change", reload);

  let searchTimer = 0;
  $("project-search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(reload, 250);
  });

  // The slider filters what is displayed; the stored results do not change.
  on("conf:preview", () => isActive() && gallery.repaint());
  on("conf:commit", () => {
    if (!isActive()) return;
    gallery.repaint();
    lightbox.refresh();
  });

  // A *different* model makes stored results stale — say so rather than
  // silently showing predictions from a model that is no longer loaded.
  //
  // Compared against the project's own model snapshot, not fired on every
  // `model:loaded`: the page adopts whatever the server already had on startup,
  // and warning about that would cry wolf on every reload.
  on("model:loaded", ({ model }) => {
    if (!isActive() || !active?.stats.done) return;
    const ranWith = active.model || {};
    if (!ranWith.name || (model?.name === ranWith.name && model?.device === ranWith.device)) return;
    toast.info(
      "Model changed",
      `These results came from ${ranWith.name}. Press Re-run to redo them with ${model?.name}.`,
    );
  });

  available = await warnIfUnavailable();
  show($("projects-section"), available);
  await renderProjectList();

  const wanted = projectFromUrl();
  if (available && wanted) {
    try {
      await openProject(wanted);
    } catch {
      rememberInUrl(null); // the project was deleted since that link was made
    }
  }
}
