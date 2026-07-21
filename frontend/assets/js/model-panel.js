/**
 * Sidebar: loading a model, showing what loaded, switching device, benchmarking.
 *
 * The load flow's one non-obvious rule: a `waiting` response (an OpenVINO `.xml`
 * with no `.bin` yet) means *nothing changed on the server* — so the UI must keep
 * showing whatever was already loaded rather than assuming the new file took.
 * The server tells us what is still loaded; we mirror that instead of guessing.
 */

import * as api from "./api.js";
import { $, clear, el, replace, setText, show, wireDropZone } from "./dom.js";
import { clearModel, setModel, store } from "./state.js";
import * as toast from "./toast.js";

let modelExtensions = new Set([".pt", ".onnx", ".xml", ".bin", ".engine", ".torchscript", ".tflite", ".mlpackage", ".pb"]);

export const isModelFile = (file) => modelExtensions.has(extensionOf(file.name));

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

// --- capabilities -----------------------------------------------------------

export async function loadCapabilities() {
  try {
    const formats = await api.getFormats();
    store.capabilities = formats;
    setText($("model-formats"), (formats.model_formats || []).join(" "));
    modelExtensions = new Set([...(formats.model_formats || []), ".bin", ".yaml", ".yml"]);
    buildDeviceControl(formats.devices || [{ id: "cpu", label: "CPU" }]);
    return formats;
  } catch (error) {
    toast.error("Could not reach the server", error.message);
    return null;
  }
}

// --- loading ----------------------------------------------------------------

export async function loadModelFiles(files) {
  const drop = $("model-drop");
  drop.classList.remove("error", "ready");
  setText(drop, `⏳ Loading ${files.map((f) => f.name).join(", ")} …`);

  try {
    const response = await api.uploadModel(files, { device: store.device });

    if (response.status === "waiting") {
      // Nothing was replaced server-side. Restore the panel to whatever is
      // actually loaded, so the UI never claims a model it does not have.
      setText(drop, `📎 ${response.message}`);
      if (response.model) {
        applyModel(response.model, { announce: false });
      } else {
        clearModel();
        renderNoModel();
      }
      return response;
    }

    applyModel(response.model);
    return response;
  } catch (error) {
    drop.classList.add("error");
    setText(drop, `❌ ${error.message}`);
    clearModel();
    renderNoModel();
    toast.error("Model failed to load", error.message);
    return null;
  }
}

/**
 * Adopt a model the server had already loaded before this page opened, so a
 * refresh does not make a working model look absent.
 */
export function adoptExistingModel(model) {
  applyModel(model, { announce: false });
}

function applyModel(model, { announce = true } = {}) {
  const drop = $("model-drop");
  drop.classList.remove("error");
  drop.classList.add("ready");
  setText(drop, `✅ ${model.name}`);
  renderModelInfo(model);
  setModel(model);
  if (announce && model.task_assumed) {
    toast.warn(
      "Task was assumed",
      `This model does not record its task, so "${model.task}" was assumed. ` +
        "If it is segmentation, pose or classification, supply the model's metadata.yaml.",
    );
  }
}

function renderNoModel() {
  show($("model-info"), false);
  show($("device-wrap"), false);
  clearBenchmark();
}

// --- info card --------------------------------------------------------------

function renderModelInfo(model) {
  show($("model-info"), true);

  const task = $("mi-task");
  setText(task, model.task_assumed ? `${model.task} ⚠` : model.task);
  task.className = model.task_assumed ? "pill assumed" : "pill";
  task.title = model.task_assumed
    ? `Task could not be read from the model — assumed "${model.task}". Verify it matches your model.`
    : "";

  setText($("mi-format"), model.format);
  setText($("mi-backend"), model.backend);
  setText($("mi-device"), model.device);
  setText($("mi-classes"), model.num_classes);
  setText($("mi-imgsz"), model.imgsz);
  setText($("mi-assumed-task"), model.task);
  show($("mi-assumed"), Boolean(model.task_assumed));

  setText($("mi-path"), model.path || "");
  show($("mi-path"), Boolean(model.path));

  updateDeviceAvailability(model);
  clearBenchmark();
}

// --- device -----------------------------------------------------------------

function buildDeviceControl(devices) {
  const segment = clear($("device-seg"));
  for (const device of devices) {
    segment.append(
      el("button", {
        className: "btn",
        text: device.label,
        dataset: { id: device.id },
        on: { click: () => switchDevice(device.id) },
      }),
    );
  }
}

function updateDeviceAvailability(model) {
  // OpenVINO runs on its own CPU runtime here, so a GPU button would lie.
  const isOpenVino = (model.format || "").toLowerCase().includes("openvino");
  for (const button of $("device-seg").children) {
    button.hidden = isOpenVino && button.dataset.id !== "cpu";
  }
  show($("device-wrap"), $("device-seg").children.length > 0);
  markDevice((model.device || "").toLowerCase().startsWith("gpu") ? "cuda" : "cpu");
}

function markDevice(id) {
  store.device = id;
  for (const button of $("device-seg").children) {
    button.classList.toggle("active", button.dataset.id === id);
  }
}

async function switchDevice(id) {
  if (id === store.device || !store.model) {
    markDevice(id);
    return;
  }
  const buttons = [...$("device-seg").children];
  buttons.forEach((b) => (b.disabled = true));
  try {
    const response = await api.setDevice(id);
    renderModelInfo(response.model);
    setModel(response.model);
  } catch (error) {
    toast.error("Could not switch device", error.message);
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

// --- unload -----------------------------------------------------------------

async function unload() {
  try {
    await api.unloadModel();
  } catch (error) {
    toast.error("Could not unload the model", error.message);
    return;
  }
  clearModel();
  renderNoModel();
  const drop = $("model-drop");
  drop.classList.remove("ready", "error");
  replace(drop, ["Drop a model file", el("br"), el("small", { text: [...modelExtensions].join(" ") })]);
  toast.info("Model unloaded", "Memory has been released.");
}

// --- benchmark --------------------------------------------------------------

export function clearBenchmark() {
  const output = $("bench-out");
  show(output, false);
  clear(output);
}

function statRow(label, value, accent = false) {
  return el("div", { className: "row" }, [
    el("span", { className: "k", text: label }),
    el("span", { className: "v", text: value, style: accent ? { color: "var(--accent)" } : {} }),
  ]);
}

async function runBenchmark() {
  if (!store.model) return;
  const button = $("bench-btn");
  const output = $("bench-out");
  button.disabled = true;
  setText(button, "⏱ Benchmarking…");
  try {
    const result = await api.benchmark();
    show(output, true);
    replace(output, [
      statRow("Latency (1 image)", `${result.latency_ms} ms · ${result.latency_fps} img/s`),
      statRow("Throughput (many)", `${result.throughput_ms_per_image} ms/img · ${result.throughput_fps} img/s`, true),
      el("p", {
        className: "empty",
        style: { padding: "6px 0 0", textAlign: "left", lineHeight: "1.5" },
        text:
          "One image is latency-bound; many at once run faster per image " +
          `(GPU batching / OpenVINO parallel requests). On CPU the two are similar. ` +
          `Measured @${result.imgsz}px over ${result.runs} runs.`,
      }),
    ]);
  } catch (error) {
    show(output, true);
    replace(output, el("p", { className: "empty", text: `❌ ${error.message}` }));
  } finally {
    button.disabled = false;
    setText(button, "⏱ Benchmark (latency vs throughput)");
  }
}

// --- wiring -----------------------------------------------------------------

export function initModelPanel() {
  wireDropZone($("model-drop"), loadModelFiles, { multiple: true });
  $("bench-btn").addEventListener("click", runBenchmark);
  $("unload-btn").addEventListener("click", unload);
}
