/**
 * Single-item mode: one image or one video.
 *
 * Every run is tagged with an id. A result whose tag is stale — because the user
 * changed the photo, the model, the device or the confidence meanwhile — is
 * dropped instead of painting itself over newer output.
 */

import * as api from "./api.js";
import { $, el, replace, setText, show, showEmpty } from "./dom.js";
import { colorFor } from "./colors.js";
import { imageOptions, isTracking, videoOptions } from "./options.js";
import { confidence, store } from "./state.js";
import * as toast from "./toast.js";
import * as viewport from "./viewport.js";

const VIDEO_PATTERN = /\.(mp4|avi|mov|mkv|webm)$/i;
const IMAGE_PATTERN = /\.(jpg|jpeg|png|bmp|webp|tiff)$/i;

export const isVideoFile = (file) => file.type.startsWith("video") || VIDEO_PATTERN.test(file.name);
export const isMediaFile = (file) =>
  file.type.startsWith("image") || file.type.startsWith("video") ||
  IMAGE_PATTERN.test(file.name) || VIDEO_PATTERN.test(file.name);

let currentFile = null;
let currentIsVideo = false;
let runId = 0;
let videoObjectUrl = null;

export const hasMedia = () => currentFile !== null;
export const isShowingVideo = () => currentIsVideo;

// --- loading ----------------------------------------------------------------

export async function loadMediaFile(file) {
  currentFile = file;
  currentIsVideo = isVideoFile(file);

  const drop = $("media-drop");
  drop.classList.add("ready");
  replace(drop, `${currentIsVideo ? "🎬 " : "🖼️ "}${file.name}`);
  show($("hint"), false);

  if (currentIsVideo) {
    prepareVideoElement();
    if (store.model) run();
    return;
  }

  show($("video"), false);
  $("canvas").hidden = false;
  try {
    viewport.showImage(await decodeImageFile(file));
  } catch (error) {
    toast.error("Could not read that image", error.message);
    return;
  }
  if (store.model) run();
}

function decodeImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("the file is not a readable image"));
    };
    image.src = url;
  });
}

function prepareVideoElement() {
  $("canvas").hidden = true;
  const video = $("video");
  show(video, true);
  releaseVideoUrl();
  video.removeAttribute("src");
}

function releaseVideoUrl() {
  if (videoObjectUrl) {
    URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = null;
  }
}

export function clearMedia() {
  currentFile = null;
  currentIsVideo = false;
  runId++;
  releaseVideoUrl();
  viewport.clearImage();
}

// --- running ----------------------------------------------------------------

export function run() {
  if (!currentFile || !store.model) return;
  runId++;
  return currentIsVideo ? runVideo() : runImage();
}

async function runImage() {
  const tag = ++runId;
  viewport.setResults([], store.task);
  viewport.render();
  show($("spinner"), true);

  try {
    const result = await api.predictImage(currentFile, confidence(), imageOptions());
    if (tag !== runId) return; // superseded by a newer run
    viewport.setResults(result.detections || [], result.task);
    viewport.render();
    showResults(result);
  } catch (error) {
    if (tag === runId) showEmpty($("det-list"), `❌ ${error.message}`);
  } finally {
    if (tag === runId) show($("spinner"), false);
  }
}

async function runVideo() {
  const tag = ++runId;
  const spinner = $("spinner");
  show(spinner, true);
  setText(spinner, "⟳ Processing video…");

  try {
    const result = await api.predictVideo(currentFile, confidence(), videoOptions());
    if (tag !== runId) return;

    releaseVideoUrl();
    videoObjectUrl = URL.createObjectURL(result.blob);
    const video = $("video");
    video.src = videoObjectUrl;
    video.play().catch(() => {}); // autoplay policy — the user can press play

    const tracked = result.tracked ? ", tracked" : "";
    showEmpty($("det-list"), `Annotated video ready ✓ (${result.frames} frames${tracked})`);
    reportVideoCaveats(result);
  } catch (error) {
    if (tag === runId) {
      showEmpty($("det-list"), `❌ ${error.message}`);
      toast.error("Video inference failed", error.message);
    }
  } finally {
    if (tag === runId) {
      show(spinner, false);
      setText(spinner, "⟳ Running inference…");
    }
  }
}

/** Tell the user what the server had to compromise on, if anything. */
function reportVideoCaveats(result) {
  if (isTracking() && !result.tracked) {
    toast.warn(
      "Tracking was not applied",
      "This model or backend cannot track, so the clip was annotated frame by frame " +
        "without object ids.",
    );
  }
  if (!result.browserPlayable) {
    toast.warn(
      "Video may not play in this browser",
      result.codecNote || `Encoded with ${result.codec}, which Chrome and Firefox cannot decode. ` +
        "The download still works in a desktop player.",
      0,
    );
  }
  if (result.sampled) {
    toast.info(
      "Video was sampled",
      `Rendered ${result.frames} frames spread evenly across the whole clip, to stay within the frame cap. ` +
        "Raise PL_MAX_VIDEO_FRAMES to process more.",
    );
  } else if (result.truncated) {
    toast.warn(
      "Video was truncated",
      `Only the first ${result.frames} frames were processed: this file does not report its length, ` +
        "so the frames could not be spread across the clip.",
    );
  }
}

// --- sidebar ----------------------------------------------------------------

/**
 * Render one inference result into the sidebar and the badge.
 * Shared with the webcam stream, whose payload has the same shape.
 */
export function showResults(result) {
  const detections = result.detections || [];
  renderBadge(result, detections.length);
  const list = $("det-list");

  if (!detections.length) {
    showEmpty(list, "Nothing detected");
    return;
  }

  replace(
    list,
    [...detections]
      .sort((a, b) => b.confidence - a.confidence)
      .map((detection) =>
        el("div", { className: "det" }, [
          el("span", { className: "swatch", style: { background: colorFor(detection.class_name) } }),
          el("span", { className: "name", text: detection.class_name }),
          el("span", { className: "conf", text: `${(detection.confidence * 100).toFixed(1)}%` }),
        ]),
      ),
  );
}

function renderBadge(result, count) {
  const badge = $("badge");
  show(badge, true);

  const timings = result.timings || {};
  const inference = timings.inference_ms;
  const parts = [
    el("b", { text: String(count) }),
    ` detections · ${result.device || ""}`,
    el("br"),
  ];

  if (timings.total_ms !== undefined && timings.total_ms !== null) {
    const fps = inference ? ` · ${(1000 / inference).toFixed(0)} FPS` : "";
    const size = store.model?.imgsz ? ` @${store.model.imgsz}px` : "";
    parts.push(
      el("b", { text: String(inference) }),
      ` ms infer${size}${fps}`,
      el("br"),
      el("span", {
        className: "sub",
        text: `pre ${timings.preprocess_ms} · post ${timings.postprocess_ms} · total ${timings.total_ms} ms`,
      }),
    );
  } else {
    parts.push(el("b", { text: String(result.speed_ms) }), " ms");
  }
  replace(badge, parts);
}
