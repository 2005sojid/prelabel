/**
 * Live webcam inference over a WebSocket.
 *
 * One frame is in flight at a time: the next frame is only captured after a
 * result comes back. That keeps the server from queueing work it cannot keep up
 * with, and keeps latency honest — what you see is the real round trip.
 */

import * as api from "./api.js";
import { $, setText, show } from "./dom.js";
import { confidence, store } from "./state.js";
import * as toast from "./toast.js";
import * as viewport from "./viewport.js";

const CAPTURE_WIDTH = 1280;
const CAPTURE_HEIGHT = 720;
const JPEG_QUALITY = 0.7;

const IDLE_LABEL = "🎥 Webcam";
const ACTIVE_LABEL = "■ Stop webcam";

const video = document.createElement("video");
video.autoplay = true;
video.playsInline = true;
video.muted = true;

const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d");

let socket = null;
let stream = null;
let running = false;
/** Set when the user asked to stop, so a socket close is not reported as a fault. */
let stoppingDeliberately = false;

export const isStreaming = () => running;

export async function start(onFrameResult) {
  if (!store.model) {
    toast.warn("Load a model first", "The webcam streams frames to the model as it runs.");
    return false;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT },
    });
  } catch (error) {
    toast.error("Could not access the webcam", error.message);
    return false;
  }

  video.srcObject = stream;
  await video.play();

  running = true;
  stoppingDeliberately = false;
  setButton(true);
  show($("hint"), false);
  show($("video"), false);
  $("canvas").hidden = false;

  socket = api.openStream();
  socket.addEventListener("open", sendFrame);
  socket.addEventListener("message", (event) => onMessage(event, onFrameResult));
  socket.addEventListener("error", () => {
    if (!stoppingDeliberately) toast.error("Webcam stream failed", "The connection to the server was lost.");
  });
  socket.addEventListener("close", () => {
    if (running) stop();
  });
  return true;
}

function onMessage(event, onFrameResult) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    return; // malformed frame — just wait for the next one
  }

  if (payload.status === "ok") {
    viewport.renderSource(capture, payload.detections || [], payload.task);
    onFrameResult?.(payload);
  } else if (payload.detail) {
    toast.error("Stream error", payload.detail);
    stop();
    return;
  }

  if (running) requestAnimationFrame(sendFrame);
}

function sendFrame() {
  if (!running || !socket || socket.readyState !== WebSocket.OPEN) return;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width) {
    requestAnimationFrame(sendFrame); // camera has not produced a frame yet
    return;
  }

  capture.width = width;
  capture.height = height;
  captureCtx.drawImage(video, 0, 0, width, height);
  socket.send(JSON.stringify({
    image: capture.toDataURL("image/jpeg", JPEG_QUALITY),
    conf: confidence(),
  }));
}

export function stop() {
  if (!running) return;
  running = false;
  stoppingDeliberately = true;

  if (socket) {
    socket.close();
    socket = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  setButton(false);
}

function setButton(active) {
  const button = $("webcam-btn");
  button.classList.toggle("active", active);
  setText(button, active ? ACTIVE_LABEL : IDLE_LABEL);
}
