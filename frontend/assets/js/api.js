/**
 * Every call to the server lives here.
 *
 * Two rules the rest of the app relies on:
 *  - failures always arrive as an {@link ApiError} carrying the server's `detail`
 *  - a non-JSON body (a proxy error page, an empty 502) never throws a confusing
 *    "... is not valid JSON"; the raw text becomes the message instead
 */

const BASE = "";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readBody(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(BASE + path, options);
  } catch (cause) {
    throw new ApiError("Could not reach the server. Is it still running?", 0);
  }
  const body = await readBody(response);
  if (!response.ok) throw new ApiError(body.detail || `Request failed (HTTP ${response.status})`, response.status);
  return body;
}

function form(fields) {
  const data = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) value.forEach((item) => data.append(key, item));
    else data.append(key, value);
  }
  return data;
}

// --- reads ------------------------------------------------------------------

export const getFormats = () => request("/api/formats");
export const getHealth = () => request("/api/health");

// --- model ------------------------------------------------------------------

export const uploadModel = (files, { device, imgsz } = {}) =>
  request("/api/model", { method: "POST", body: form({ files, device, imgsz }) });

export const setDevice = (device) =>
  request("/api/device", { method: "POST", body: form({ device }) });

export const unloadModel = () => request("/api/model", { method: "DELETE" });

export const benchmark = (runs) =>
  request("/api/benchmark", { method: "POST", body: form({ runs }) });

// --- inference --------------------------------------------------------------

export const predictImage = (file, conf, options = {}) =>
  request("/api/predict", { method: "POST", body: form({ file, conf, ...options }) });

export const predictBatch = (files, conf, options = {}) =>
  request("/api/predict/batch", { method: "POST", body: form({ files, conf, ...options }) });

/**
 * Video inference.
 *
 * Returns the rendered clip plus what the server reported about it: which codec
 * it could use, whether a browser can play that codec, and whether the frame cap
 * sampled the clip. The UI surfaces these instead of silently handing the user a
 * video their browser cannot decode.
 */
export async function predictVideo(file, conf, options = {}) {
  let response;
  try {
    response = await fetch(BASE + "/api/predict/video", { method: "POST", body: form({ file, conf, ...options }) });
  } catch {
    throw new ApiError("Could not reach the server. Is it still running?", 0);
  }
  if (!response.ok) {
    const body = await readBody(response);
    throw new ApiError(body.detail || `Video inference failed (HTTP ${response.status})`, response.status);
  }
  const header = (name) => response.headers.get(name);
  return {
    blob: await response.blob(),
    codec: header("X-Prelabel-Codec") || "",
    codecNote: header("X-Prelabel-Codec-Note") || "",
    browserPlayable: header("X-Prelabel-Browser-Playable") !== "0",
    frames: Number(header("X-Prelabel-Frames") || 0),
    sampled: header("X-Prelabel-Sampled") === "1",
    truncated: header("X-Prelabel-Truncated") === "1",
    tracked: header("X-Prelabel-Tracked") === "1",
  };
}

/** Open the webcam inference socket. */
export function openStream() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${location.host}/api/stream`);
}

// --- auth -------------------------------------------------------------------

export const authStatus = () => request("/api/auth/status");

export const login = (token) =>
  request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });

export const logout = () => request("/api/auth/logout", { method: "POST" });

// --- projects ---------------------------------------------------------------

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const datasetRoots = () => request("/api/projects/-/roots");

export const browseFolder = (path = "") =>
  request(`/api/projects/-/browse?path=${encodeURIComponent(path)}`);

export const listProjects = () => request("/api/projects");
export const getProject = (id) => request(`/api/projects/${id}`);
export const createProject = (payload) => request("/api/projects", json("POST", payload));
export const updateProject = (id, payload) => request(`/api/projects/${id}`, json("PATCH", payload));
export const deleteProject = (id) => request(`/api/projects/${id}`, { method: "DELETE" });

export const runProject = (id, payload = {}) => request(`/api/projects/${id}/run`, json("POST", payload));
export const cancelProject = (id) => request(`/api/projects/${id}/cancel`, { method: "POST" });
export const rescanProject = (id) => request(`/api/projects/${id}/rescan`, { method: "POST" });

export function listItems(id, { offset = 0, limit = 200, order = "path", only = "", search = "" } = {}) {
  const query = new URLSearchParams({ offset, limit, order });
  if (only && only !== "all") query.set("only", only);
  if (search) query.set("search", search);
  return request(`/api/projects/${id}/items?${query}`);
}

/** URL of a project image — used directly as an `<img src>`. */
export const itemImageUrl = (projectId, itemId, { thumb = false } = {}) =>
  `${BASE}/api/projects/${projectId}/items/${itemId}/image${thumb ? "?thumb=true" : ""}`;

export const exportUrl = (id, format, conf = 0) =>
  `${BASE}/api/projects/${id}/export?format=${encodeURIComponent(format)}&conf=${conf}`;

export function importAnnotations(id, file, { replace = true, into = "current" } = {}) {
  const data = new FormData();
  data.append("file", file);
  return request(
    `/api/projects/${id}/import?replace=${replace}&into=${into}`,
    { method: "POST", body: data },
  );
}

// --- retraining --------------------------------------------------------------

export const startTraining = (id, settings = {}) =>
  request(`/api/projects/${id}/train`, json("POST", { settings }));
export const cancelTraining = (id) => request(`/api/projects/${id}/train/cancel`, { method: "POST" });
export const getTraining = (id) => request(`/api/projects/${id}/training`);
export const adoptRetrained = (id) => request(`/api/projects/${id}/train/adopt`, { method: "POST" });

// --- comparing two annotation sets -------------------------------------------

export const captureBaseline = (id) => request(`/api/projects/${id}/baseline`, { method: "POST" });
export const clearBaseline = (id) => request(`/api/projects/${id}/baseline`, { method: "DELETE" });
export const getComparison = (id) => request(`/api/projects/${id}/comparison`);
export const recompare = (id, iou) =>
  request(`/api/projects/${id}/comparison?iou=${iou}`, { method: "POST" });
export const itemComparison = (id, itemId) =>
  request(`/api/projects/${id}/items/${itemId}/comparison`);
