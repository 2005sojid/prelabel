# Changelog

All notable changes to Prelabel (formerly LabelForge) are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-07-21

Renamed from **LabelForge** to **Prelabel** (package `prelabel`, environment
prefix `PL_*`), and grown from a single-page tool into something that survives a
restart and talks to the tool you actually correct labels in.

Annotation editing was deliberately dropped from the roadmap: CVAT does it well,
so Prelabel feeds CVAT rather than competing with it.

### Added

- **Retraining on corrections.** Fine-tune the loaded PyTorch model on a
  project's labels, straight from the browser: choose the epochs and which set to
  learn from, watch the validation metrics land per epoch, then adopt the
  retrained weights as the active model and re-run to diff it against the old
  one. Closes the loop the rest of the tool sets up — pre-label, correct in CVAT,
  import back, retrain.

  The training set is built from the stored boxes rather than re-inferring them,
  in YOLO's normalised layout with a deterministic train/val split so successive
  runs stay comparable. The base model's classes are carried into the dataset, so
  fine-tuning a subset of them does not shrink its detection head. Training and a
  project run are mutually exclusive — they share the one GPU — and a fine-tune
  can be cancelled, stopping cleanly at the next epoch boundary with its
  partial weights kept. Detection models only, for now.
- **Annotation diffing.** Freeze the current annotations as a baseline, then run
  a different model or import corrected labels, and see what actually changed:
  every object is classified as agreed, reclassified, missing or added, with the
  totals rolled up per class and per image. Sorting by disagreement puts the
  images the two sets argue about at the front of the queue, and the baseline can
  be drawn over the current boxes in the lightbox as a dashed outline.

  Matching runs in two passes — same class first, differing classes second — so a
  greedy `truck`↔`bus` overlap cannot steal the box a correct `bus` was waiting
  for. Two sets that both call an image empty count as agreeing, not as a
  conflict, or every background frame would float to the top of the review queue.
- **Projects.** Point the server at a folder; images are registered and results
  written to SQLite as they are produced. A run survives closing the tab or
  restarting the server, and resumes instead of starting over. Reading
  server-side paths stays off until `PL_DATA_ROOTS` names the directories — it is
  arbitrary file access, and a permissive default is not something to enable
  silently.
- **ML backend for CVAT and Label Studio.** Press "annotate" in either tool and
  Prelabel answers: CVAT gets pixel shapes, Label Studio gets percentages.
- **COCO import**, closing the loop — export, correct downstream, bring it back.
- **Sliced (tiled) inference** for small objects in large images. On a 2400 px
  test canvas this went from 3 detections to 38.
- **Object tracking** in video (ByteTrack / BoT-SORT) with stable track ids.
- **Least-confident-first ordering** — margin sampling over the stored results,
  so review effort lands where the model is genuinely undecided.
- **Class filtering** pushed into the backend, so the model's own NMS applies it.
- **Token authentication**, off by default. An `HttpOnly` cookie, because the
  gallery is full of image tags and an image tag cannot carry a header.
- **Server-side export**, streamed to disk with ZIP64 — no browser memory ceiling.
- **Dockerfile and compose file** including ffmpeg, so H.264 video works out of
  the box rather than after an OpenCV rebuild.
- `DELETE /api/model`, cached thumbnails, a server folder browser, project rescan.

### Fixed

- **Tracking shipped without its dependency.** `lap` was missing from
  requirements, so `track=true` failed part-way through rendering. It is declared
  now, and `supports_tracking` reports honestly and falls back to plain detection
  instead of crashing.
- **A model that loaded but could not infer was accepted.** A plain ResNet ONNX
  was taken by the Ultralytics backend, which reported a made-up task and 999
  classes and then failed on the first image. Every engine now runs one inference
  before it is accepted, so the factory falls through to the right backend.
- **`POST /run` returned before the status changed**, so a client polling
  immediately read the previous run's "done" and concluded the new one had
  already finished.
- A NUL byte in a requested path raised `ValueError` past the handler — a 500
  where it should have been a 400.
- A video whose model was swapped mid-render was annotated by two different
  models. Runs are now pinned to a model generation and fail with 409.
- The test suite shared one database in the developer's tree instead of an
  isolated one per test.

**Found by driving the UI in a real browser** — the gap the 0.2.0 notes admitted
to. None of these were visible to the API tests, because none of them are API
bugs:

- **Detection boxes on project thumbnails were offset.** The overlay scaled by
  width alone, while the thumbnail is drawn with `object-fit: cover` — so every
  box sat half the cropped edge out of place.
- **A pending project card hid its own thumbnail** behind the batch gallery's
  striped placeholder canvas, reused by class name.
- **Cards showed the stored detection count, not the count at the current
  threshold** — a card read 9 while opening it showed 4.
- **Dropping a photo while a project was open rendered the result behind the
  project view.** `routeMedia` stood down batch mode but did not know about
  projects.
- **"Model changed" warned on every page load**, because adopting the model the
  server already had looks like a change if you only listen for the event.
- **`window.prompt` and `window.confirm` had survived in the project flow** after
  `alert()` was removed everywhere else. Replaced with a name field in the folder
  dialog and a two-step delete button that says what it will and will not touch.
- **The open project is now kept in the URL fragment**, so a reload returns to it
  instead of an empty screen — which is the whole point of results that outlive
  the tab, and shareable as a link besides.
- **"Re-run" did nothing on a finished project.** A run resumes what is still
  pending, and on a finished project nothing is — so the button reported success,
  changed no results, and made a new model look identical to the old one. It now
  clears the previous results when there is nothing left to resume. Found by
  loading a second model in the browser and watching the diff stay at 100%.
- **The header stayed on "Training…" after a fine-tune finished.** The run-state
  render read the training status before the poll had refreshed it, so the status
  line and the disabled Run button lagged a cycle behind. Rendering it last, once
  the fresh state is in, fixed it — caught by watching a real training run end in
  the browser.
- **The lightbox header did not follow the confidence slider.** Moving it
  repainted the boxes but left the counts as they were, so the header claimed a
  number that was no longer on screen.

### Changed

- `main.py` gained a project store and a background runner; `api/` grew routers
  for projects, integrations and auth.
- `Detection` carries `class_id` and `track_id`; `InferenceResult` carries
  `review_priority`.
- Test suite: 178 → **378 tests**, coverage 82% → **85%**.
- `main.py` gained a background training runner beside the project runner;
  `store` schema v3 adds a per-project training-state column.

## [0.2.0] — 2026-07-21

A correctness and structure release. Two bugs here produced output that *looked*
right — the worst kind — and the model-loading path could lose a working model.

### Fixed

- **Video was truncated instead of sampled.** The frame cap read frames until it
  hit the limit, so any clip between one and two times the cap came back as only
  its beginning — a 60-second video silently became 30 seconds, with no error.
  Frames are now spread evenly across the whole clip, the full frame budget is
  used, and the output frame rate preserves the original duration.
- **Rendered video would not play in Chrome or Firefox.** The `mp4v` fourcc
  writes MPEG-4 Part 2, which neither browser decodes, so the download succeeded
  and the player stayed blank. Prelabel now probes for a real H.264 encoder at
  startup by inspecting the *container* it produces, and reports honestly (in
  `/api/formats` and in response headers) when only a fallback is available.
- **A failed model load could destroy the loaded model.** Uploading half an
  OpenVINO model, an oversized file, or a corrupt one released the running
  engine and wiped the model directory *before* discovering the new files were
  unusable. Uploads now stage in a pending directory, each load gets its own
  slot, and the replacement engine is built before the old one is released.
- **A model that loaded but did not work was accepted.** Some backends construct
  successfully from a garbage file and only fail later; the engine is now asked
  to describe itself before it is installed.
- **Upload size limits were checked after the upload was already in memory**, so
  they protected nothing. `/api/predict/batch` had no size limit at all, and no
  endpoint limited the *number* of files.
- **Switching device discarded an explicit `imgsz`**, silently changing how the
  model ran.
- **Batch export could produce a broken ZIP**: non-Latin filenames were replaced
  with underscores (and could collide), same-named files from different folders
  overwrote each other, and archives past 4 GB were written corrupt rather than
  refused.
- Missing or hostile upload filenames (`None`, `../../evil.pt`) caused a 500.
- Warmup failures were swallowed silently, so a broken model appeared to load.
- Decoded thumbnails (`ImageBitmap`) were never released when a batch was cleared.
- The `R` shortcut fired while typing in the filename filter.

### Added

- **Origin guard on state-changing requests.** A cross-origin `POST` of
  `multipart/form-data` needs no preflight, so any page you had open could
  upload a model to your local server — and loading a `.pt` unpickles arbitrary
  code. Requests without an `Origin` (curl, scripts) still work; extra origins
  via `PL_ALLOWED_ORIGINS`.
- `DELETE /api/model` and an **Unload model** button, to free memory (including
  GPU memory) deliberately.
- `/api/formats` now reports the video codec and every configured limit.
- `/api/predict/video` reports codec, playability, frame count and whether the
  clip was sampled, in response headers — surfaced in the UI as notifications.
- `pyproject.toml` with pytest, coverage and ruff configuration; GitHub Actions CI.
- Test suite grown from 23 to **178 tests** (82% coverage), including the video
  sampling regression, the model-lifecycle guarantee, the origin guard, and
  cross-validation of the frontend's ZIP writer against Python's `zipfile`.

### Changed

- **`main.py` split into an application factory plus four routers.** The single
  mutable global engine is replaced by `ModelRegistry`, which owns the lock and
  makes check-then-use atomic — the inference routes are `def` handlers, so
  FastAPI runs them concurrently in a thread pool and requests really do overlap.
- **The 1 200-line `index.html` is now markup, one stylesheet, and 19 ES
  modules** with no build step. Nothing interpolates data into `innerHTML` any
  more, so a filename like `<img onerror=…>.jpg` renders as a filename.
- `alert()` replaced with non-blocking notifications.
- CORS no longer defaults to `*`.
- Server-side video annotation now uses the same class-colour hash as the
  browser, so a class looks identical on the canvas and in a rendered MP4.
- Errors are typed and rendered by one handler, always as JSON.

### Removed

- Export formats that were listed but are better served by the COCO family:
  Ultralytics YOLO, Pascal VOC, CVAT for images 1.1, and classification CSV. The
  0.1.0 notes advertised these; the export menu now offers the standard
  interchange format per task and nothing that duplicates it.

## [0.1.0] — 2026-06-24

First tagged release.

### Added
- Universal model loading: PyTorch, ONNX, OpenVINO (`.xml`+`.bin`, with
  auto-built `metadata.yaml`), TorchScript, TensorRT, TFLite, CoreML — backend
  chosen automatically, input size read from the model.
- Tasks: detection, segmentation, pose, oriented boxes, classification.
- Inputs: single image, video (annotated `.mp4`), live webcam (WebSocket), and a
  **batch gallery** for whole folders.
- **Server-side batched inference** (`/api/predict/batch`) — chunks run as a
  single forward pass for GPU throughput instead of one request per image.
- Task-aware dataset export including images.
- Batch review: zoomable lightbox, per-class sidebar, filename search, filter,
  sort, thumbnail-size control, instant confidence filtering.
- CPU/GPU switch, latency-vs-throughput benchmark, honest per-request timing.
- Non-blocking server (threadpooled inference), JSON errors, upload size caps,
  temporary-file cleanup.

### Notes
- Single-user tool; one model in memory at a time behind a lock.
- `CORS` defaulted to `*` and there was no auth — intended for local use.
