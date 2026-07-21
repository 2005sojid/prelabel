# Prelabel

**The missing layer between inference and annotation.**

Point any CV model at a folder. Review what it found, least-confident first.
Hand the result to CVAT — or let CVAT call it directly.

Prelabel loads a model in almost any format, runs it over images, video or a
webcam, and turns the predictions into a dataset your annotation tool can open —
so people correct a model instead of drawing from scratch.

> Point it at a folder → it pre-labels → review the least-confident images →
> export to CVAT, or let CVAT call Prelabel directly.
> Runs locally, your data never leaves the machine.

`v0.3` · MIT licensed · Python + FastAPI, zero-build ES-module frontend.

---

## Why

Labelling is the slowest, most expensive part of any CV project. If you already
have a decent model, Prelabel pre-labels a new image set with it, shows you the
images the model is *least sure about*, and hands the result to the tool you
already correct labels in.

**Prelabel does not edit annotations.** That is deliberate — CVAT and Label
Studio are good at it. Prelabel's job is everything before and after: run the
model at scale, order the review queue, diff one set of labels against another,
fine-tune the model on the corrections, and move datasets in and out.

That diff is the part annotation tools do not do. "Did the new model get better?"
and "where is my ground truth actually wrong?" are the same question asked from
two sides, and both are answered by comparing two sets of boxes over the same
images — which is why the comparison is per object, not a single mAP number.

And it closes: pre-label a folder → correct the worst guesses in CVAT → import
them back → **retrain the model on the corrections** → re-run and diff the new
model against the old. The next folder starts from a better model.

## Features

**Models**
- PyTorch `.pt`, ONNX, OpenVINO `.xml/.bin`, TorchScript — all four verified end
  to end. TensorRT `.engine`, TFLite and CoreML are implemented through the same
  backend but **have not been run** (see [What is and isn't verified](#what-is-and-isnt-verified)).
- The backend is chosen automatically and *verified*: a model that loads but
  cannot infer is rejected so the next backend gets a turn.
- The input size is read from the model artifact, not guessed.
- Detection, segmentation, pose, oriented boxes and classification, auto-detected.

**Running**
- **Projects** — point at a folder on the server; results are stored in SQLite as
  they are produced. Close the tab, restart the server, pick up where it left off.
- **Sliced inference** for small objects in large images — on a 2400 px canvas
  this took a test from 3 detections to 38.
- **Object tracking** in video (ByteTrack / BoT-SORT). Ids are drawn on the
  rendered clip, so `#3 person` stays `#3` from frame to frame.
- **Class filtering** at inference time, not after.
- Batched throughput, CPU/GPU switching, and a latency-vs-throughput benchmark.

**Reviewing**
- **Least-confident-first ordering** — margin sampling puts the images where the
  model is genuinely undecided at the front of the queue.
- **Diff two sets of annotations.** Freeze what you have as a baseline, then run
  a different model — or import corrected labels — and see exactly what changed:
  per object (agreed / reclassified / missing / added), per class, per image.
  Sort by biggest disagreement to review the argument instead of the dataset.
- Filter, search, zoomable lightbox, instant confidence threshold (results are
  stored once; the slider never re-runs the model).

**Improving**
- **Retrain on corrections.** Fine-tune the loaded PyTorch model on the project's
  labels — the corrected ones, once a fixed COCO is imported over the model's
  guesses — from the browser: pick epochs, watch the per-epoch metrics, adopt the
  result as the active model, re-run, and diff it against the old one. The
  training set is built from the stored boxes (no re-inference), the base model's
  classes are preserved so a subset of them does not shrink its head, and the
  train/val split is deterministic so successive runs stay comparable.

**Moving data**
- Export to COCO / COCO Keypoints / ImageNet, built server-side and streamed —
  no browser memory limit, ZIP64 past 4 GB.
- **Import COCO back**, so corrections made in CVAT return to the project.
- **ML backend for CVAT** — one command registers Prelabel as an auto-annotation
  model; press "annotate" in CVAT and it answers. A **Label Studio** backend is
  implemented to the same contract but has not been run against a live instance.

## Quick start

```bash
pip install -r requirements.txt
python run.py                       # http://127.0.0.1:8000
```

Drop a model and some photos on the page and it works immediately.

For **projects** over a folder, tell the server which directories it may read:

```bash
PL_DATA_ROOTS=/data/images python run.py        # Linux/macOS
$env:PL_DATA_ROOTS="D:\datasets"; python run.py # Windows
```

Or with Docker, which brings the native stack (OpenCV's system libraries and an
ffmpeg with H.264) so you do not have to assemble it:

```bash
docker compose up --build     # put images in ./images
```

## Using it with CVAT

CVAT will not call a plain URL. It lists whatever is registered in **Nuclio**, so
the model has to exist there as a function. One command does that:

```bash
python run.py                              # in one terminal, with a model loaded
python serverless/cvat/deploy.py           # in another
```

That reads the loaded model's labels from `/api/cvat/info`, generates a Nuclio
function declaring them, and deploys it. The function is a thin proxy — no ML
dependencies, builds in seconds, and never needs rebuilding when the model
changes. Inference stays in Prelabel, on your GPU.

In CVAT: open a task → **Actions → Automatic annotation** → pick
*Prelabel (your-model)*.

> **Turn on "Clean previous annotations".** With it off, CVAT *appends* each run's
> results to what is already there, so running twice gives you every box twice.
> That is CVAT's behaviour, not the model's.

Options worth knowing:

```bash
python serverless/cvat/deploy.py --container-url http://host.docker.internal:8000
python serverless/cvat/deploy.py --network cvat_cvat    # must match Nuclio's network
python serverless/cvat/deploy.py --delete
```

Re-run `deploy.py` after loading a different model: the label list is baked into
the function's metadata, and that is what CVAT shows you.

For **Label Studio**, add an ML backend at `http://prelabel:8000/api/label-studio`
— it serves `/health`, `/setup` and `/predict`, with geometry in percentages as
Label Studio requires.

Or work file-first: export COCO from a project, correct it in CVAT, and import
the result back with **⬆ Import**.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Status + loaded model |
| GET | `/api/formats` | Formats, devices, codec, limits, features |
| POST · GET · DELETE | `/api/model` | Load, inspect, unload |
| POST | `/api/device` · `/api/benchmark` | Switch device · latency vs throughput |
| POST | `/api/predict` | One image (`conf`, `classes`, `tiled`, `tile_size`) |
| POST | `/api/predict/batch` | Several images, batched |
| POST | `/api/predict/video` | Video → annotated MP4 (`track`) |
| WS | `/api/stream` | Live webcam |
| GET · POST | `/api/projects` | List · create from a server folder |
| GET · PATCH · DELETE | `/api/projects/{id}` | Inspect · rename · delete |
| POST | `/api/projects/{id}/run` · `/cancel` · `/rescan` | Run control |
| GET | `/api/projects/{id}/items` | Paged results (`order=priority\|disputed`, `only`, `search`) |
| GET | `/api/projects/{id}/items/{n}/image` | Image or cached thumbnail |
| GET | `/api/projects/{id}/items/{n}/comparison` | Per-object diff for one image |
| POST · DELETE | `/api/projects/{id}/baseline` | Freeze the current set to compare against · drop it |
| GET · POST | `/api/projects/{id}/comparison` | Read the diff · recompute it |
| POST | `/api/projects/{id}/train` · `/train/cancel` | Fine-tune on the labels · stop |
| GET · POST | `/api/projects/{id}/training` · `/train/adopt` | Training state · load the retrained weights |
| GET · POST | `/api/projects/{id}/export` · `/import` | Dataset out · COCO in (`?into=baseline`) |
| GET · POST | `/api/cvat/info` · `/api/cvat/invoke` | CVAT ML backend |
| GET · POST | `/api/label-studio/*` | Label Studio ML backend |
| POST | `/api/auth/login` · `/logout` | Only when a token is set |

Errors are always JSON: `{"status": "error", "detail": "..."}`.

```bash
curl -F "files=@yolov8n.pt" http://127.0.0.1:8000/api/model
curl -F "file=@street.jpg" -F "conf=0.3" -F "tiled=true" http://127.0.0.1:8000/api/predict
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `PL_HOST` / `PL_PORT` | `127.0.0.1` / `8000` | Bind address |
| `PL_DATA_ROOTS` | *(none)* | Folders the server may read datasets from. **Required for projects** — empty means the feature is off. |
| `PL_AUTH_TOKEN` | *(none)* | Shared token. Empty = no auth, right for loopback. |
| `PL_ALLOWED_ORIGINS` | *(none)* | Extra browser origins. The server's own always works. |
| `PL_DEVICE` | *(auto)* | Force `cpu`, `cuda`, `cuda:0` |
| `PL_DEFAULT_CONF` | `0.5` | Default confidence |
| `PL_PROJECT_CONF_FLOOR` | `0.05` | Confidence a project run infers at |
| `PL_MAX_VIDEO_FRAMES` | `900` | Frame budget, sampled evenly across the clip |
| `PL_MAX_MODEL_MB` / `PL_MAX_MEDIA_MB` | `1024` / `512` | Upload caps |
| `PL_MAX_MODEL_FILES` / `PL_MAX_BATCH_FILES` | `16` / `64` | Files per request |
| `PL_MAX_PROJECT_IMAGES` | `100000` | Images per project |
| `PL_THUMBNAIL_EDGE` | `480` | Gallery thumbnail size |
| `PL_TRAIN_EPOCHS` / `PL_TRAIN_IMGSZ` / `PL_TRAIN_BATCH` | `40` / `640` / `16` | Fine-tune defaults |
| `PL_TRAIN_VAL_FRACTION` / `PL_TRAIN_MIN_CONF` | `0.2` / `0.25` | Val hold-out · ignore boxes below this when building the set |
| `PL_VERBOSE_ERRORS` | `1` | Return the real error message to the client |
| `PL_STORAGE_DIR` / `PL_DATABASE` | `./storage` | Where state lives |

## Tests

```bash
pip install -e ".[dev]"
pytest                       # 378 tests
pytest -m "not integration"  # fast: no model downloads
pytest --cov=prelabel        # coverage (85%)
npm test                     # frontend ZIP writer
ruff check prelabel tests run.py
```

## Design notes

Decisions that are easy to get wrong, and why they are the way they are:

- **A failed model load never costs you the working model.** Uploads stage in a
  pending directory and are only promoted once loadable; the replacement engine
  is built and *self-tested* before the old one is released.
- **Constructing an engine is not the same as it working.** A backend can accept
  the wrong file, populate itself with nonsense, and only fail on the first real
  image — a plain ResNet handed to a YOLO loader does exactly that. Every engine
  runs one inference before it is accepted, so the factory can fall through to
  the right backend.
- **Video frames are sampled, never truncated.** A frame cap picks evenly spaced
  frames across the whole clip and adjusts the output frame rate so duration is
  preserved.
- **The video codec is probed, not assumed.** OpenCV's usual `mp4v` writes MPEG-4
  Part 2, which Chrome and Firefox will not play. We encode a throwaway clip at
  startup and inspect the container.
- **Reading server-side folders is off until you name the directories.** It is
  arbitrary file access; a default of "anything this process can read" is not
  something to switch on silently.
- **State-changing requests are origin-checked.** A cross-origin `POST` of
  `multipart/form-data` needs no preflight, so without this any page you have
  open could load a model into your server — and a `.pt` is a pickle.
- **The UI never puts data in `innerHTML`.** Filenames, class names and server
  messages are set as text.
- **Optional dependencies degrade, they do not crash.** Tracking needs `lap`;
  without it `supports_tracking` is false and video still renders, just without ids.
- **A diff matches same-class boxes first.** Two passes: pair identical classes,
  and only then look for overlapping boxes that disagree about *what* they are.
  One pass would let a greedy `truck`↔`bus` match steal the box a correct `bus`
  was waiting for, inventing a disagreement out of an agreement.
- **Two images both called empty agree.** Nothing to compare is not a conflict,
  so those score 1.0 rather than 0 — otherwise every background frame would rise
  to the top of a queue sorted by disagreement.
- **Retraining learns from corrections, not from predictions.** A model
  fine-tuned on its own output only reinforces what it already believes. Nothing
  can prove a label set was reviewed, so the contract is explicit: training runs
  on the set you choose, and it is meant to be the corrected one. The base
  model's classes are carried into the dataset so fine-tuning a subset of them
  leaves the detection head intact, and the train/val split is hashed from the
  path so a retrain's metrics are comparable to the last one's.
- **A fine-tune and a run never overlap.** They share the one GPU, so starting
  either is refused while the other is going, rather than letting them fight over
  the device and make both slower.

## What is and isn't verified

Everything below is either exercised by the test suite, driven through the real
HTTP API on a GPU, or clicked through in a browser. What is *not* is listed just
as plainly, because a feature nobody has run is a guess.

| | Status |
|---|---|
| PyTorch, TorchScript, ONNX, OpenVINO | ✅ 12 end-to-end checks each, on GPU |
| Generic ONNX classifiers (ResNet etc.) | ✅ routed, classified, filtered |
| Detection, segmentation, pose, classification | ✅ on real photographs |
| Oriented boxes (OBB) | ⚠️ model loads and runs; never fed aerial imagery, so no OBB output has been inspected |
| TensorRT, TFLite, CoreML | ❌ implemented, **never run** — no runtimes installed here |
| Projects: scan, run, resume, review, export | ✅ including a browser walkthrough |
| Sliced inference | ✅ 3 → 38 detections on a 2400 px canvas |
| Object tracking | ✅ stable ids across frames, drawn on the clip |
| COCO export / import | ✅ archives validated with Python's `zipfile` |
| Annotation diff | ✅ yolov8n vs yolov8s over the same folder: 48.3% agreement, 43 missing, 32 added |
| Retraining | ✅ real fine-tune in the browser: build → train → adopt → re-run → diff, plus cancel mid-run |
| CVAT ML backend | ✅ against a live CVAT 2.42.1: deployed, listed, annotated |
| Label Studio ML backend | ❌ contract implemented and unit-tested; **never run against Label Studio** |
| Docker image | ✅ builds, runs, healthy, video comes out as H.264 |
| Webcam | ❌ protocol tested with synthetic frames; **never run with a camera** |

## Notes & limitations

- **Single user, one model.** `/api/benchmark` and a project run hold the engine
  exclusively — the numbers are meaningless otherwise.
- **Trust your models.** Loading a `.pt` unpickles arbitrary code. The origin
  guard stops a *web page* doing it behind your back; it cannot make an untrusted
  file safe.
- **No annotation editing**, by design — correct in CVAT and import back.
- **Retraining is detection-only, and needs a PyTorch model.** Fine-tuning runs
  through Ultralytics on a loaded `.pt`; an exported ONNX/OpenVINO model has no
  trainable weights, and segmentation/pose retraining is not wired up yet. A GPU
  is not required but is strongly wanted.
- **Ultralytics is AGPL-3.0.** Prelabel is MIT, but the default inference backend
  is AGPL. Check what that implies before you distribute or network-serve this.
- TensorRT, TFLite, CoreML and TensorFlow paths are implemented but were not
  verified here — those runtimes were not installed. PyTorch, TorchScript, ONNX,
  OpenVINO and generic ONNX classifiers are verified end to end.

## Roadmap

- Retraining for segmentation and pose, not just detection.
- Background images in the training set, and a "train only on reviewed images" flag.
- Diff more than two sets at once, and across projects.
- Playwright browser tests.

## License

MIT — see [LICENSE](LICENSE).
