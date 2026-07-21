"""Runtime configuration, overridable through ``PL_*`` environment variables.

Every setting is read once at import time and exposed as a module-level
constant. Nothing here does I/O beyond :func:`ensure_dirs`, so importing this
module is always cheap and side-effect free.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths ------------------------------------------------------------------

#: Project root (one level above the ``app`` package).
ROOT = Path(__file__).resolve().parent.parent

FRONTEND_DIR = ROOT / "frontend"

STORAGE_DIR = Path(os.getenv("PL_STORAGE_DIR", ROOT / "storage"))

#: Where project metadata and results are kept between runs.
DATABASE_PATH = Path(os.getenv("PL_DATABASE", STORAGE_DIR / "prelabel.db"))
#: Holds one ``active-<uuid>/`` directory per loaded model, plus ``pending/``
#: where a multi-part upload accumulates until it is complete. See
#: :mod:`app.state` for why models live in per-load slots.
MODELS_DIR = STORAGE_DIR / "models"
#: Scratch space for uploaded videos and the rendered results.
OUTPUTS_DIR = STORAGE_DIR / "outputs"
#: Retrain artifacts: one directory per run, holding the produced weights. Kept
#: between restarts — a project's stored training state points into it.
TRAINING_DIR = STORAGE_DIR / "training"

#: Where an incomplete multi-part upload (e.g. a lone OpenVINO ``.xml``) waits
#: for its companion files. Kept apart from the live model so an incomplete
#: upload can never disturb what is already loaded.
PENDING_DIR = MODELS_DIR / "pending"

# --- server -----------------------------------------------------------------

HOST = os.getenv("PL_HOST", "127.0.0.1")
PORT = int(os.getenv("PL_PORT", "8000"))


def _csv(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


#: Extra browser origins allowed to call the API. The server's own origin is
#: always accepted, so the bundled UI works with this list empty.
#:
#: This is deliberately *not* ``*``. A cross-origin ``POST`` of
#: ``multipart/form-data`` is a CORS "simple request" — no preflight — so any web
#: page the user happens to have open could otherwise upload a model to this
#: server, and loading a ``.pt`` unpickles arbitrary code. See
#: :mod:`app.security`.
ALLOWED_ORIGINS = _csv("PL_ALLOWED_ORIGINS")

#: Directories the server may read image folders from, as an OS-path list
#: (``;`` on Windows, ``:`` elsewhere).
#:
#: Empty means folder projects are disabled. That is deliberate: reading a
#: server-side path on request is arbitrary file access, and a default of "any
#: path this process can read" is not something to switch on silently. One
#: variable turns it on for exactly the directories you name.
DATA_ROOTS: list[Path] = [
    Path(entry).expanduser().resolve()
    for entry in os.getenv("PL_DATA_ROOTS", "").split(os.pathsep)
    if entry.strip()
]

#: Shared secret required on every request. Empty = no authentication, which is
#: the right default for a tool bound to loopback. Set it before exposing the
#: server to anyone else.
AUTH_TOKEN = os.getenv("PL_AUTH_TOKEN", "").strip()

#: Paths reachable without a token even when one is configured — the login page
#: and the liveness probe a container orchestrator needs.
PUBLIC_PATHS = ("/api/health",)

#: Return the underlying exception message to the client on an unhandled error.
#: On by default because this is a local tool and the UI shows the text — turn it
#: off (``PL_VERBOSE_ERRORS=0``) when the server is reachable by anyone else, as
#: the messages can contain filesystem paths.
VERBOSE_ERRORS = os.getenv("PL_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no")

# --- inference --------------------------------------------------------------

DEFAULT_CONF = float(os.getenv("PL_DEFAULT_CONF", "0.5"))

#: Force a specific device ("cpu", "cuda", "cuda:0"). Empty = auto-detect.
DEVICE = os.getenv("PL_DEVICE", "").strip()

#: Bounds for ``POST /api/benchmark``. The benchmark needs exclusive use of the
#: engine, so the upper bound also bounds how long the server can be busy.
BENCHMARK_MIN_RUNS = 5
BENCHMARK_MAX_RUNS = int(os.getenv("PL_BENCHMARK_MAX_RUNS", "200"))

# --- limits -----------------------------------------------------------------

#: Cap how many video frames we process so an accidental 4K/60fps upload cannot
#: lock the server up. Frames are sampled evenly across the clip, never
#: truncated. Set to 0 to disable.
MAX_VIDEO_FRAMES = int(os.getenv("PL_MAX_VIDEO_FRAMES", "900"))

#: Upload guards in megabytes. 0 = unlimited.
MAX_MODEL_MB = int(os.getenv("PL_MAX_MODEL_MB", "1024"))
MAX_MEDIA_MB = int(os.getenv("PL_MAX_MEDIA_MB", "512"))

#: Upper bound on files accepted by a single request. Without this, one request
#: can pin an unbounded amount of memory regardless of the per-file cap.
MAX_MODEL_FILES = int(os.getenv("PL_MAX_MODEL_FILES", "16"))
MAX_BATCH_FILES = int(os.getenv("PL_MAX_BATCH_FILES", "64"))

#: Largest single WebSocket frame accepted from the webcam stream.
MAX_STREAM_FRAME_MB = int(os.getenv("PL_MAX_STREAM_FRAME_MB", "16"))

# --- formats ----------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# --- projects ---------------------------------------------------------------

#: Images processed per batch by the background project runner.
PROJECT_BATCH_SIZE = int(os.getenv("PL_PROJECT_BATCH_SIZE", "16"))

#: Confidence a project run infers at. Kept low on purpose: results are stored
#: once and the review threshold is applied when reading them, so raising the
#: slider never means re-running the model.
PROJECT_CONF_FLOOR = float(os.getenv("PL_PROJECT_CONF_FLOOR", "0.05"))

#: Cap on images per project, so one mistaken path cannot enqueue a whole disk.
MAX_PROJECT_IMAGES = int(os.getenv("PL_MAX_PROJECT_IMAGES", "100000"))

#: Longest edge of a generated gallery thumbnail.
THUMBNAIL_EDGE = int(os.getenv("PL_THUMBNAIL_EDGE", "480"))

# --- retraining -------------------------------------------------------------

#: Defaults for a fine-tune run. Epochs are kept modest because the point is to
#: adapt a model to corrections, not train one from scratch — raise it for a
#: bigger correction set.
TRAIN_EPOCHS = int(os.getenv("PL_TRAIN_EPOCHS", "40"))
TRAIN_IMGSZ = int(os.getenv("PL_TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.getenv("PL_TRAIN_BATCH", "16"))
#: Fraction of labelled images held out to measure the retrained model on.
TRAIN_VAL_FRACTION = float(os.getenv("PL_TRAIN_VAL_FRACTION", "0.2"))
#: Ignore source boxes below this confidence when building the training set, so
#: training on raw predictions does not learn the model's own low-confidence
#: noise. Corrections arrive at confidence 1.0 and are unaffected.
TRAIN_MIN_CONF = float(os.getenv("PL_TRAIN_MIN_CONF", "0.25"))


def ensure_dirs() -> None:
    """Create the storage layout. Safe to call repeatedly."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
