"""Loading, inspecting and benchmarking the model.

The upload endpoint is the interesting one. It is written so that **an upload
that cannot be loaded costs you nothing**: files accumulate in a pending
directory, are only promoted into a model slot once they form something
loadable, and the running model is only replaced once its replacement is
already running. Dropping half of an OpenVINO model, an unsupported file or an
oversized one all leave the current model exactly as it was.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from .. import config, model_loader
from ..errors import ModelLoadError, ModelNotFound, UnsupportedMedia
from ..state import ModelRegistry
from ..uploads import enforce_file_count, safe_filename, save_streaming
from .deps import get_registry

log = logging.getLogger("prelabel.api.models")

router = APIRouter(prefix="/api", tags=["model"])

#: Guard against a typo'd imgsz turning into a multi-minute allocation.
MIN_IMGSZ, MAX_IMGSZ = 32, 8192


def _validate_extensions(files: list[UploadFile]) -> None:
    unknown = sorted(
        {
            Path(safe_filename(f.filename)).suffix.lower() or "(no extension)"
            for f in files
            if Path(safe_filename(f.filename)).suffix.lower() not in model_loader.ACCEPTED_EXTENSIONS
        }
    )
    if unknown:
        accepted = ", ".join(sorted(model_loader.ACCEPTED_EXTENSIONS))
        raise UnsupportedMedia(f"Unsupported file type(s): {', '.join(unknown)}. Accepted: {accepted}")


def _validate_imgsz(imgsz: int | None) -> int | None:
    if imgsz is None:
        return None
    value = int(imgsz)
    if not MIN_IMGSZ <= value <= MAX_IMGSZ:
        raise ModelLoadError(f"imgsz must be between {MIN_IMGSZ} and {MAX_IMGSZ} (got {value})")
    return value


def _stage_uploads(files: list[UploadFile]) -> None:
    """Write uploads into the pending directory, restarting on a primary file."""
    # A primary file means "load this model", so previous leftovers are dropped.
    # Companion files (a lone .bin arriving after its .xml) accumulate instead.
    if any(model_loader.is_primary(f.filename) for f in files):
        shutil.rmtree(config.PENDING_DIR, ignore_errors=True)
    config.PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for upload in files:
        save_streaming(upload, config.PENDING_DIR, config.MAX_MODEL_MB)


def _promote(registry: ModelRegistry, imgsz: int | None, device: str | None) -> dict:
    """Copy the pending files into a fresh slot and load them.

    On any failure the slot is removed and the pending files are kept, so the
    user can fix the upload (add a ``metadata.yaml``, re-send a truncated file)
    without starting over — and whatever was already loaded keeps working.
    """
    slot = registry.new_slot()
    try:
        for item in config.PENDING_DIR.iterdir():
            if item.is_file():
                shutil.copy2(item, slot / item.name)
        target = model_loader.prepare(slot)
        info = registry.load(target, slot, imgsz=imgsz, device=device)
    except BaseException:
        registry.discard_slot(slot)
        raise

    shutil.rmtree(config.PENDING_DIR, ignore_errors=True)
    return info


@router.post("/model")
def upload_model(
    files: list[UploadFile] = File(...),
    imgsz: int | None = Form(None),
    device: str | None = Form(None),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Upload and load a model — one file or several at once.

    Several files are accepted so multi-part formats work in a single drop:
    an OpenVINO ``.xml`` together with its ``.bin`` (and optional
    ``metadata.yaml``) load in one go.

    Declared ``def`` rather than ``async def`` so FastAPI runs the blocking model
    load in a worker thread instead of stalling the event loop.
    """
    files = enforce_file_count(files, config.MAX_MODEL_FILES, "model files")
    _validate_extensions(files)
    imgsz = _validate_imgsz(imgsz)

    _stage_uploads(files)

    plan = model_loader.inspect(config.PENDING_DIR)
    if not plan.is_ready:
        # Nothing has been touched: whatever was loaded is still loaded.
        return {
            "status": "waiting",
            "message": plan.message,
            "model_loaded": registry.is_loaded,
            "model": registry.info(),
        }

    return {"status": "ok", "model": _promote(registry, imgsz, device)}


@router.get("/model")
def model_info(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Details of the model currently in memory."""
    info = registry.info()
    if info is None:
        raise ModelNotFound()
    return info


@router.delete("/model")
def unload_model(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Release the model and free its memory (including GPU memory)."""
    registry.unload()
    return {"status": "ok", "model_loaded": False, "model": None}


@router.post("/device")
def set_device(
    device: str = Form(...),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Switch the inference device and reload the current model on it.

    The files are still on disk, so this rebuilds the engine — no re-upload. The
    image size chosen at load time is preserved.
    """
    return {"status": "ok", "model": registry.reload_on_device(device)}


@router.post("/benchmark")
def benchmark(
    runs: int = Form(50),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Measure single-image latency against best-case throughput.

    Holds the engine exclusively for the duration — the numbers are meaningless
    if other inference runs alongside — which is why the run count is capped.
    """
    runs = max(config.BENCHMARK_MIN_RUNS, min(int(runs), config.BENCHMARK_MAX_RUNS))
    with registry.engine() as engine:
        result = engine.benchmark(runs)
    return {"status": "ok", **result}
