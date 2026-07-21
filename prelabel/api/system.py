"""Liveness, capabilities and the frontend itself."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse

from .. import config, datasets
from ..engines import SUPPORTED_EXTENSIONS
from ..engines.ultralytics_engine import TRACKERS
from ..media import effective_codec
from ..state import ModelRegistry
from .deps import get_registry

log = logging.getLogger("prelabel.api.system")

router = APIRouter()

_INDEX = config.FRONTEND_DIR / "index.html"

_MISSING_FRONTEND = """<!doctype html><meta charset="utf-8">
<title>Prelabel</title>
<h1>Prelabel</h1>
<p>The frontend was not found at <code>frontend/index.html</code>.
The API is running — see <a href="/docs">/docs</a>.</p>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():  # noqa: ANN201
    """Serve the single-page UI."""
    if _INDEX.exists():
        return FileResponse(_INDEX, media_type="text/html")
    return HTMLResponse(_MISSING_FRONTEND, status_code=200)


@router.get("/api/health")
def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Liveness plus which model, if any, is currently loaded."""
    info = registry.info()
    return {"status": "ok", "version": _version(), "model_loaded": info is not None, "model": info}


@router.get("/api/formats")
def formats() -> dict:
    """Everything the UI needs to know about what this server accepts."""
    codec = effective_codec()
    return {
        "model_formats": SUPPORTED_EXTENSIONS,
        "image_formats": sorted(config.IMAGE_EXTENSIONS),
        "video_formats": sorted(config.VIDEO_EXTENSIONS),
        "devices": available_devices(),
        "video_codec": {
            "fourcc": codec.fourcc,
            "browser_playable": codec.browser_playable,
            "note": codec.note,
        },
        "limits": {
            "max_model_mb": config.MAX_MODEL_MB,
            "max_media_mb": config.MAX_MEDIA_MB,
            "max_model_files": config.MAX_MODEL_FILES,
            "max_batch_files": config.MAX_BATCH_FILES,
            "max_video_frames": config.MAX_VIDEO_FRAMES,
        },
        "features": {
            "projects": datasets.is_configured(),
            "auth_required": bool(config.AUTH_TOKEN),
            "trackers": list(TRACKERS),
        },
    }


def available_devices() -> list[dict[str, str]]:
    """Selectable inference devices. CPU is always present."""
    devices = [{"id": "cpu", "label": "CPU"}]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append({"id": "cuda", "label": f"GPU · {torch.cuda.get_device_name(0)}"})
    except Exception as exc:  # noqa: BLE001 - no torch or no driver: CPU still works
        log.debug("CUDA device probe failed: %s", exc)
    return devices


def _version() -> str:
    from .. import __version__

    return __version__
