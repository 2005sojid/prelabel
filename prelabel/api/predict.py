"""Inference endpoints: one image, a batch of images, or a video.

All three are plain ``def`` handlers, so FastAPI runs them in a worker thread and
the event loop stays free while the model works.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import config
from ..engines.base import InferenceResult
from ..engines.tiling import predict_tiled
from ..errors import InferenceError, NoModelLoaded, PayloadTooLarge, UnsupportedMedia
from ..media import VideoReader, decode_image, draw, try_decode_image, write_video
from ..state import ModelRegistry
from ..uploads import enforce_file_count, read_capped, safe_filename, save_streaming
from .deps import clamp_confidence, get_registry, parse_classes

log = logging.getLogger("prelabel.api.predict")

router = APIRouter(prefix="/api", tags=["inference"])


@router.post("/predict")
def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    classes: str | None = Form(None),
    tiled: bool = Form(False),
    tile_size: int = Form(0),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Inference on a single image.

    ``tiled=true`` slices the image and runs the model at native resolution on
    each piece — the difference between finding small objects in a large photo
    and downscaling them out of existence. ``classes`` narrows the output to a
    comma-separated list of class ids.
    """
    # Checked up front so "no model" beats "bad image" — the actionable error
    # wins. The authoritative check still happens inside the engine lock.
    if not registry.is_loaded:
        raise NoModelLoaded()

    raw = read_capped(file, config.MAX_MEDIA_MB, "Image")
    image = decode_image(raw)
    wanted = parse_classes(classes)
    threshold = clamp_confidence(conf)

    if tiled:
        result = _run_tiled(registry, image, threshold, wanted, tile_size or None)
    else:
        result = _run(registry, image, threshold, classes=wanted)
    return {"status": "ok", **result.to_dict()}


@router.post("/predict/batch")
def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    classes: str | None = Form(None),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Inference on several images in one batched forward pass.

    This is the throughput path the batch gallery uses: instead of one HTTP
    round-trip (and one cold forward pass) per image, a chunk is sent together
    and the engine runs it as a batch — a large win on a GPU.

    Results come back in the same order as the uploaded files. A file that is too
    large or cannot be decoded is reported in its own slot rather than failing the
    whole chunk.
    """
    files = enforce_file_count(files, config.MAX_BATCH_FILES, "images")
    if not registry.is_loaded:
        raise NoModelLoaded()

    results: list[dict | None] = [None] * len(files)
    images: list[np.ndarray] = []
    slots: list[int] = []  # maps each decoded image back to its file position

    for position, upload in enumerate(files):
        try:
            raw = read_capped(upload, config.MAX_MEDIA_MB, "Image")
        except PayloadTooLarge as exc:
            results[position] = {"status": "error", "detail": exc.detail}
            continue
        image = try_decode_image(raw)
        if image is None:
            results[position] = {"status": "error", "detail": "could not decode image"}
        else:
            images.append(image)
            slots.append(position)

    if images:
        batch = _run_batch(registry, images, clamp_confidence(conf), parse_classes(classes))
        # strict: one result per decoded image is an engine invariant, not a hope.
        for slot, result in zip(slots, batch, strict=True):
            results[slot] = {"status": "ok", **result.to_dict()}

    return {"status": "ok", "results": results}


@router.post("/predict/video")
def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    classes: str | None = Form(None),
    track: bool = Form(False),
    registry: ModelRegistry = Depends(get_registry),
) -> FileResponse:
    """Inference on a video, returning an annotated MP4.

    ``track=true`` follows each object across frames and stamps a stable id on
    it. Without tracking, a video is a stack of unrelated per-frame detections;
    with it, you get trajectories — which is what makes video annotation worth
    anything downstream.

    Both files are temporary: the source is deleted once rendering finishes, and
    the output once the response has been streamed back, so the storage directory
    does not grow without bound.

    Response headers describe what actually happened — which codec was used,
    whether it is playable in a browser, and whether the frame cap sampled the
    clip — so the UI never has to guess.
    """
    if not registry.is_loaded:
        raise NoModelLoaded()

    name = safe_filename(file.filename, ".mp4")
    extension = Path(name).suffix.lower()
    if extension not in config.VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(config.VIDEO_EXTENSIONS))
        raise UnsupportedMedia(f"Unsupported video format '{extension or name}'. Supported: {supported}")

    source = save_streaming(file, config.OUTPUTS_DIR, config.MAX_MEDIA_MB)
    output = config.OUTPUTS_DIR / f"annotated_{uuid.uuid4().hex[:8]}.mp4"
    threshold = clamp_confidence(conf)
    wanted = parse_classes(classes)
    tracked = bool(track) and registry.supports_tracking

    try:
        with VideoReader(str(source), max_frames=config.MAX_VIDEO_FRAMES) as reader:
            size = (reader.width, reader.height) if reader.width and reader.height else None
            frames = _annotate(registry, reader, threshold, wanted, tracked)
            report = write_video(output, frames, reader.output_fps, size)
            sampled, truncated, planned = reader.is_sampled, reader.truncated, reader.planned_frames
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)

    if report.is_empty or not output.exists():
        output.unlink(missing_ok=True)
        raise InferenceError("The video contained no readable frames")

    log.info(
        "Rendered %s: %d frames @ %.2f fps, codec=%s, sampled=%s",
        output.name, report.frames_written, report.fps, report.codec.fourcc, sampled,
    )

    return FileResponse(
        str(output),
        media_type="video/mp4",
        filename=output.name,
        background=BackgroundTask(lambda: output.unlink(missing_ok=True)),
        headers={
            "X-Prelabel-Codec": report.codec.fourcc,
            "X-Prelabel-Browser-Playable": "1" if report.codec.browser_playable else "0",
            "X-Prelabel-Codec-Note": _header_safe(report.codec.note),
            "X-Prelabel-Frames": str(report.frames_written),
            "X-Prelabel-Frames-Planned": str(planned),
            "X-Prelabel-Sampled": "1" if sampled else "0",
            "X-Prelabel-Truncated": "1" if truncated else "0",
            "X-Prelabel-Tracked": "1" if tracked else "0",
        },
    )


# --- internals --------------------------------------------------------------


def _header_safe(value: str) -> str:
    """Reduce a string to something an HTTP header can carry.

    Header values are latin-1. A single non-encodable character raises deep
    inside the response layer and turns a working endpoint into a 500 — so any
    text that reaches a header goes through here, whatever its source.
    """
    return value.encode("latin-1", errors="replace").decode("latin-1")


def _run(
    registry: ModelRegistry,
    image: np.ndarray,
    conf: float,
    classes: Sequence[int] | None = None,
    generation: int | None = None,
    tracking: bool = False,
    reset_tracker: bool = False,
) -> InferenceResult:
    """One inference under the engine lock, with failures reported as 422.

    No pre-warming or other massaging: the reported time is exactly what this
    call took. Use ``/api/benchmark`` for controlled steady-state numbers.

    ``generation`` pins the call to one specific model — see :func:`_annotate`.
    """
    with registry.engine(expect_generation=generation) as engine:
        try:
            if tracking:
                return engine.track(image, conf=conf, classes=classes, reset=reset_tracker)
            return engine.predict(image, conf=conf, classes=classes)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as 422
            log.exception("Inference failed")
            raise InferenceError(f"Inference failed: {exc}") from exc


def _run_batch(
    registry: ModelRegistry,
    images: list[np.ndarray],
    conf: float,
    classes: Sequence[int] | None = None,
) -> list[InferenceResult]:
    with registry.engine() as engine:
        try:
            return engine.predict_batch(images, conf=conf, classes=classes)
        except Exception as exc:  # noqa: BLE001
            log.exception("Batch inference failed")
            raise InferenceError(f"Batch inference failed: {exc}") from exc


def _run_tiled(
    registry: ModelRegistry,
    image: np.ndarray,
    conf: float,
    classes: Sequence[int] | None,
    tile: int | None,
) -> InferenceResult:
    with registry.engine() as engine:
        try:
            return predict_tiled(engine, image, conf=conf, classes=classes, tile=tile)
        except Exception as exc:  # noqa: BLE001
            log.exception("Tiled inference failed")
            raise InferenceError(f"Tiled inference failed: {exc}") from exc


def _annotate(
    registry: ModelRegistry,
    reader: VideoReader,
    conf: float,
    classes: Sequence[int] | None = None,
    tracking: bool = False,
) -> Iterator[np.ndarray]:
    """Yield each sampled frame with its detections drawn on.

    The engine lock is taken per frame rather than held for the whole render, so
    a long video does not block every other request for minutes. That leaves gaps
    between frames in which the model could be replaced — which would annotate
    the rest of the clip with a *different* model and produce a result that looks
    fine and is quietly wrong. Pinning the render to the generation it started
    with turns that into a clear 409 instead.
    """
    generation = registry.generation
    for index, frame in enumerate(reader.frames()):
        result = _run(
            registry, frame, conf,
            classes=classes, generation=generation,
            tracking=tracking,
            # The tracker carries state between calls, so a new clip has to start
            # from nothing or it inherits the previous video's ids.
            reset_tracker=tracking and index == 0,
        )
        yield draw(frame, result)
