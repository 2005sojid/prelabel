"""Serving other annotation tools directly.

If you correct labels in CVAT or Label Studio, the useful place for a model is
*inside* that tool: you press "annotate" and the boxes appear, instead of
exporting a dataset, pre-labelling it elsewhere, and importing it back.

Both tools speak simple HTTP to an external model, and both are supported here:

* **Label Studio** expects an ML backend with ``GET /health``, ``POST /setup``
  and ``POST /predict``, returning its own result shape (percent coordinates).
* **CVAT** expects a Nuclio-style function that takes a base64 image and returns
  a flat list of shapes in pixel coordinates.

Neither needs anything installed on their side beyond the URL of this server.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from .. import config
from ..errors import NoModelLoaded, UnsupportedMedia
from ..media import decode_image
from ..state import ModelRegistry
from .deps import clamp_confidence, get_registry

log = logging.getLogger("prelabel.api.integrations")

router = APIRouter(tags=["integrations"])

#: Label Studio maps a task to a labelling config; these are the control types we
#: can populate, by task.
LABEL_STUDIO_CONTROLS = {
    "detect": ("RectangleLabels", "rectanglelabels"),
    "obb": ("RectangleLabels", "rectanglelabels"),
    "segment": ("PolygonLabels", "polygonlabels"),
    "pose": ("KeyPointLabels", "keypointlabels"),
    "classify": ("Choices", "choices"),
}


def _decode(payload: str) -> Any:
    """Decode a base64 image, tolerating a data-URL prefix."""
    if not payload:
        raise UnsupportedMedia("No image supplied")
    encoded = payload.partition(",")[2] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(encoded, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedMedia(f"Malformed base64 image: {exc}") from exc
    return decode_image(raw)


def _run(registry: ModelRegistry, image, conf: float):  # noqa: ANN001
    with registry.engine() as engine:
        return engine.predict(image, conf=conf)


# --- CVAT -------------------------------------------------------------------


@router.get("/api/cvat/info")
def cvat_info(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Model description, in the shape CVAT's function metadata expects."""
    info = registry.info()
    if info is None:
        raise NoModelLoaded()
    spec = [
        {"id": int(class_id), "name": str(name), "type": _cvat_shape(info["task"])}
        for class_id, name in info["class_names"].items()
    ]
    return {
        "name": f"prelabel-{info['name']}",
        "type": "detector",
        "framework": info["backend"],
        "task": info["task"],
        "spec": spec,
    }


def _cvat_shape(task: str) -> str:
    return {"segment": "polygon", "pose": "points", "obb": "rectangle"}.get(task, "rectangle")


@router.post("/api/cvat/invoke")
async def cvat_invoke(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> list[dict]:
    """Run one image for CVAT.

    Body: ``{"image": "<base64>", "threshold": 0.5}``. The reply is the flat list
    of shapes CVAT's serverless detector interface reads, in pixel coordinates.
    """
    body = await request.json()
    image = _decode(str(body.get("image", "")))
    conf = clamp_confidence(body.get("threshold", config.DEFAULT_CONF))

    result = _run(registry, image, conf)
    shapes: list[dict] = []
    for detection in result.detections:
        shape = _to_cvat_shape(detection, result.task)
        if shape is not None:
            shapes.append(shape)
    return shapes


def _to_cvat_shape(detection, task: str) -> dict | None:  # noqa: ANN001
    label = detection.class_name
    confidence = str(round(float(detection.confidence), 4))

    if task == "segment" and detection.mask:
        return {
            "confidence": confidence, "label": label, "type": "polygon",
            "points": [round(float(v), 2) for point in detection.mask for v in point],
        }
    if task == "pose" and detection.keypoints:
        return {
            "confidence": confidence, "label": label, "type": "points",
            "points": [round(float(p[0]), 2) for p in detection.keypoints]
                      + [round(float(p[1]), 2) for p in detection.keypoints],
        }
    if detection.box:
        return {
            "confidence": confidence, "label": label, "type": "rectangle",
            "points": [round(float(v), 2) for v in detection.box],
        }
    if task == "classify":
        return {"confidence": confidence, "label": label, "type": "tag", "points": []}
    return None


# --- Label Studio -----------------------------------------------------------


@router.get("/api/label-studio/health")
def label_studio_health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Liveness probe Label Studio polls before sending work."""
    info = registry.info()
    return {
        "status": "UP" if info else "NO_MODEL",
        "model_class": info["name"] if info else None,
    }


@router.post("/api/label-studio/setup")
async def label_studio_setup(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Handshake: Label Studio sends the project's labelling config."""
    info = registry.info()
    if info is None:
        raise NoModelLoaded()
    return {"model_version": f"{info['name']}:{info['task']}"}


@router.post("/api/label-studio/predict")
async def label_studio_predict(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Predict for a batch of Label Studio tasks.

    Label Studio expresses geometry in **percentages of the image**, not pixels,
    which is the one thing that makes this more than a re-serialisation.
    """
    body = await request.json()
    tasks = body.get("tasks") or []
    info = registry.info()
    if info is None:
        raise NoModelLoaded()

    from_name, result_type = LABEL_STUDIO_CONTROLS.get(info["task"], LABEL_STUDIO_CONTROLS["detect"])
    conf = clamp_confidence(body.get("threshold", config.DEFAULT_CONF))

    predictions = []
    for task in tasks:
        try:
            image = _decode(_image_payload(task))
        except UnsupportedMedia as exc:
            log.warning("Skipping Label Studio task: %s", exc)
            predictions.append({"result": [], "score": 0.0, "model_version": info["name"]})
            continue

        result = _run(registry, image, conf)
        height, width = (result.image_shape or [image.shape[0], image.shape[1]])[:2]
        entries = [
            entry for detection in result.detections
            if (entry := _to_label_studio(detection, result.task, width, height, from_name, result_type))
        ]
        scores = [float(d.confidence) for d in result.detections]
        predictions.append({
            "result": entries,
            "score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "model_version": info["name"],
        })

    return {"results": predictions}


def _image_payload(task: dict) -> str:
    """Pull the base64 image out of a Label Studio task."""
    data = task.get("data") or {}
    for key in ("image", "image_base64", "img"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    raise UnsupportedMedia("Task has no inline image; Prelabel needs base64 image data")


def _to_label_studio(
    detection,  # noqa: ANN001
    task: str,
    width: int,
    height: int,
    from_name: str,
    result_type: str,
) -> dict | None:
    if not width or not height:
        return None

    base = {
        "from_name": from_name,
        "to_name": "image",
        "type": result_type,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "score": round(float(detection.confidence), 4),
    }
    percent = lambda value, extent: round(100.0 * float(value) / extent, 4)  # noqa: E731

    if task == "classify":
        return {**base, "value": {"choices": [detection.class_name]}}

    if task == "segment" and detection.mask:
        return {**base, "value": {
            "points": [[percent(x, width), percent(y, height)] for x, y in detection.mask],
            "polygonlabels": [detection.class_name],
        }}

    if task == "pose" and detection.keypoints:
        # Label Studio keypoints are one result each, so the first is emitted
        # here and the rest would need separate entries; a single grouped shape
        # keeps the response valid and reviewable.
        first = detection.keypoints[0]
        return {**base, "value": {
            "x": percent(first[0], width), "y": percent(first[1], height),
            "width": 0.5, "keypointlabels": [detection.class_name],
        }}

    if detection.box:
        x1, y1, x2, y2 = detection.box
        return {**base, "value": {
            "x": percent(x1, width), "y": percent(y1, height),
            "width": percent(x2 - x1, width), "height": percent(y2 - y1, height),
            "rotation": 0, "rectanglelabels": [detection.class_name],
        }}
    return None
