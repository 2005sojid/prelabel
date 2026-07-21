"""Real-time inference over a WebSocket, used by the webcam mode.

The browser sends ``{"image": <data-url>, "conf": <float>}`` per frame and gets
back the same JSON a ``/api/predict`` call would return. The client only sends
the next frame after receiving a result, so the server never queues up work it
cannot keep up with.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import config
from ..errors import NoModelLoaded, PrelabelError
from ..media import decode_data_url
from ..state import ModelRegistry
from .deps import clamp_confidence, registry_for_socket

log = logging.getLogger("prelabel.api.stream")

router = APIRouter()


@router.websocket("/api/stream")
async def stream(websocket: WebSocket) -> None:
    """Frame in, detections out, until the client disconnects."""
    registry = registry_for_socket(websocket)
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            await websocket.send_json(await _handle_frame(registry, message))
    except WebSocketDisconnect:
        pass
    except (ValueError, TypeError) as exc:
        # A malformed (non-JSON) payload: tell the client, then let it reconnect.
        log.info("Closing stream after malformed message: %s", exc)
        await _close_quietly(websocket)


async def _handle_frame(registry: ModelRegistry, message: Any) -> dict:
    """Run one frame, converting any failure into an error payload.

    Errors are sent rather than raised so a single bad frame — a truncated
    data-URL, a momentary shape mismatch — does not tear down the stream.
    """
    try:
        if not isinstance(message, dict):
            raise ValueError("expected a JSON object")
        if not registry.is_loaded:
            raise NoModelLoaded()

        image = decode_data_url(message.get("image", ""), config.MAX_STREAM_FRAME_MB * 1024 * 1024)
        conf = clamp_confidence(message.get("conf", config.DEFAULT_CONF))

        # Run the blocking forward pass in a worker thread so the event loop
        # (and any other client) stays responsive.
        result = await asyncio.to_thread(_predict, registry, image, conf)
        return {"status": "ok", **result.to_dict()}
    except PrelabelError as exc:
        return {"status": "error", "detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 - report, keep the socket open
        log.exception("Stream frame failed")
        return {"status": "error", "detail": str(exc)}


def _predict(registry: ModelRegistry, image, conf: float):  # noqa: ANN001, ANN202
    with registry.engine() as engine:
        return engine.predict(image, conf=conf)


async def _close_quietly(websocket: WebSocket) -> None:
    try:
        await websocket.close(code=1003)  # unsupported data
    except RuntimeError:
        pass  # already closed
