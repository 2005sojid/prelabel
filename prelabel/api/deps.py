"""Shared dependencies for the API routers.

Everything lives on ``app.state`` and is reached through these functions rather
than as module globals, so a test can build an isolated application — its own
registry, its own database — instead of reaching into shared state.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import Request, WebSocket

from ..jobs import ProjectRunner, TrainingRunner
from ..state import ModelRegistry
from ..store import Store


def get_registry(request: Request) -> ModelRegistry:
    """The application's model registry."""
    return request.app.state.registry


def get_store(request: Request) -> Store:
    """The project database."""
    return request.app.state.store


def get_runner(request: Request) -> ProjectRunner:
    """The background project runner."""
    return request.app.state.runner


def get_trainer(request: Request) -> TrainingRunner:
    """The background training runner."""
    return request.app.state.trainer


def registry_for_socket(websocket: WebSocket) -> ModelRegistry:
    """Same registry, reached from a WebSocket scope."""
    return websocket.app.state.registry


def clamp_confidence(value: float) -> float:
    """Confidence thresholds outside 0–1 are always a client bug; clamp quietly."""
    return min(1.0, max(0.0, float(value)))


def parse_classes(raw: str | None) -> Sequence[int] | None:
    """Parse a ``classes`` form field: ``"0,2,5"`` into ``[0, 2, 5]``.

    Ignores anything that is not an integer rather than failing the request — a
    stray space or trailing comma should not cost the user their inference.
    """
    if not raw or not raw.strip():
        return None
    ids = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return sorted(set(ids)) or None
