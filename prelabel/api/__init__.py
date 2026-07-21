"""HTTP and WebSocket routers.

One router per concern; :func:`prelabel.main.create_app` includes them in this
order.
"""

from fastapi import APIRouter

from . import auth, integrations, models, predict, projects, stream, system

#: Every router, in the order they are mounted.
ROUTERS: tuple[APIRouter, ...] = (
    system.router,
    auth.router,
    models.router,
    predict.router,
    projects.router,
    integrations.router,
    stream.router,
)

__all__ = ["ROUTERS", "auth", "integrations", "models", "predict", "projects", "stream", "system"]
