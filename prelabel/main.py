"""Application factory.

This module wires things together and nothing else: configuration, middleware,
error handlers and routers. The behaviour lives in :mod:`prelabel.api`
(endpoints), :mod:`prelabel.state` (the loaded model), :mod:`prelabel.store`
(projects) and :mod:`prelabel.engines` (inference).

Run it with ``python run.py``, or point any ASGI server at ``prelabel.main:app``.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__, config, datasets
from .api import ROUTERS
from .errors import register_handlers
from .jobs import ProjectRunner, TrainingRunner
from .media import effective_codec
from .security import AuthMiddleware, OriginGuardMiddleware
from .state import ModelRegistry, clear_stale_slots
from .store import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("prelabel")

DESCRIPTION = "The missing layer between inference and annotation."

#: Response headers the browser is allowed to read cross-origin.
_EXPOSED_HEADERS = [
    "X-Prelabel-Codec",
    "X-Prelabel-Codec-Note",
    "X-Prelabel-Browser-Playable",
    "X-Prelabel-Frames",
    "X-Prelabel-Frames-Planned",
    "X-Prelabel-Sampled",
    "X-Prelabel-Truncated",
    "X-Prelabel-Tracked",
    "X-Prelabel-Images",
    "X-Prelabel-Annotations",
]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage on startup; release everything on shutdown."""
    config.ensure_dirs()
    _clear_outputs()
    clear_stale_slots()
    _resume_interrupted_runs(app.state.store)

    # Probe now rather than on the first request: it takes a moment, and the
    # answer is part of what /api/formats tells the UI on page load.
    codec = effective_codec()
    log.info(
        "Prelabel %s ready · storage=%s · video=%s · auth=%s · dataset roots=%s",
        __version__,
        config.STORAGE_DIR,
        codec.fourcc,
        "on" if config.AUTH_TOKEN else "off",
        len(config.DATA_ROOTS) if datasets.is_configured() else "none",
    )
    try:
        yield
    finally:
        app.state.runner.shutdown()
        app.state.trainer.shutdown()
        app.state.registry.unload()
        app.state.store.close()


def _clear_outputs() -> None:
    """Remove rendered videos and exports left behind by a previous run."""
    for leftover in config.OUTPUTS_DIR.glob("*"):
        try:
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
            else:
                shutil.rmtree(leftover, ignore_errors=True)
        except OSError as exc:
            log.debug("Could not remove %s: %s", leftover, exc)


def _resume_interrupted_runs(store: Store) -> None:
    """Mark runs that were in flight when the process died.

    Nothing is running at startup, so a project still marked ``running`` was
    interrupted. Leaving it that way would show a progress bar that never moves;
    the results already written are kept, and the run can simply be started
    again to pick up the remaining images.
    """
    for project in store.list_projects():
        if project.status == "running":
            store.update_project(
                project.id,
                status="cancelled",
                detail="Interrupted by a server restart — press Run to continue.",
            )
            log.info("Marked interrupted project %s as cancelled", project.id)

        # A training run marked 'running' was likewise interrupted; its weights,
        # if any were written, are still on disk and can be adopted, but the run
        # itself is over.
        if project.training.get("status") == "running":
            training = {
                **project.training,
                "status": "failed",
                "detail": "Interrupted by a server restart.",
            }
            store.update_project(project.id, training=training)
            log.info("Marked interrupted training for %s as failed", project.id)


def create_app() -> FastAPI:
    """Build a fully configured application.

    A factory rather than a module-level singleton so tests can create isolated
    instances — each gets its own registry, database and runner.
    """
    app = FastAPI(
        title="Prelabel",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    config.ensure_dirs()
    app.state.registry = ModelRegistry()
    app.state.store = Store(config.DATABASE_PATH)
    app.state.runner = ProjectRunner(app.state.store, app.state.registry)
    app.state.trainer = TrainingRunner(app.state.store, app.state.registry)

    register_handlers(app)

    # Added innermost-first. Auth runs before the origin guard so an
    # unauthenticated cross-origin request is reported as unauthenticated, and
    # both sit inside CORS so their rejections still carry CORS headers.
    app.add_middleware(AuthMiddleware, token=config.AUTH_TOKEN, public_paths=config.PUBLIC_PATHS)
    app.add_middleware(OriginGuardMiddleware, allowed_origins=config.ALLOWED_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS or [],
        allow_origin_regex=None,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=_EXPOSED_HEADERS,
    )

    for router in ROUTERS:
        app.include_router(router)

    assets = config.FRONTEND_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    return app


app = create_app()
