"""Application errors and the handlers that turn them into JSON.

Every failure the client can hit is one of these types. Routers raise them and
never build error responses themselves, so the wire format is defined in exactly
one place.

The rule the whole API follows: **an error is always JSON**. A plain-text
``Internal Server Error`` body makes the frontend's ``response.json()`` throw a
confusing "... is not valid JSON", hiding the real problem.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config

log = logging.getLogger("prelabel.errors")


class PrelabelError(Exception):
    """Base class for errors that map onto a specific HTTP status."""

    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NoModelLoaded(PrelabelError):
    """No model is loaded, so the request cannot be served."""

    status_code = 400

    def __init__(self, detail: str = "No model loaded") -> None:
        super().__init__(detail)


class ModelNotFound(NoModelLoaded):
    """Nothing to describe — used by ``GET /api/model``, where absence is a 404.

    Distinct from :class:`NoModelLoaded` (400) because *asking about* a model
    that isn't there is a missing resource, whereas *acting* without one is a bad
    request.
    """

    status_code = 404


class ModelLoadError(PrelabelError):
    """The uploaded files could not be turned into a working engine."""

    status_code = 400


class UnsupportedMedia(PrelabelError):
    """The upload is not a format this endpoint accepts."""

    status_code = 400


class PayloadTooLarge(PrelabelError):
    """The upload exceeds a configured size or count limit."""

    status_code = 413


class InferenceError(PrelabelError):
    """The model was loaded but failed to run on this input."""

    status_code = 422


class ModelChanged(PrelabelError):
    """The model was replaced while a multi-step job was still using it.

    Relevant to video rendering, which runs frame by frame over a long period.
    Continuing would annotate the second half of a clip with a different model
    than the first — a result that looks fine and is quietly wrong.
    """

    status_code = 409

    def __init__(self, detail: str = "The model was changed while this job was running") -> None:
        super().__init__(detail)


class OriginRejected(PrelabelError):
    """A state-changing request came from an untrusted browser origin."""

    status_code = 403


class Unauthorized(PrelabelError):
    """A token is required and was missing or wrong."""

    status_code = 401

    def __init__(self, detail: str = "Authentication required") -> None:
        super().__init__(detail)


class DatasetAccessError(PrelabelError):
    """A dataset folder is not configured, not allowed, or not readable."""

    status_code = 400


class NotFound(PrelabelError):
    """A project or item that does not exist."""

    status_code = 404


class Conflict(PrelabelError):
    """The request contradicts the resource's current state."""

    status_code = 409


def _payload(detail: str) -> dict:
    return {"status": "error", "detail": detail}


def register_handlers(app: FastAPI) -> None:
    """Install the JSON error handlers on ``app``."""

    @app.exception_handler(PrelabelError)
    async def _prelabel_error(request: Request, exc: PrelabelError):  # noqa: ANN202
        if exc.status_code >= 500:
            log.exception("Error on %s", request.url.path)
        else:
            log.info("%s on %s: %s", type(exc).__name__, request.url.path, exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_payload(exc.detail))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):  # noqa: ANN202
        return JSONResponse(status_code=exc.status_code, content=_payload(str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):  # noqa: ANN202
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "invalid request")
        detail = f"{field}: {message}" if field else message
        return JSONResponse(status_code=422, content=_payload(detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        log.exception("Unhandled error on %s", request.url.path)
        detail = str(exc) if config.VERBOSE_ERRORS else "Internal server error"
        return JSONResponse(status_code=500, content=_payload(detail))
