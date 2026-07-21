"""Origin checking for state-changing requests.

Why this exists
---------------
A cross-origin ``POST`` of ``multipart/form-data`` is a CORS *simple request*:
the browser sends it without a preflight, and CORS only stops the page from
*reading* the reply. The request itself still arrives and still executes.

For Prelabel that is a real problem, because ``POST /api/model`` loads a model
and loading a ``.pt`` unpickles arbitrary code. Without this guard, any page the
user happens to have open could upload a model to their local server and get
code execution — no interaction required.

The rule
--------
For every state-changing request (and the WebSocket handshake):

* **No ``Origin`` header** → allow. Browsers always send one on cross-origin
  state-changing requests, so its absence means a non-browser client: ``curl``,
  a script, the test suite.
* **``Origin`` matches the host the request was sent to** → allow. This is the
  bundled UI talking to its own server.
* **``Origin`` is in ``PL_ALLOWED_ORIGINS``** → allow.
* Anything else → ``403``.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit

log = logging.getLogger("prelabel.security")

#: Methods that cannot change server state and therefore need no origin check.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_REJECTION = json.dumps(
    {
        "status": "error",
        "detail": (
            "Cross-origin request rejected. Prelabel only accepts requests from its own "
            "page; set PL_ALLOWED_ORIGINS to allow another origin."
        ),
    }
).encode("utf-8")


def _header(headers: Iterable[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _authority(origin: str) -> str:
    """Reduce an origin URL to ``host:port`` for comparison against ``Host``."""
    parts = urlsplit(origin)
    return parts.netloc.lower() if parts.netloc else origin.lower()


def is_origin_allowed(origin: str | None, host: str | None, allowed: Sequence[str]) -> bool:
    """Decide whether ``origin`` may make a state-changing request to ``host``."""
    if not origin:
        return True  # not a browser
    if origin.lower() == "null":
        return False  # sandboxed iframe / file:// — never trusted
    if host and _authority(origin) == host.lower():
        return True
    return any(origin.lower() == _authority(entry) or origin.lower() == entry.lower() for entry in allowed)


class OriginGuardMiddleware:
    """Pure-ASGI middleware so it covers WebSocket handshakes as well as HTTP."""

    def __init__(self, app, allowed_origins: Sequence[str] = ()) -> None:
        self.app = app
        self.allowed_origins = list(allowed_origins)

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if kind == "http" and scope.get("method", "GET").upper() in SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        origin = _header(headers, b"origin")
        host = _header(headers, b"host")

        if is_origin_allowed(origin, host, self.allowed_origins):
            await self.app(scope, receive, send)
            return

        log.warning("Rejected %s from origin %r (host %r)", scope.get("path"), origin, host)
        if kind == "websocket":
            await receive()  # consume websocket.connect so the close is well-formed
            await send({"type": "websocket.close", "code": 1008})
            return

        await _reject(send, 403, _REJECTION)


async def _reject(send, status: int, body: bytes) -> None:  # noqa: ANN001
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --- authentication ---------------------------------------------------------

#: Cookie the browser gets after a successful login.
#:
#: A cookie rather than a header because ``<img src="...">`` cannot carry one,
#: and the gallery is full of image tags. It is ``HttpOnly`` so page scripts
#: cannot read it, and the origin guard above already blocks the cross-site
#: request forgery that cookie auth would otherwise invite.
AUTH_COOKIE = "prelabel_token"

_UNAUTHORIZED = json.dumps(
    {"status": "error", "detail": "Authentication required. Send the token or sign in."}
).encode("utf-8")


def _cookie_value(cookie_header: str | None, name: str) -> str | None:
    if not cookie_header:
        return None
    for part in cookie_header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return None


def presented_token(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """The token this request carries, from any of the accepted places."""
    authorization = _header(headers, b"authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    direct = _header(headers, b"x-prelabel-token")
    if direct:
        return direct.strip()
    return _cookie_value(_header(headers, b"cookie"), AUTH_COOKIE)


def token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison, so a wrong token leaks nothing by timing."""
    if not presented:
        return False
    return secrets.compare_digest(presented, expected)


class AuthMiddleware:
    """Requires a shared token when one is configured.

    Off by default: a tool bound to loopback that asks its only user to log in is
    friction with no security gain. Setting ``PL_AUTH_TOKEN`` turns it on for
    every request except the liveness probe and the login endpoint itself.
    """

    def __init__(self, app, token: str = "", public_paths: Sequence[str] = ()) -> None:
        self.app = app
        self.token = token
        self.public_paths = tuple(public_paths)

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def _is_public(self, path: str) -> bool:
        return path in self.public_paths or path.startswith("/api/auth/")

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if not self.enabled or scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if self._is_public(path):
            await self.app(scope, receive, send)
            return

        if token_matches(presented_token(scope.get("headers") or []), self.token):
            await self.app(scope, receive, send)
            return

        log.warning("Unauthenticated request to %s", path)
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        await _reject(send, 401, _UNAUTHORIZED)
