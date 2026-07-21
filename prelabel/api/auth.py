"""Signing in, when a token is configured.

Exchanging the token for a cookie once beats attaching a header to every request:
the gallery is full of ``<img src="...">`` tags, and an image tag cannot carry an
Authorization header. The cookie is ``HttpOnly`` so page scripts cannot read it,
and :class:`~prelabel.security.OriginGuardMiddleware` already blocks the
cross-site requests that cookie auth would otherwise invite.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from .. import config
from ..errors import Unauthorized
from ..security import AUTH_COOKIE, presented_token, token_matches

log = logging.getLogger("prelabel.api.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

#: A month. Long enough not to nag, short enough to expire on a shared machine.
COOKIE_MAX_AGE = 30 * 24 * 3600


@router.get("/status")
def status(request: Request) -> dict:
    """Whether a token is required, and whether this client has one."""
    required = bool(config.AUTH_TOKEN)
    authenticated = not required or token_matches(
        presented_token(request.scope.get("headers") or []), config.AUTH_TOKEN
    )
    return {"required": required, "authenticated": authenticated}


@router.post("/login")
async def login(request: Request, response: Response) -> dict:
    """Exchange the shared token for a session cookie."""
    if not config.AUTH_TOKEN:
        return {"status": "ok", "required": False, "authenticated": True}

    body = await _json(request)
    if not token_matches(str(body.get("token", "")), config.AUTH_TOKEN):
        log.warning("Failed login attempt from %s", request.client.host if request.client else "?")
        raise Unauthorized("That token is not correct")

    response.set_cookie(
        AUTH_COOKIE,
        config.AUTH_TOKEN,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        # Only over HTTPS when the request itself arrived that way, so a plain
        # http://127.0.0.1 session still works.
        secure=request.url.scheme == "https",
    )
    return {"status": "ok", "required": True, "authenticated": True}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(AUTH_COOKIE)
    return {"status": "ok", "authenticated": False}


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty body is just a failed login
        return {}
    return body if isinstance(body, dict) else {}
