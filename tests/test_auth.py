"""Token authentication.

Off by default — a loopback tool that asks its only user to log in is friction
with no security gain. When it is on, it applies to everything except the
liveness probe and the login endpoint itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prelabel import config
from prelabel.main import create_app
from prelabel.security import AUTH_COOKIE, presented_token, token_matches

TOKEN = "s3cret-token"


@pytest.fixture
def secured(monkeypatch):
    """An application with authentication switched on."""
    monkeypatch.setattr(config, "AUTH_TOKEN", TOKEN)
    with TestClient(create_app()) as client:
        yield client


# --- the primitives ---------------------------------------------------------


def test_token_comparison_rejects_a_wrong_value():
    assert token_matches(TOKEN, TOKEN) is True
    assert token_matches("wrong", TOKEN) is False
    assert token_matches(None, TOKEN) is False
    assert token_matches("", TOKEN) is False


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ([(b"authorization", b"Bearer abc")], "abc"),
        ([(b"authorization", b"bearer abc")], "abc"),
        ([(b"x-prelabel-token", b"abc")], "abc"),
        ([(b"cookie", b"other=1; prelabel_token=abc; more=2")], "abc"),
        ([(b"authorization", b"Basic abc")], None),
        ([], None),
    ],
)
def test_token_is_read_from_every_accepted_place(headers, expected):
    assert presented_token(headers) == expected


# --- disabled by default ----------------------------------------------------


def test_no_token_configured_means_open(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/formats").status_code == 200
    assert client.get("/api/auth/status").json() == {"required": False, "authenticated": True}


# --- enabled ----------------------------------------------------------------


def test_requests_without_a_token_are_rejected(secured):
    response = secured.get("/api/formats")
    assert response.status_code == 401
    assert response.json()["status"] == "error"


def test_health_stays_public_for_container_probes(secured):
    """An orchestrator's liveness check cannot carry a token."""
    assert secured.get("/api/health").status_code == 200


def test_a_bearer_token_is_accepted(secured):
    response = secured.get("/api/formats", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200


def test_a_direct_header_is_accepted(secured):
    assert secured.get("/api/formats", headers={"X-Prelabel-Token": TOKEN}).status_code == 200


def test_a_wrong_token_is_rejected(secured):
    assert secured.get("/api/formats", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_login_sets_a_cookie_that_then_works(secured):
    """The cookie exists because ``<img src>`` cannot carry a header."""
    response = secured.post("/api/auth/login", json={"token": TOKEN})
    assert response.status_code == 200
    assert AUTH_COOKIE in response.cookies or AUTH_COOKIE in secured.cookies

    # The client keeps the cookie, so the next call needs no header.
    assert secured.get("/api/formats").status_code == 200


def test_the_session_cookie_is_not_readable_by_scripts(secured):
    response = secured.post("/api/auth/login", json={"token": TOKEN})
    cookie_header = response.headers.get("set-cookie", "")
    assert "httponly" in cookie_header.lower()
    assert "samesite=strict" in cookie_header.lower()


def test_login_with_a_wrong_token_is_401(secured):
    assert secured.post("/api/auth/login", json={"token": "nope"}).status_code == 401


def test_login_with_no_body_is_401(secured):
    assert secured.post("/api/auth/login").status_code == 401


def test_logout_clears_the_session(secured):
    secured.post("/api/auth/login", json={"token": TOKEN})
    assert secured.get("/api/formats").status_code == 200

    secured.post("/api/auth/logout")
    secured.cookies.clear()
    assert secured.get("/api/formats").status_code == 401


def test_auth_status_reports_the_requirement(secured):
    before = secured.get("/api/auth/status").json()
    assert before == {"required": True, "authenticated": False}

    secured.post("/api/auth/login", json={"token": TOKEN})
    assert secured.get("/api/auth/status").json()["authenticated"] is True


def test_websockets_are_guarded_too(secured):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), secured.websocket_connect("/api/stream"):
        pass


def test_projects_are_guarded(secured):
    assert secured.get("/api/projects").status_code == 401
    assert secured.post("/api/projects", json={"path": "/tmp"}).status_code == 401


def test_integrations_are_guarded(secured):
    """A CVAT server has to authenticate like anything else."""
    assert secured.post("/api/cvat/invoke", json={"image": ""}).status_code == 401
    assert secured.get("/api/label-studio/health").status_code == 401
