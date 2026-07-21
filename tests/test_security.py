"""The origin guard.

Why this matters: a cross-origin ``POST`` of ``multipart/form-data`` is a CORS
*simple request*. No preflight is sent, and CORS only prevents the attacking page
from *reading* the reply — the request itself arrives and executes. Since
``POST /api/model`` loads a model, and loading a ``.pt`` unpickles arbitrary
code, an unguarded server is remote code execution for any page the user has open.
"""

from __future__ import annotations

import pytest

from prelabel.security import SAFE_METHODS, is_origin_allowed
from tests.helpers import upload

# --- the rule ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "host", "allowed", "expected"),
    [
        # No Origin at all: not a browser. curl, scripts and the test suite.
        (None, "127.0.0.1:8000", [], True),
        ("", "127.0.0.1:8000", [], True),
        # Same origin as the server: the bundled UI.
        ("http://127.0.0.1:8000", "127.0.0.1:8000", [], True),
        ("http://localhost:8000", "localhost:8000", [], True),
        ("https://app.example.com", "app.example.com", [], True),
        # Cross origin: rejected unless explicitly allowed.
        ("https://evil.example", "127.0.0.1:8000", [], False),
        ("http://127.0.0.1:9999", "127.0.0.1:8000", [], False),
        ("http://evil.localhost", "localhost:8000", [], False),
        # A sandboxed iframe or a file:// page sends "null" — never trusted.
        ("null", "127.0.0.1:8000", [], False),
        # Explicit allow-list.
        ("https://studio.example", "127.0.0.1:8000", ["https://studio.example"], True),
        ("https://other.example", "127.0.0.1:8000", ["https://studio.example"], False),
    ],
)
def test_is_origin_allowed(origin, host, allowed, expected):
    assert is_origin_allowed(origin, host, allowed) is expected


def test_safe_methods_are_the_read_only_ones():
    assert frozenset({"GET", "HEAD", "OPTIONS"}) == SAFE_METHODS


# --- over HTTP --------------------------------------------------------------

HOSTILE = {"Origin": "https://evil.example"}


def test_reads_are_allowed_from_anywhere(client):
    """GET cannot change state, so it needs no origin check."""
    assert client.get("/api/health", headers=HOSTILE).status_code == 200
    assert client.get("/api/formats", headers=HOSTILE).status_code == 200


def test_cross_origin_model_upload_is_rejected(client):
    """The attack this guard exists for: a drive-by model load."""
    response = client.post(
        "/api/model",
        files={"files": ("evil.pt", b"pickled payload", "application/octet-stream")},
        headers=HOSTILE,
    )
    assert response.status_code == 403
    assert response.json()["status"] == "error"


@pytest.mark.parametrize(
    "path",
    ["/api/predict", "/api/predict/batch", "/api/predict/video", "/api/device", "/api/benchmark"],
)
def test_every_state_changing_endpoint_is_guarded(client, loaded, path):
    assert client.post(path, headers=HOSTILE).status_code == 403


def test_delete_is_guarded(client, loaded):
    assert client.delete("/api/model", headers=HOSTILE).status_code == 403
    assert client.get("/api/health").json()["model_loaded"] is True


def test_same_origin_requests_pass(client, loaded):
    response = client.post(
        "/api/predict",
        files={"file": upload()},
        headers={"Origin": "http://testserver", "Host": "testserver"},
    )
    assert response.status_code == 200


def test_requests_without_an_origin_pass(client, loaded):
    """Native clients — curl, scripts — keep working."""
    assert client.post("/api/predict", files={"file": upload()}).status_code == 200


def test_allow_list_is_honoured(monkeypatch):
    from fastapi.testclient import TestClient

    from prelabel import config
    from prelabel.main import create_app

    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["https://studio.example"])
    with TestClient(create_app()) as client:
        allowed = client.post("/api/benchmark", headers={"Origin": "https://studio.example"})
        blocked = client.post("/api/benchmark", headers={"Origin": "https://evil.example"})

    assert allowed.status_code == 400   # reached the handler: no model loaded
    assert blocked.status_code == 403   # never reached it


# --- over WebSocket ---------------------------------------------------------


def test_cross_origin_websocket_is_rejected(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/stream", headers=HOSTILE):
        pass


def test_same_origin_websocket_is_accepted(client, loaded):
    with client.websocket_connect("/api/stream", headers={"Origin": "http://testserver"}) as socket:
        socket.send_json({"image": "data:image/jpeg;base64,"})
        assert socket.receive_json()["status"] == "error"  # bad frame, but connected
