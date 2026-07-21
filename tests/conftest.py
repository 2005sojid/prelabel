"""Shared fixtures.

Every test gets an isolated storage directory and its own application instance,
so nothing leaks between tests and none of them touch the developer's real
``storage/`` folder.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prelabel import config
from prelabel.main import create_app
from tests.helpers import StubEngine, jpeg_bytes


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every storage path at a per-test temporary directory."""
    storage = tmp_path / "storage"
    models = storage / "models"
    outputs = storage / "outputs"

    monkeypatch.setattr(config, "STORAGE_DIR", storage)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(config, "PENDING_DIR", models / "pending")
    # Derived at import time from the real STORAGE_DIR, so it needs redirecting
    # too — otherwise every test shares one database in the developer's tree.
    monkeypatch.setattr(config, "DATABASE_PATH", storage / "prelabel.db")

    models.mkdir(parents=True)
    outputs.mkdir(parents=True)
    return storage


@pytest.fixture
def app():
    """A fresh application, with its own model registry."""
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    """Client with lifespan run, so startup and shutdown are exercised."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registry(app):
    return app.state.registry


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def loaded(registry, stub_engine):
    """Install a stub engine as the active model, bypassing a real load."""
    slot = registry.new_slot()
    registry._engine = stub_engine        # noqa: SLF001 - deliberate test seam
    registry._slot = slot                 # noqa: SLF001
    registry._target = slot / "stub.pt"   # noqa: SLF001
    return stub_engine


@pytest.fixture
def jpeg() -> bytes:
    return jpeg_bytes()


@pytest.fixture
def dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder of real images, allowed as a dataset root.

    Folder projects are off unless ``PL_DATA_ROOTS`` names a directory, so a test
    that wants them has to allow one — the same explicit step a user takes.
    """
    root = (tmp_path / "images").resolve()
    (root / "sub").mkdir(parents=True)

    for name in ("a.jpg", "b.jpg", "c.png"):
        (root / name).write_bytes(jpeg_bytes(32, 24))
    (root / "sub" / "nested.jpg").write_bytes(jpeg_bytes(16, 16))
    (root / "notes.txt").write_bytes(b"not an image")

    monkeypatch.setattr(config, "DATA_ROOTS", [root])
    return root


@pytest.fixture
def project(client, dataset_root):
    """A created project over ``dataset_root``, not yet run."""
    response = client.post("/api/projects", json={"name": "Test", "path": str(dataset_root)})
    assert response.status_code == 201, response.text
    return response.json()
