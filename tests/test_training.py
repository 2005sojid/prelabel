"""Fine-tuning on a project's labels: dataset building, the runner, the API.

The dataset builder is the correctness-critical part and gets the most coverage
here — it turns stored pixel boxes into the normalised YOLO form, and a bug there
would train the model on the wrong geometry without ever raising. The actual
training call is exercised once, end to end, behind the ``integration`` marker.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from prelabel import config, training
from prelabel.store import Store
from prelabel.training import TrainingSettings, build_yolo_dataset
from tests.helpers import jpeg_bytes

# --- settings ---------------------------------------------------------------


def test_settings_clamp_out_of_range_values():
    settings = TrainingSettings.from_dict(
        {"epochs": 100000, "imgsz": 999, "batch": 9999, "val_fraction": 0.99, "source": "nonsense"}
    )
    assert settings.epochs == 1000            # capped
    assert settings.imgsz == 992              # 999 rounded to a multiple of 32
    assert settings.batch == 128              # capped
    assert settings.val_fraction == 0.5       # capped
    assert settings.source == "current"       # unknown source falls back


def test_settings_defaults_come_from_config():
    settings = TrainingSettings.from_dict({})
    assert settings.epochs == config.TRAIN_EPOCHS
    assert settings.source == "current"


# --- dataset building -------------------------------------------------------


def box(x1, y1, x2, y2, name="car", conf=1.0):
    return {"class_name": name, "confidence": conf, "kind": "box", "box": [x1, y1, x2, y2]}


@pytest.fixture
def images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder of real (if tiny) images, allowed as a dataset root."""
    root = (tmp_path / "imgs").resolve()
    (root / "sub").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (root / name).write_bytes(jpeg_bytes(100, 100))
    (root / "sub" / "a.jpg").write_bytes(jpeg_bytes(100, 100))  # same stem, different folder
    monkeypatch.setattr(config, "DATA_ROOTS", [root])
    return root


@pytest.fixture
def store(tmp_path: Path) -> Store:
    instance = Store(tmp_path / "train.db")
    yield instance
    instance.close()


def _seed(store: Store, root: Path, labels: dict[str, list[dict]]) -> str:
    """Create a project over ``root`` and give each named item its detections."""
    project = store.create_project("Train", str(root))
    store.add_items(project.id, list(labels.keys()))
    for item in store.list_items(project.id, limit=100):
        store.save_result(
            project.id, item.id, width=100, height=100, task="detect",
            inference_ms=1, review_priority=0, detections=labels.get(item.rel_path, []),
        )
    return project.id


def test_builder_writes_the_yolo_layout(store, images, tmp_path):
    pid = _seed(store, images, {
        "a.jpg": [box(0, 0, 10, 10)],
        "b.jpg": [box(5, 5, 20, 20)],
        "c.jpg": [box(0, 0, 50, 50)],
        "d.jpg": [],  # no boxes — a background image, left out of v1
    })
    dest = tmp_path / "ds"
    summary = build_yolo_dataset(store, store.get_project(pid), dest, TrainingSettings(), {0: "car"})

    assert (dest / "data.yaml").exists()
    assert summary.images == 3           # the empty one is excluded
    assert summary.train + summary.val == 3
    assert summary.val >= 1              # always something to measure on
    # Every image copied has a matching label file with the same stem.
    for split in ("train", "val"):
        for image in (dest / "images" / split).glob("*"):
            assert (dest / "labels" / split / f"{image.stem}.txt").exists()


def test_builder_normalises_boxes(store, images, tmp_path):
    pid = _seed(store, images, {"a.jpg": [box(10, 20, 50, 60)]})
    dest = tmp_path / "ds"
    build_yolo_dataset(store, store.get_project(pid), dest, TrainingSettings(val_fraction=0.05), {0: "car"})

    label = next((dest / "labels").rglob("*.txt")).read_text().strip()
    cls, cx, cy, bw, bh = label.split()
    # box (10,20)-(50,60) on 100x100 → centre (30,40), size (40,40).
    assert cls == "0"
    assert (float(cx), float(cy), float(bw), float(bh)) == (0.3, 0.4, 0.4, 0.4)


def test_builder_keeps_base_model_classes(store, images, tmp_path):
    """A subset of the model's classes must not shrink its detection head."""
    pid = _seed(store, images, {"a.jpg": [box(0, 0, 10, 10, "car")]})
    dest = tmp_path / "ds"
    build_yolo_dataset(store, store.get_project(pid), dest,
                       TrainingSettings(val_fraction=0.05), {0: "person", 1: "car"})

    names = (dest / "data.yaml").read_text()
    assert '0: "person"' in names and '1: "car"' in names
    # 'car' is index 1, the model's own index — so the head lines up.
    assert next((dest / "labels").rglob("*.txt")).read_text().split()[0] == "1"


def test_builder_appends_unknown_classes(store, images, tmp_path):
    pid = _seed(store, images, {"a.jpg": [box(0, 0, 10, 10, "dog")]})
    dest = tmp_path / "ds"
    build_yolo_dataset(store, store.get_project(pid), dest,
                       TrainingSettings(val_fraction=0.05), {0: "person", 1: "car"})

    names = (dest / "data.yaml").read_text()
    assert '2: "dog"' in names  # a new class grows the head rather than vanishing


def test_builder_drops_low_confidence_boxes(store, images, tmp_path):
    pid = _seed(store, images, {
        "a.jpg": [box(0, 0, 10, 10, conf=0.9), box(20, 20, 30, 30, conf=0.1)],
    })
    dest = tmp_path / "ds"
    build_yolo_dataset(store, store.get_project(pid), dest,
                       TrainingSettings(val_fraction=0.05, min_conf=0.25), {0: "car"})

    lines = next((dest / "labels").rglob("*.txt")).read_text().strip().splitlines()
    assert len(lines) == 1  # the 0.1 box was noise, filtered out


def test_builder_refuses_when_there_is_nothing_to_learn(store, images, tmp_path):
    pid = _seed(store, images, {"a.jpg": [], "b.jpg": []})
    with pytest.raises(ValueError, match="[Nn]othing to train"):
        build_yolo_dataset(store, store.get_project(pid), tmp_path / "ds", TrainingSettings(), {0: "car"})


def test_builder_gives_colliding_stems_distinct_files(store, images, tmp_path):
    """``a.jpg`` and ``sub/a.jpg`` must not overwrite each other's labels."""
    pid = _seed(store, images, {
        "a.jpg": [box(0, 0, 10, 10)],
        "sub/a.jpg": [box(5, 5, 15, 15)],
    })
    dest = tmp_path / "ds"
    summary = build_yolo_dataset(store, store.get_project(pid), dest,
                                 TrainingSettings(val_fraction=0.05), {0: "car"})

    assert summary.images == 2
    labels = list((dest / "labels").rglob("*.txt"))
    assert len({p.stem for p in labels}) == 2  # two distinct stems, nothing clobbered


def test_split_is_deterministic():
    a = training._split("images/x.jpg", 0.2)
    b = training._split("images/x.jpg", 0.2)
    assert a == b  # same path, same side, every run — so metrics stay comparable


# --- store: state round-trip and migration ----------------------------------


def test_training_state_survives_a_round_trip(store):
    project = store.create_project("t", "/data")
    store.update_project(project.id, training={"status": "done", "metrics": {"map50": 0.7}})
    assert store.get_project(project.id).training == {"status": "done", "metrics": {"map50": 0.7}}


def test_a_v2_database_gains_the_training_column(tmp_path):
    """Opening a database written before retraining existed must add the column."""
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, source_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new', detail TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            model_json TEXT NOT NULL DEFAULT '{}', settings_json TEXT NOT NULL DEFAULT '{}',
            comparison_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
            rel_path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            review_priority REAL NOT NULL DEFAULT 0,
            baseline_json TEXT NOT NULL DEFAULT '[]',
            baseline_count INTEGER NOT NULL DEFAULT 0, disputed INTEGER NOT NULL DEFAULT 0,
            agreement REAL NOT NULL DEFAULT 1.0, UNIQUE(project_id, rel_path)
        );
        INSERT INTO projects (id, name, source_dir, created_at, updated_at)
        VALUES ('p1', 'Old', '/data', '2026-01-01', '2026-01-01');
        """
    )
    old.commit()
    old.close()

    store = Store(path)
    try:
        columns = {row["name"] for row in store._connection.execute("PRAGMA table_info(projects)")}  # noqa: SLF001
        assert "training_json" in columns
        assert store.get_project("p1").training == {}
    finally:
        store.close()


# --- the API ----------------------------------------------------------------


def test_train_is_refused_without_a_model(client, project):
    response = client.post(f"/api/projects/{project['id']}/train")
    assert response.status_code == 409
    assert "PyTorch" in response.json()["detail"]


def test_train_is_refused_for_a_non_detection_model(client, project, registry):
    from tests.helpers import StubEngine

    slot = registry.new_slot()
    registry._engine = StubEngine(task="classify")  # noqa: SLF001
    registry._slot = slot                            # noqa: SLF001
    registry._target = slot / "stub.pt"              # noqa: SLF001

    response = client.post(f"/api/projects/{project['id']}/train")
    assert response.status_code == 409
    assert "detection" in response.json()["detail"]


def test_run_is_refused_while_training(client, project):
    # White-box: mark a training active without spawning a real one.
    client.app.state.trainer._active = project["id"]  # noqa: SLF001
    try:
        response = client.post(f"/api/projects/{project['id']}/run")
        assert response.status_code == 409
        assert "training" in response.json()["detail"].lower()
    finally:
        client.app.state.trainer._active = None  # noqa: SLF001


def test_adopt_needs_a_finished_training(client, project):
    response = client.post(f"/api/projects/{project['id']}/train/adopt")
    assert response.status_code == 409


def test_training_status_starts_idle(client, project):
    body = client.get(f"/api/projects/{project['id']}/training").json()
    assert body["active"] is False
    assert body.get("status") in (None, "")


def test_the_runner_drives_state_to_done(client, project, registry, monkeypatch, tmp_path):
    """Full orchestration with the training call stubbed: running → done → adopt.

    Exercises the runner, the stored state transitions and the adopt path without
    ultralytics, so it stays in the fast suite.
    """
    from tests.helpers import StubEngine

    slot = registry.new_slot()
    registry._engine = StubEngine(task="detect")   # noqa: SLF001
    registry._slot = slot                          # noqa: SLF001
    checkpoint = slot / "stub.pt"
    checkpoint.write_bytes(b"not really weights")   # so copy2 has something to copy
    registry._target = checkpoint                  # noqa: SLF001

    def fake_build(store, proj, dest, settings, base_names=None):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "data.yaml").write_text("names:\n  0: car\n")
        return training.DatasetSummary(dest / "data.yaml", images=3, boxes=3, train=2, val=1, skipped_empty=0, classes=["car"])

    def fake_train(weights, data_yaml, run_dir, settings, on_progress=None, should_cancel=None):
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        best = run_dir / "weights" / "best.pt"
        best.write_bytes(b"trained")
        if on_progress:
            on_progress(1, settings.epochs, {"map50": 0.42})
        return training.TrainingResult(best, {"map50": 0.42, "map": 0.3})

    monkeypatch.setattr(training, "build_yolo_dataset", fake_build)
    monkeypatch.setattr(training, "train_yolo", fake_train)

    started = client.post(f"/api/projects/{project['id']}/train", json={"settings": {"epochs": 1}})
    assert started.status_code == 200, started.text

    deadline = time.time() + 10
    state = {}
    while time.time() < deadline:
        state = client.get(f"/api/projects/{project['id']}/training").json()
        if state.get("status") in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "done", state
    assert state["metrics"]["map50"] == 0.42
    assert Path(state["weights"]).exists()

    # Adopt without a real engine build.
    monkeypatch.setattr(registry, "load", lambda target, s, **kw: {"name": "retrained.pt", "task": "detect"})
    adopted = client.post(f"/api/projects/{project['id']}/train/adopt")
    assert adopted.status_code == 200, adopted.text
    assert adopted.json()["model"]["name"] == "retrained.pt"


# --- the real thing ---------------------------------------------------------


@pytest.mark.integration
def test_a_real_fine_tune_produces_loadable_weights(store, images, tmp_path):
    """One real epoch: build → train → the weights load back as a model.

    Small and fast (imgsz 64, one epoch) — it proves the pipeline end to end, not
    that the model learned anything from four black squares.
    """
    ultralytics = pytest.importorskip("ultralytics")

    pid = _seed(store, images, {
        "a.jpg": [box(10, 10, 60, 60)],
        "b.jpg": [box(20, 20, 80, 80)],
        "c.jpg": [box(0, 0, 40, 40)],
        "d.jpg": [box(30, 30, 90, 90)],
    })
    dest = tmp_path / "ds"
    summary = build_yolo_dataset(store, store.get_project(pid), dest,
                                 TrainingSettings(val_fraction=0.25), {0: "car"})
    assert summary.images == 4

    try:
        base = ultralytics.YOLO("yolov8n.pt")
        weights = Path(base.ckpt_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not obtain base model: {exc}")

    progress: list[int] = []
    result = training.train_yolo(
        weights, summary.data_yaml, tmp_path / "run",
        TrainingSettings(epochs=1, imgsz=64, batch=2, val_fraction=0.25),
        on_progress=lambda epoch, total, metrics: progress.append(epoch),
    )

    assert result.weights.exists()
    assert progress and progress[0] == 1         # the epoch callback fired, clamped to the total
    assert max(progress) <= 1                     # never reports past the requested epoch count
    # The produced checkpoint is a real, loadable model.
    reloaded = ultralytics.YOLO(str(result.weights))
    assert reloaded.task == "detect"
