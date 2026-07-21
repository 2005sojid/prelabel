"""Projects end to end: create, run, review, export, import.

The run itself is a background thread, so these tests wait for it to finish
rather than assuming it has — the same thing the UI does.
"""

from __future__ import annotations

import io
import json
import time
import zipfile

from prelabel import config


def wait_for(client, project_id: str, status: str = "done", timeout: float = 30.0) -> dict:
    """Poll a project until it reaches ``status``, or fail with what it did reach."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/projects/{project_id}").json()
        if body["status"] == status:
            return body
        if body["status"] in ("failed", "done", "cancelled") and body["status"] != status:
            break
        time.sleep(0.05)
    raise AssertionError(f"project ended as {body.get('status')!r} ({body.get('detail')!r}), expected {status!r}")


# --- discovery --------------------------------------------------------------


def test_roots_reports_the_feature_is_off_by_default(client, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOTS", [])
    body = client.get("/api/projects/-/roots").json()
    assert body["configured"] is False
    assert "PL_DATA_ROOTS" in body["hint"]


def test_roots_lists_configured_directories(client, dataset_root):
    body = client.get("/api/projects/-/roots").json()
    assert body["configured"] is True
    assert str(dataset_root) in body["roots"]


def test_browse_lists_subdirectories_and_counts_images(client, dataset_root):
    body = client.get("/api/projects/-/browse", params={"path": str(dataset_root)}).json()
    assert [entry["name"] for entry in body["directories"]] == ["sub"]
    assert body["image_count"] == 4


def test_browse_refuses_a_path_outside_the_roots(client, dataset_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    response = client.get("/api/projects/-/browse", params={"path": str(outside)})
    assert response.status_code == 400
    assert "outside the allowed" in response.json()["detail"]


# --- lifecycle --------------------------------------------------------------


def test_create_registers_every_image(client, project):
    assert project["stats"]["total"] == 4
    assert project["stats"]["pending"] == 4
    assert project["status"] == "new"


def test_create_refuses_an_empty_folder(client, dataset_root):
    empty = dataset_root / "empty"
    empty.mkdir()
    response = client.post("/api/projects", json={"path": str(empty)})
    assert response.status_code == 400
    assert "No images" in response.json()["detail"]


def test_create_refuses_an_unallowed_path(client, dataset_root, tmp_path):
    response = client.post("/api/projects", json={"path": str(tmp_path / "nope")})
    assert response.status_code == 400


def test_projects_are_listed(client, project):
    listed = client.get("/api/projects").json()["projects"]
    assert [p["id"] for p in listed] == [project["id"]]


def test_rename(client, project):
    body = client.patch(f"/api/projects/{project['id']}", json={"name": "Renamed"}).json()
    assert body["name"] == "Renamed"


def test_delete(client, project):
    assert client.delete(f"/api/projects/{project['id']}").status_code == 200
    assert client.get(f"/api/projects/{project['id']}").status_code == 404


def test_missing_project_is_404(client):
    assert client.get("/api/projects/nope").status_code == 404


def test_rescan_picks_up_new_images(client, project, dataset_root):
    (dataset_root / "added.jpg").write_bytes((dataset_root / "a.jpg").read_bytes())
    body = client.post(f"/api/projects/{project['id']}/rescan").json()
    assert body["added"] == 1
    assert body["stats"]["total"] == 5


# --- running ----------------------------------------------------------------


def test_running_without_a_model_is_refused(client, project):
    response = client.post(f"/api/projects/{project['id']}/run")
    assert response.status_code == 409
    assert "model" in response.json()["detail"].lower()


def test_a_run_processes_every_image(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    body = wait_for(client, project["id"])

    assert body["stats"]["done"] == 4
    assert body["stats"]["pending"] == 0
    assert body["stats"]["detections"] == 4  # the stub finds one per image
    assert body["model"]["name"] == "stub"


def test_results_survive_and_are_readable(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    items = client.get(f"/api/projects/{project['id']}/items").json()["items"]
    assert len(items) == 4
    assert all(item["status"] == "done" for item in items)
    assert items[0]["detections"][0]["class_name"] == "car"
    assert items[0]["width"] > 0


def test_an_unreadable_file_fails_only_itself(client, project, loaded, dataset_root):
    (dataset_root / "broken.jpg").write_bytes(b"not an image at all")
    client.post(f"/api/projects/{project['id']}/rescan")
    client.post(f"/api/projects/{project['id']}/run")
    body = wait_for(client, project["id"])

    assert body["stats"]["failed"] == 1
    assert body["stats"]["done"] == 4


def test_a_second_run_is_refused_while_one_is_active(app, client, project, loaded):
    """One model, one lock: two runs would take turns and lie about progress."""
    import threading

    from prelabel.jobs import RunSettings

    # Hold the worker inside its first batch so the run is genuinely in flight.
    released = threading.Event()
    original = loaded.predict_batch
    loaded.predict_batch = lambda *args, **kwargs: (released.wait(5), original(*args, **kwargs))[1]

    try:
        app.state.runner.start(project["id"], RunSettings())
        response = client.post(f"/api/projects/{project['id']}/run")

        assert response.status_code == 409
        assert "still running" in response.json()["detail"]
    finally:
        released.set()
        app.state.runner.shutdown()


def test_cancelling_a_project_that_is_not_running_is_409(client, project, loaded):
    assert client.post(f"/api/projects/{project['id']}/cancel").status_code == 409


def test_restart_reprocesses_everything(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    calls_after_first = loaded.calls

    client.post(f"/api/projects/{project['id']}/run", json={"restart": True})
    wait_for(client, project["id"])

    assert loaded.calls > calls_after_first
    assert client.get(f"/api/projects/{project['id']}").json()["stats"]["done"] == 4


def test_a_resumed_run_only_does_what_is_left(client, project, loaded):
    """Resuming must not redo work already stored."""
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    calls_after_first = loaded.calls

    client.post(f"/api/projects/{project['id']}/run")  # no restart
    wait_for(client, project["id"])

    assert loaded.calls == calls_after_first, "a resumed run reprocessed finished images"


def test_class_filter_reaches_the_engine(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run", json={"settings": {"classes": [7]}})
    wait_for(client, project["id"])
    assert loaded.last_classes == [7]


# --- review -----------------------------------------------------------------


def test_items_can_be_ordered_by_review_priority(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    response = client.get(f"/api/projects/{project['id']}/items", params={"order": "priority"})
    assert response.status_code == 200
    assert len(response.json()["items"]) == 4


def test_items_can_be_searched(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    items = client.get(f"/api/projects/{project['id']}/items", params={"search": "sub/"}).json()["items"]
    assert [item["rel_path"] for item in items] == ["sub/nested.jpg"]


def test_full_image_and_thumbnail_are_served(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    item = client.get(f"/api/projects/{project['id']}/items").json()["items"][0]

    full = client.get(f"/api/projects/{project['id']}/items/{item['id']}/image")
    thumb = client.get(f"/api/projects/{project['id']}/items/{item['id']}/image", params={"thumb": True})

    assert full.status_code == 200 and len(full.content) > 0
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_a_missing_item_image_is_404(client, project):
    assert client.get(f"/api/projects/{project['id']}/items/9999/image").status_code == 404


# --- export / import --------------------------------------------------------


def test_export_produces_a_valid_coco_archive(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    response = client.get(f"/api/projects/{project['id']}/export", params={"format": "coco"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert int(response.headers["x-prelabel-images"]) == 4

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "annotations/instances_default.json" in names
        assert sum(1 for n in names if n.startswith("images/default/")) == 4

        document = json.loads(archive.read("annotations/instances_default.json"))
        assert len(document["images"]) == 4
        assert len(document["annotations"]) == 4
        assert document["categories"][0]["id"] == 1  # COCO ids are 1-based
        assert all(a["category_id"] >= 1 for a in document["annotations"])


def test_export_to_imagenet_groups_by_class(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    response = client.get(f"/api/projects/{project['id']}/export", params={"format": "imagenet"})
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert all(name.startswith("car/") for name in archive.namelist())


def test_export_rejects_an_unknown_format(client, project, loaded):
    response = client.get(f"/api/projects/{project['id']}/export", params={"format": "parquet"})
    assert response.status_code == 400


def test_export_leaves_no_temporary_file_behind(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    response = client.get(f"/api/projects/{project['id']}/export")
    response.read()  # drain so the cleanup task runs
    assert list(config.OUTPUTS_DIR.glob("export_*")) == []


def test_import_replaces_predictions_with_corrections(client, project, loaded):
    """The round trip: export, correct elsewhere, bring it back."""
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 32, "height": 24}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1, 1, 4, 4]},
        ],
        "categories": [{"id": 1, "name": "corrected"}, {"id": 2, "name": "extra"}],
    }
    response = client.post(
        f"/api/projects/{project['id']}/import",
        files={"file": ("labels.json", json.dumps(coco).encode(), "application/json")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["matched"] == 1
    assert body["annotations"] == 2

    items = client.get(f"/api/projects/{project['id']}/items", params={"search": "a.jpg"}).json()["items"]
    names = {d["class_name"] for d in items[0]["detections"]}
    assert names == {"corrected", "extra"}
    assert all(d["confidence"] == 1.0 for d in items[0]["detections"])


def test_import_reports_files_it_could_not_match(client, project, loaded):
    coco = {
        "images": [{"id": 1, "file_name": "not_in_this_project.jpg", "width": 10, "height": 10}],
        "annotations": [],
        "categories": [],
    }
    body = client.post(
        f"/api/projects/{project['id']}/import",
        files={"file": ("labels.json", json.dumps(coco).encode(), "application/json")},
    ).json()
    assert body["matched"] == 0
    assert body["unmatched"] == 1
    assert "not_in_this_project.jpg" in body["unknown_files"]


def test_import_rejects_a_file_that_is_not_coco(client, project):
    response = client.post(
        f"/api/projects/{project['id']}/import",
        files={"file": ("labels.json", b"{\"nope\": true}", "application/json")},
    )
    assert response.status_code == 400
    assert "images" in response.json()["detail"]


def test_import_rejects_broken_json(client, project):
    response = client.post(
        f"/api/projects/{project['id']}/import",
        files={"file": ("labels.json", b"{not json", "application/json")},
    )
    assert response.status_code == 400
