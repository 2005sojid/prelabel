"""Comparing two annotation sets over a project, through the API.

The three situations this is for, all the same machinery:
model A vs model B, model vs corrected ground truth, before vs after retraining.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from prelabel import comparison
from prelabel.store import Store
from tests.test_projects import wait_for


def box(x1, y1, x2, y2, name="car"):
    return {"class_name": name, "confidence": 0.9, "kind": "box", "box": [x1, y1, x2, y2]}


@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path / "cmp.db")
    yield instance
    instance.close()


@pytest.fixture
def seeded(store):
    """A project whose three items already have results."""
    project = store.create_project("Compare", "/data")
    store.add_items(project.id, ["a.jpg", "b.jpg", "c.jpg"])
    for item in store.list_items(project.id):
        store.save_result(project.id, item.id, width=100, height=100, task="detect",
                          inference_ms=5, review_priority=0.5, detections=[box(0, 0, 10, 10)])
    return store, project


# --- migration --------------------------------------------------------------


def test_a_database_from_the_previous_version_is_upgraded(tmp_path):
    """The upgrade path a user actually walks: an existing projects database.

    The comparison columns did not exist in v1, and one of the indexes is built
    over them — so the migration has to run between creating the tables and
    creating the indexes, or opening an old database raises.
    """
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, source_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new', detail TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            model_json TEXT NOT NULL DEFAULT '{}', settings_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
            rel_path TEXT NOT NULL, width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
            detail TEXT NOT NULL DEFAULT '', task TEXT NOT NULL DEFAULT '',
            inference_ms REAL NOT NULL DEFAULT 0, review_priority REAL NOT NULL DEFAULT 0,
            detection_count INTEGER NOT NULL DEFAULT 0,
            detections_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE(project_id, rel_path)
        );
        """
    )
    old.execute(
        "INSERT INTO projects (id, name, source_dir, created_at, updated_at)"
        " VALUES ('p1', 'Old', '/data', '2020-01-01', '2020-01-01')"
    )
    old.execute(
        "INSERT INTO items (project_id, rel_path, status, detection_count, detections_json)"
        " VALUES ('p1', 'a.jpg', 'done', 1, ?)",
        (json.dumps([box(0, 0, 10, 10)]),),
    )
    old.commit()
    old.close()

    upgraded = Store(path)
    try:
        project = upgraded.get_project("p1")
        assert project is not None, "the existing project was lost"
        assert project.comparison == {}

        item = upgraded.list_items("p1")[0]
        assert item.detection_count == 1, "existing results were lost"
        assert item.baseline == []
        assert item.agreement == 1.0
    finally:
        upgraded.close()


# --- capturing --------------------------------------------------------------


def test_capture_copies_the_current_annotations(seeded):
    store, project = seeded
    assert store.baseline_size(project.id) == 0

    captured = store.capture_baseline(project.id)

    assert captured == 3
    assert store.baseline_size(project.id) == 3
    item = store.list_items(project.id)[0]
    assert item.baseline == item.detections


def test_capture_is_a_copy_not_a_reference(seeded):
    """The point is to keep what this run said while the next one overwrites it."""
    store, project = seeded
    store.capture_baseline(project.id)

    item = store.list_items(project.id)[0]
    store.save_result(project.id, item.id, width=100, height=100, task="detect",
                      inference_ms=5, review_priority=0.5,
                      detections=[box(50, 50, 60, 60, "bus")])

    updated = store.get_item(project.id, item.id)
    assert updated.detections[0]["class_name"] == "bus"
    assert updated.baseline[0]["class_name"] == "car", "the baseline moved with the current set"


def test_clearing_the_baseline_leaves_the_annotations_alone(seeded):
    store, project = seeded
    store.capture_baseline(project.id)
    store.clear_baseline(project.id)

    assert store.baseline_size(project.id) == 0
    assert store.list_items(project.id)[0].detection_count == 1


def test_a_restart_keeps_the_baseline_but_drops_the_verdict(seeded):
    """Re-running is how you get a second set — the baseline has to survive it.

    The per-item scores must not: they describe results the restart just
    deleted, and a re-run that is cancelled half-way would otherwise leave
    stale disagreement counts on images it never reached.
    """
    store, project = seeded
    store.capture_baseline(project.id)
    item = store.list_items(project.id)[0]
    store.save_result(project.id, item.id, width=100, height=100, task="detect",
                      inference_ms=5, review_priority=0.5,
                      detections=[box(80, 80, 90, 90, "bus")])
    comparison.refresh(store, project.id)
    assert store.get_item(project.id, item.id).disputed == 2

    store.reset_items(project.id)

    assert store.baseline_size(project.id) == 3
    after = store.get_item(project.id, item.id)
    assert (after.disputed, after.agreement, after.detection_count) == (0, 0, 0)
    assert after.baseline_count == 1


# --- the project-wide diff --------------------------------------------------


def test_refresh_scores_every_item_and_caches_the_summary(seeded):
    store, project = seeded
    store.capture_baseline(project.id)

    # One item now disagrees: the object moved away and a new class appeared.
    item = store.list_items(project.id)[0]
    store.save_result(project.id, item.id, width=100, height=100, task="detect",
                      inference_ms=5, review_priority=0.5,
                      detections=[box(80, 80, 90, 90, "bus")])

    result = comparison.refresh(store, project.id)

    assert result.compared == 3
    assert result.summary.counts["agreed"] == 2
    assert result.summary.counts["missing"] == 1
    assert result.summary.counts["added"] == 1

    # Cached on the project, so reading it later costs nothing.
    assert store.get_project(project.id).comparison["compared"] == 3
    # And on the item, so the queue can sort in SQL.
    assert store.get_item(project.id, item.id).disputed == 2


def test_describe_reads_the_cache_without_recomputing(seeded, monkeypatch):
    store, project = seeded
    store.capture_baseline(project.id)
    comparison.refresh(store, project.id)

    def explode(*args, **kwargs):
        raise AssertionError("describe() recomputed the diff")

    monkeypatch.setattr(comparison, "compare_item", explode)
    described = comparison.describe(store, project.id)

    assert described["available"] is True
    assert described["compared"] == 3


def test_describe_says_so_when_nothing_has_been_compared(seeded):
    store, project = seeded
    assert comparison.describe(store, project.id)["available"] is False


def test_items_can_be_ordered_by_disagreement(seeded):
    store, project = seeded
    store.capture_baseline(project.id)

    loud = store.list_items(project.id)[2]
    store.save_result(project.id, loud.id, width=100, height=100, task="detect",
                      inference_ms=5, review_priority=0.5,
                      detections=[box(80, 80, 90, 90, "bus"), box(20, 20, 30, 30, "dog")])
    comparison.refresh(store, project.id)

    ordered = store.list_items(project.id, order="disputed")
    assert ordered[0].rel_path == loud.rel_path
    assert ordered[0].disputed > ordered[-1].disputed


def test_only_disputed_and_only_agreed_filters(seeded):
    store, project = seeded
    store.capture_baseline(project.id)
    item = store.list_items(project.id)[0]
    store.save_result(project.id, item.id, width=100, height=100, task="detect",
                      inference_ms=5, review_priority=0.5, detections=[])
    comparison.refresh(store, project.id)

    assert len(store.list_items(project.id, only="disputed")) == 1
    assert len(store.list_items(project.id, only="agreed")) == 2


# --- through the API --------------------------------------------------------


def test_capture_needs_something_to_capture(client, project):
    assert client.post(f"/api/projects/{project['id']}/baseline").status_code == 409


def test_comparison_is_unavailable_until_a_baseline_exists(client, project):
    body = client.get(f"/api/projects/{project['id']}/comparison").json()
    assert body["available"] is False
    assert body["baseline_items"] == 0


def test_the_full_two_model_flow(client, project, loaded):
    """Run, capture, change the model's answers, re-run, diff."""
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    captured = client.post(f"/api/projects/{project['id']}/baseline").json()
    assert captured["captured"] == 4
    assert captured["comparison"]["agreement"] == 1.0  # identical to itself

    # Re-run as if with a second model. The stub answers the same way, so this
    # asserts the mechanism holds an agreement rather than inventing conflict.
    client.post(f"/api/projects/{project['id']}/run", json={"restart": True})
    wait_for(client, project["id"])

    body = client.get(f"/api/projects/{project['id']}/comparison").json()
    assert body["available"] is True
    assert body["compared"] == 4
    # Same stub, same answers: everything should still agree.
    assert body["counts"]["agreed"] == 4


def test_importing_ground_truth_as_a_baseline_produces_a_diff(client, project, loaded):
    """The label-QA case: corrected labels beside the model's own output."""
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])

    items = client.get(f"/api/projects/{project['id']}/items").json()["items"]
    truth = {
        "images": [{"id": 1, "file_name": items[0]["name"], "width": 32, "height": 24}],
        # The human says "van" where the model said "car", in the same place.
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 2, 2]}],
        "categories": [{"id": 1, "name": "van"}],
    }

    response = client.post(
        f"/api/projects/{project['id']}/import?into=baseline",
        files={"file": ("truth.json", json.dumps(truth).encode(), "application/json")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["into"] == "baseline"
    assert body["matched"] == 1

    diff = body["comparison"]
    assert diff["counts"]["reclassified"] == 1
    assert diff["reclassifications"][0]["swap"] == "van -> car"


def test_importing_into_the_baseline_does_not_touch_the_model_output(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    items = client.get(f"/api/projects/{project['id']}/items").json()["items"]
    before = items[0]["detections"]

    truth = {
        "images": [{"id": 1, "file_name": items[0]["name"], "width": 32, "height": 24}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 2, 2]}],
        "categories": [{"id": 1, "name": "van"}],
    }
    client.post(
        f"/api/projects/{project['id']}/import?into=baseline",
        files={"file": ("truth.json", json.dumps(truth).encode(), "application/json")},
    )

    after = client.get(f"/api/projects/{project['id']}/items").json()["items"][0]
    assert after["detections"] == before
    assert after["baseline"][0]["class_name"] == "van"


def test_import_rejects_an_unknown_target(client, project):
    response = client.post(
        f"/api/projects/{project['id']}/import?into=nowhere",
        files={"file": ("x.json", b"{}", "application/json")},
    )
    assert response.status_code == 400


def test_per_item_comparison_lists_every_difference(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    client.post(f"/api/projects/{project['id']}/baseline")

    item = client.get(f"/api/projects/{project['id']}/items").json()["items"][0]
    body = client.get(f"/api/projects/{project['id']}/items/{item['id']}/comparison").json()

    assert body["has_baseline"] is True
    assert body["counts"]["agreed"] == 1
    assert body["pairings"][0]["kind"] == "agreed"
    assert body["baseline"] and body["current"]


def test_clearing_the_baseline_through_the_api(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    client.post(f"/api/projects/{project['id']}/baseline")

    assert client.delete(f"/api/projects/{project['id']}/baseline").status_code == 200
    body = client.get(f"/api/projects/{project['id']}/comparison").json()
    assert body["available"] is False
    assert body["baseline_items"] == 0


def test_recomputing_at_a_different_overlap_threshold(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    client.post(f"/api/projects/{project['id']}/baseline")

    body = client.post(f"/api/projects/{project['id']}/comparison?iou=0.9").json()
    assert body["compared"] == 4


def test_recompute_without_a_baseline_is_409(client, project, loaded):
    client.post(f"/api/projects/{project['id']}/run")
    wait_for(client, project["id"])
    assert client.post(f"/api/projects/{project['id']}/comparison").status_code == 409
