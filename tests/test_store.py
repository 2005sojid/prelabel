"""The project database.

Covers the parts that are easy to get subtly wrong: re-adding the same image,
paging, the active-learning ordering, and concurrent access — the runner writes
from its own thread while the UI reads from the request threads.
"""

from __future__ import annotations

import threading

import pytest

from prelabel.store import Store


@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path / "test.db")
    yield instance
    instance.close()


@pytest.fixture
def seeded(store):
    project = store.create_project("Demo", "/data/images")
    store.add_items(project.id, ["a.jpg", "b.jpg", "sub/c.jpg"])
    return store, project


# --- projects ---------------------------------------------------------------


def test_create_and_read_back(store):
    project = store.create_project("My project", "/data/images", {"conf": 0.3})
    loaded = store.get_project(project.id)

    assert loaded is not None
    assert loaded.name == "My project"
    assert loaded.source_dir == "/data/images"
    assert loaded.settings == {"conf": 0.3}
    assert loaded.status == "new"
    assert loaded.created_at


def test_a_blank_name_gets_a_placeholder(store):
    assert store.create_project("   ", "/data").name == "Untitled"


def test_missing_project_is_none(store):
    assert store.get_project("nope") is None


def test_projects_list_newest_first(store):
    first = store.create_project("first", "/a")
    second = store.create_project("second", "/b")
    store.update_project(first.id, name="first again")  # touches updated_at

    assert [p.id for p in store.list_projects()][0] == first.id
    assert {p.id for p in store.list_projects()} == {first.id, second.id}


def test_update_ignores_unknown_fields(store):
    project = store.create_project("x", "/a")
    store.update_project(project.id, name="renamed", nonsense="ignored")
    assert store.get_project(project.id).name == "renamed"


def test_delete_removes_the_project_and_its_items(seeded):
    store, project = seeded
    assert store.delete_project(project.id) is True
    assert store.get_project(project.id) is None
    assert store.list_items(project.id) == []


def test_deleting_a_missing_project_is_false(store):
    assert store.delete_project("nope") is False


# --- items ------------------------------------------------------------------


def test_items_are_registered_once(seeded):
    """Rescanning a folder must not duplicate what is already known."""
    store, project = seeded
    added = store.add_items(project.id, ["a.jpg", "b.jpg", "new.jpg"])
    assert added == 1
    assert len(store.list_items(project.id)) == 4


def test_pending_items_are_returned_in_order(seeded):
    store, project = seeded
    pending = store.pending_items(project.id, limit=2)
    assert len(pending) == 2
    assert all(item.status == "pending" for item in pending)


def test_saving_a_result_updates_everything_the_gallery_needs(seeded):
    store, project = seeded
    item = store.list_items(project.id)[0]

    store.save_result(
        project.id, item.id,
        width=640, height=480, task="detect",
        inference_ms=12.5, review_priority=0.8,
        detections=[{"class_name": "car", "confidence": 0.9, "kind": "box", "box": [1, 2, 3, 4]}],
    )

    updated = store.get_item(project.id, item.id)
    assert updated.status == "done"
    assert (updated.width, updated.height) == (640, 480)
    assert updated.detection_count == 1
    assert updated.detections[0]["class_name"] == "car"
    assert updated.review_priority == 0.8


def test_saving_an_error_records_why(seeded):
    store, project = seeded
    item = store.list_items(project.id)[0]
    store.save_error(project.id, item.id, "not a readable image")
    assert store.get_item(project.id, item.id).status == "error"
    assert "readable" in store.get_item(project.id, item.id).detail


def test_reset_returns_everything_to_pending(seeded):
    store, project = seeded
    item = store.list_items(project.id)[0]
    store.save_result(project.id, item.id, width=1, height=1, task="detect",
                      inference_ms=1, review_priority=1, detections=[{"class_name": "a", "confidence": 1}])

    store.reset_items(project.id)

    assert all(i.status == "pending" and i.detection_count == 0 for i in store.list_items(project.id))


# --- querying ---------------------------------------------------------------


@pytest.fixture
def with_results(seeded):
    store, project = seeded
    items = store.list_items(project.id)
    # a.jpg: 2 detections, confident.  b.jpg: 0.  sub/c.jpg: 1, uncertain.
    store.save_result(project.id, items[0].id, width=10, height=10, task="detect",
                      inference_ms=5, review_priority=0.1,
                      detections=[{"class_name": "car", "confidence": 0.95},
                                  {"class_name": "van", "confidence": 0.9}])
    store.save_result(project.id, items[1].id, width=10, height=10, task="detect",
                      inference_ms=5, review_priority=1.0, detections=[])
    store.save_result(project.id, items[2].id, width=10, height=10, task="detect",
                      inference_ms=5, review_priority=0.95,
                      detections=[{"class_name": "car", "confidence": 0.52}])
    return store, project


def test_filter_by_having_detections(with_results):
    store, project = with_results
    assert len(store.list_items(project.id, only="with")) == 2
    assert len(store.list_items(project.id, only="without")) == 1


def test_search_matches_the_path(with_results):
    store, project = with_results
    found = store.list_items(project.id, search="sub/")
    assert [item.rel_path for item in found] == ["sub/c.jpg"]


def test_priority_order_puts_the_least_certain_first(with_results):
    """The active-learning view: review where the model is most likely wrong."""
    store, project = with_results
    order = [item.rel_path for item in store.list_items(project.id, order="priority")]
    assert order[0] == "b.jpg"      # found nothing at all
    assert order[-1] == "a.jpg"     # confident, least in need of a look


def test_order_by_detection_count(with_results):
    store, project = with_results
    assert store.list_items(project.id, order="most")[0].detection_count == 2
    assert store.list_items(project.id, order="least")[0].detection_count == 0


def test_paging(with_results):
    store, project = with_results
    page = store.list_items(project.id, offset=1, limit=1)
    assert len(page) == 1
    assert page[0].rel_path == "b.jpg"


def test_detections_can_be_left_out_of_a_listing(with_results):
    """The gallery needs counts, not polygons; skipping them keeps pages small."""
    store, project = with_results
    assert store.list_items(project.id, with_detections=False)[0].detections == []


def test_stats(with_results):
    store, project = with_results
    stats = store.stats(project.id)
    assert stats == {"total": 3, "done": 3, "failed": 0, "pending": 0,
                     "detections": 3, "average_ms": 5.0}


def test_class_counts_are_ranked(with_results):
    store, project = with_results
    assert store.class_counts(project.id) == [
        {"class_name": "car", "count": 2},
        {"class_name": "van", "count": 1},
    ]


def test_iterating_done_items_streams_every_one(with_results):
    store, project = with_results
    assert len(list(store.iter_done_items(project.id, chunk=2))) == 3


# --- concurrency ------------------------------------------------------------


def test_writes_from_several_threads_all_land(store):
    """The runner writes from its own thread while requests read from others."""
    project = store.create_project("concurrent", "/data")
    paths = [f"img_{i:03d}.jpg" for i in range(60)]
    store.add_items(project.id, paths)
    items = store.list_items(project.id, limit=100)

    errors: list[BaseException] = []

    def worker(subset):
        try:
            for item in subset:
                store.save_result(project.id, item.id, width=8, height=8, task="detect",
                                  inference_ms=1.0, review_priority=0.5,
                                  detections=[{"class_name": "x", "confidence": 0.5}])
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(items[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert store.stats(project.id)["done"] == 60


def test_reads_work_while_another_thread_writes(store):
    project = store.create_project("mixed", "/data")
    store.add_items(project.id, [f"{i}.jpg" for i in range(40)])
    stop = threading.Event()
    seen: list[int] = []

    def reader():
        while not stop.is_set():
            seen.append(store.stats(project.id)["done"])

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for item in store.list_items(project.id, limit=100):
            store.save_result(project.id, item.id, width=1, height=1, task="detect",
                              inference_ms=1, review_priority=0, detections=[])
    finally:
        stop.set()
        thread.join(timeout=5)

    assert seen, "the reader never completed a query"
    assert store.stats(project.id)["done"] == 40
