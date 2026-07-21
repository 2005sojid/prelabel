"""Serving CVAT and Label Studio directly.

Both are contract tests: the point is not that we produce *some* JSON, but that
we produce the exact shape the other tool parses. Getting a field name or a
coordinate system wrong means the boxes silently fail to appear.
"""

from __future__ import annotations

import base64

from tests.helpers import StubEngine, jpeg_bytes


def encoded_image() -> str:
    return base64.b64encode(jpeg_bytes(64, 48)).decode()


# --- CVAT -------------------------------------------------------------------


def test_cvat_info_describes_the_labels(client, loaded):
    body = client.get("/api/cvat/info").json()
    assert body["type"] == "detector"
    assert body["spec"] == [{"id": 0, "name": "car", "type": "rectangle"}]


def test_cvat_info_without_a_model_is_400(client):
    assert client.get("/api/cvat/info").status_code == 400


def test_cvat_invoke_returns_a_flat_list_of_shapes(client, loaded):
    """CVAT expects a bare array, in pixel coordinates."""
    response = client.post("/api/cvat/invoke", json={"image": encoded_image(), "threshold": 0.3})
    assert response.status_code == 200

    shapes = response.json()
    assert isinstance(shapes, list)
    assert shapes[0]["type"] == "rectangle"
    assert shapes[0]["label"] == "car"
    assert shapes[0]["points"] == [1.0, 2.0, 3.0, 4.0]  # pixels, not fractions
    assert float(shapes[0]["confidence"]) == 0.9


def test_cvat_invoke_accepts_a_data_url(client, loaded):
    payload = "data:image/jpeg;base64," + encoded_image()
    assert client.post("/api/cvat/invoke", json={"image": payload}).status_code == 200


def test_cvat_invoke_rejects_a_broken_image(client, loaded):
    response = client.post("/api/cvat/invoke", json={"image": "%%%not base64%%%"})
    assert response.status_code == 400


def test_cvat_polygons_for_a_segmentation_model(client, registry):
    engine = StubEngine(task="segment")
    engine.predict = _with_mask(engine)
    registry._engine = engine  # noqa: SLF001

    shapes = client.post("/api/cvat/invoke", json={"image": encoded_image()}).json()
    assert shapes[0]["type"] == "polygon"
    assert shapes[0]["points"] == [10.0, 10.0, 20.0, 10.0, 20.0, 20.0]


def _with_mask(engine):
    original = engine.predict

    def predict(image, conf=0.25, classes=None):
        result = original(image, conf=conf, classes=classes)
        result.detections[0].mask = [[10, 10], [20, 10], [20, 20]]
        result.detections[0].kind = "segmentation"
        return result

    return predict


# --- Label Studio -----------------------------------------------------------


def test_label_studio_health_reports_the_model(client, loaded):
    body = client.get("/api/label-studio/health").json()
    assert body["status"] == "UP"
    assert body["model_class"] == "stub"


def test_label_studio_health_without_a_model(client):
    assert client.get("/api/label-studio/health").json()["status"] == "NO_MODEL"


def test_label_studio_setup_returns_a_model_version(client, loaded):
    body = client.post("/api/label-studio/setup", json={}).json()
    assert body["model_version"] == "stub:detect"


def test_label_studio_predict_uses_percentage_coordinates(client, loaded):
    """The one thing that is not a re-serialisation: LS geometry is in percent."""
    tasks = [{"data": {"image": encoded_image()}}]
    body = client.post("/api/label-studio/predict", json={"tasks": tasks, "threshold": 0.2}).json()

    assert len(body["results"]) == 1
    entry = body["results"][0]["result"][0]
    assert entry["type"] == "rectanglelabels"
    assert entry["from_name"] == "RectangleLabels"
    assert entry["original_width"] == 64
    assert entry["original_height"] == 48

    # The stub's box is [1,2,3,4] on a 64x48 image.
    value = entry["value"]
    assert value["x"] == round(100 * 1 / 64, 4)
    assert value["y"] == round(100 * 2 / 48, 4)
    assert value["width"] == round(100 * 2 / 64, 4)
    assert value["height"] == round(100 * 2 / 48, 4)
    assert value["rectanglelabels"] == ["car"]


def test_label_studio_predict_scores_each_task(client, loaded):
    tasks = [{"data": {"image": encoded_image()}}]
    body = client.post("/api/label-studio/predict", json={"tasks": tasks}).json()
    assert body["results"][0]["score"] == 0.9


def test_label_studio_skips_a_task_it_cannot_decode(client, loaded):
    """One bad task must not fail the batch — the rest still get predictions."""
    tasks = [{"data": {"image": "not-base64"}}, {"data": {"image": encoded_image()}}]
    body = client.post("/api/label-studio/predict", json={"tasks": tasks}).json()

    assert len(body["results"]) == 2
    assert body["results"][0]["result"] == []
    assert len(body["results"][1]["result"]) == 1


def test_label_studio_task_without_an_image_is_skipped(client, loaded):
    body = client.post("/api/label-studio/predict", json={"tasks": [{"data": {}}]}).json()
    assert body["results"][0]["result"] == []


def test_label_studio_predict_without_a_model_is_400(client):
    assert client.post("/api/label-studio/predict", json={"tasks": []}).status_code == 400


def test_label_studio_uses_the_control_matching_the_task(client, registry):
    registry._engine = StubEngine(task="classify")  # noqa: SLF001
    body = client.post(
        "/api/label-studio/predict", json={"tasks": [{"data": {"image": encoded_image()}}]}
    ).json()
    entry = body["results"][0]["result"][0]
    assert entry["type"] == "choices"
    assert entry["value"]["choices"] == ["car"]
