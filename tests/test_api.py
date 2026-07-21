"""The HTTP contract.

Locks down what the frontend depends on: response shapes, status codes, and the
rule that an error is *always* JSON — a plain-text 500 makes the client's
``response.json()`` throw a misleading "... is not valid JSON" and hides the
real failure.
"""

from __future__ import annotations

import pytest

from prelabel import config
from tests.helpers import StubEngine, jpeg_bytes, upload


def assert_json_error(response, status: int):
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["status"] == "error"
    assert body["detail"]
    return body


# --- system -----------------------------------------------------------------


def test_health_without_a_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model"] is None
    assert body["version"]


def test_health_reports_the_loaded_model(client, loaded):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert body["model"]["task"] == "detect"


def test_formats_describes_the_server(client):
    body = client.get("/api/formats").json()
    assert ".pt" in body["model_formats"]
    assert ".xml" in body["model_formats"]
    assert any(device["id"] == "cpu" for device in body["devices"])  # CPU is always present
    assert isinstance(body["video_codec"]["browser_playable"], bool)
    assert body["limits"]["max_batch_files"] > 0


def test_index_serves_the_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# --- model ------------------------------------------------------------------


def test_model_info_is_404_when_absent(client):
    assert_json_error(client.get("/api/model"), 404)


def test_model_info_when_loaded(client, loaded):
    body = client.get("/api/model").json()
    assert body["name"] == "stub"
    assert body["num_classes"] == 1


def test_unload_releases_the_model(client, loaded):
    assert client.delete("/api/model").json()["model_loaded"] is False
    assert loaded.closed is True
    assert client.get("/api/health").json()["model_loaded"] is False


def test_upload_rejects_an_unsupported_extension(client):
    body = assert_json_error(
        client.post("/api/model", files={"files": ("notes.txt.exe", b"x", "application/octet-stream")}),
        400,
    )
    assert ".exe" in body["detail"]


def test_upload_rejects_too_many_files(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_FILES", 2)
    files = [("files", (f"m{i}.onnx", b"x", "application/octet-stream")) for i in range(3)]
    assert_json_error(client.post("/api/model", files=files), 413)


def test_incomplete_openvino_upload_waits_without_disturbing_the_model(client, loaded):
    """A lone .xml must not cost the user their working model."""
    response = client.post("/api/model", files={"files": ("net.xml", b"<net/>", "text/xml")})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting"
    assert ".bin" in body["message"]
    # The critical part: the server still has the model it had before.
    assert body["model_loaded"] is True
    assert body["model"]["name"] == "stub"
    assert loaded.closed is False
    assert client.get("/api/health").json()["model_loaded"] is True


def test_oversized_model_upload_keeps_the_previous_model(client, loaded, monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_MB", 1)
    oversized = b"\0" * (2 * 1024 * 1024)

    assert_json_error(client.post("/api/model", files={"files": ("big.pt", oversized, "application/octet-stream")}), 413)

    assert client.get("/api/health").json()["model_loaded"] is True
    assert loaded.closed is False


def test_unloadable_model_keeps_the_previous_model(client, loaded):
    """A file with a valid extension but garbage contents must fail cleanly."""
    assert_json_error(client.post("/api/model", files={"files": ("broken.onnx", b"not a model", "application/octet-stream")}), 400)

    assert client.get("/api/health").json()["model_loaded"] is True
    assert loaded.closed is False


def test_device_switch_without_a_model_is_rejected(client):
    assert_json_error(client.post("/api/device", data={"device": "cuda"}), 400)


# --- benchmark --------------------------------------------------------------


def test_benchmark_reports_latency_and_throughput(client, loaded):
    body = client.post("/api/benchmark", data={"runs": "10"}).json()
    assert body["latency_ms"] == 20.0
    assert body["throughput_fps"] > body["latency_fps"]  # throughput beats latency


def test_benchmark_clamps_the_run_count(client, loaded):
    body = client.post("/api/benchmark", data={"runs": "100000"}).json()
    assert body["runs"] == config.BENCHMARK_MAX_RUNS


def test_benchmark_without_a_model_is_400(client):
    assert_json_error(client.post("/api/benchmark", data={"runs": "10"}), 400)


# --- predict ----------------------------------------------------------------


def test_predict_without_a_model_is_json_400(client):
    assert_json_error(client.post("/api/predict", files={"file": upload()}), 400)


def test_predict_returns_timings_and_detections(client, loaded):
    body = client.post("/api/predict", files={"file": upload()}).json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    assert body["timings"]["inference_ms"] == 5
    assert body["detections"][0]["class_name"] == "car"


def test_predict_calls_the_model_exactly_once(client, loaded):
    """No pre-warm massaging: the reported time is one real inference."""
    client.post("/api/predict", files={"file": upload()})
    client.post("/api/predict", files={"file": upload()})
    assert loaded.calls == 2


def test_inference_failure_is_json_422(client, registry):
    engine = StubEngine(fail=True)
    registry._engine = engine  # noqa: SLF001
    body = assert_json_error(client.post("/api/predict", files={"file": upload()}), 422)
    assert "mismatch" in body["detail"]


def test_undecodable_image_is_rejected(client, loaded):
    assert_json_error(client.post("/api/predict", files={"file": upload("x.jpg", b"not an image")}), 400)


def test_oversized_image_is_rejected(client, loaded, monkeypatch):
    monkeypatch.setattr(config, "MAX_MEDIA_MB", 1)
    assert_json_error(client.post("/api/predict", files={"file": upload("big.jpg", b"\0" * (2 * 1024 * 1024))}), 413)


@pytest.mark.parametrize(("sent", "expected"), [("1.5", 1.0), ("-2", 0.0), ("0.3", 0.3)])
def test_confidence_is_clamped(client, loaded, sent, expected):
    client.post("/api/predict", files={"file": upload()}, data={"conf": sent})
    assert loaded.last_conf == pytest.approx(expected)


# --- batch ------------------------------------------------------------------


def test_batch_returns_one_aligned_result_per_file(client, loaded):
    files = [("files", upload("a.jpg")), ("files", upload("b.jpg", b"not an image")), ("files", upload("c.jpg"))]
    body = client.post("/api/predict/batch", files=files).json()

    results = body["results"]
    assert len(results) == 3
    assert results[0]["status"] == "ok" and results[0]["count"] == 1
    assert results[1]["status"] == "error"  # undecodable, reported in its own slot
    assert results[2]["status"] == "ok"


def test_batch_uses_a_single_batched_call(client, loaded):
    files = [("files", upload(f"{i}.jpg")) for i in range(5)]
    client.post("/api/predict/batch", files=files)
    assert loaded.batch_calls == 1


def test_batch_without_a_model_is_400(client):
    assert_json_error(client.post("/api/predict/batch", files=[("files", upload())]), 400)


def test_batch_rejects_too_many_files(client, loaded, monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH_FILES", 2)
    files = [("files", upload(f"{i}.jpg")) for i in range(3)]
    assert_json_error(client.post("/api/predict/batch", files=files), 413)


def test_batch_reports_an_oversized_file_inline(client, loaded, monkeypatch):
    """One huge file must not fail the whole chunk."""
    monkeypatch.setattr(config, "MAX_MEDIA_MB", 1)
    files = [
        ("files", upload("ok.jpg")),
        ("files", upload("big.jpg", b"\0" * (2 * 1024 * 1024))),
    ]
    results = client.post("/api/predict/batch", files=files).json()["results"]
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert "limit" in results[1]["detail"]


# --- video ------------------------------------------------------------------


def test_video_rejects_an_unsupported_extension(client, loaded):
    body = assert_json_error(
        client.post("/api/predict/video", files={"file": ("clip.gif", b"GIF89a", "image/gif")}),
        400,
    )
    assert ".gif" in body["detail"]


def test_video_without_a_model_is_400(client):
    assert_json_error(client.post("/api/predict/video", files={"file": ("c.mp4", b"\0", "video/mp4")}), 400)


def test_video_renders_and_reports_its_codec(client, loaded, tmp_path):
    from tests.helpers import write_test_video

    source = write_test_video(tmp_path / "in.mp4", frames=12)
    response = client.post(
        "/api/predict/video",
        files={"file": ("in.mp4", source.read_bytes(), "video/mp4")},
        data={"conf": "0.5"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["X-Prelabel-Codec"]
    assert response.headers["X-Prelabel-Browser-Playable"] in ("0", "1")
    assert int(response.headers["X-Prelabel-Frames"]) == 12
    assert response.headers["X-Prelabel-Sampled"] == "0"
    assert len(response.content) > 0


def test_video_reports_when_the_frame_cap_sampled_the_clip(client, loaded, tmp_path, monkeypatch):
    from tests.helpers import write_test_video

    monkeypatch.setattr(config, "MAX_VIDEO_FRAMES", 10)
    source = write_test_video(tmp_path / "long.mp4", frames=40)
    response = client.post("/api/predict/video", files={"file": ("long.mp4", source.read_bytes(), "video/mp4")})

    assert response.status_code == 200
    assert response.headers["X-Prelabel-Sampled"] == "1"
    assert int(response.headers["X-Prelabel-Frames"]) == 10


def test_video_with_no_readable_frames_is_422(client, loaded):
    assert_json_error(client.post("/api/predict/video", files={"file": ("c.mp4", b"nonsense", "video/mp4")}), 400)


def test_video_cleans_up_its_temporary_files(client, loaded, tmp_path):
    from tests.helpers import write_test_video

    source = write_test_video(tmp_path / "in.mp4", frames=6)
    response = client.post("/api/predict/video", files={"file": ("in.mp4", source.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    response.read()  # drain, so the background cleanup task runs

    leftovers = list(config.OUTPUTS_DIR.iterdir())
    assert leftovers == [], f"temporary files were left behind: {leftovers}"


# --- stream -----------------------------------------------------------------


def test_stream_reports_missing_model_and_stays_open(client):
    with client.websocket_connect("/api/stream") as socket:
        socket.send_json({"image": "data:image/jpeg;base64,", "conf": 0.5})
        assert socket.receive_json()["status"] == "error"


def test_stream_runs_a_frame(client, loaded):
    import base64

    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes()).decode()
    with client.websocket_connect("/api/stream") as socket:
        socket.send_json({"image": data_url, "conf": 0.4})
        payload = socket.receive_json()
    assert payload["status"] == "ok"
    assert payload["count"] == 1


def test_stream_survives_a_bad_frame(client, loaded):
    """One malformed frame must not tear down the stream."""
    import base64

    good = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes()).decode()
    with client.websocket_connect("/api/stream") as socket:
        socket.send_json({"image": "data:image/jpeg;base64,%%%not-base64%%%"})
        assert socket.receive_json()["status"] == "error"
        socket.send_json({"image": good})
        assert socket.receive_json()["status"] == "ok"
