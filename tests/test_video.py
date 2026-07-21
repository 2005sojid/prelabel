"""Video reading and writing.

These cover two failures that are invisible without a test, because both produce
output that *looks* correct:

* a frame cap that truncates the clip instead of sampling it, so a 60-second
  video silently comes back as 30 seconds
* a codec no browser can decode, so the download succeeds and the player is blank
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from prelabel.media.video import (
    BROWSER_SAFE_TAGS,
    VideoReader,
    preferred_codec,
    sampling_plan,
    write_video,
)
from tests.helpers import write_test_video

# --- sampling plan ----------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "cap", "expected"),
    [
        (100, 0, 100),      # no cap: everything
        (100, 900, 100),    # under the cap: everything
        (900, 900, 900),    # exactly at the cap
        (1000, 900, 900),   # over the cap: use the whole budget
        (1500, 900, 900),
        (1799, 900, 900),
        (5400, 900, 900),
        (0, 900, 900),      # unknown length: cap is all we can promise
    ],
)
def test_sampling_plan_uses_the_full_budget(total, cap, expected):
    assert sampling_plan(total, cap) == expected


# --- reader -----------------------------------------------------------------


def test_reader_returns_every_frame_when_under_the_cap(tmp_path):
    path = write_test_video(tmp_path / "short.mp4", frames=20)
    with VideoReader(str(path), max_frames=900) as reader:
        frames = list(reader.frames())
    assert len(frames) == 20
    assert reader.is_sampled is False


@pytest.mark.parametrize("total", [120, 200, 359])
def test_reader_spans_the_whole_clip_when_capped(tmp_path, total):
    """The regression: a cap must sample across the clip, not truncate it.

    ``total=359`` with a cap of 90 is the shape that used to break — a naive
    ``total // cap`` step of 3 is fine here, but ``total`` just under twice the
    cap produced a step of 1 and stopped at the halfway point.
    """
    cap = 90
    path = write_test_video(tmp_path / f"clip{total}.mp4", frames=total)

    with VideoReader(str(path), max_frames=cap) as reader:
        frames = list(reader.frames())
        assert reader.is_sampled is True

    assert len(frames) == cap

    # Frame i was filled with value i*2, so the last frame returned must come
    # from near the end of the source rather than from the middle.
    last_value = int(frames[-1][0, 0, 0])
    expected_last = min(255, (total - 1) * 2)
    assert last_value >= expected_last - 12, (
        f"last returned frame has value {last_value}, expected close to {expected_last} — "
        "the clip was truncated instead of sampled"
    )


def test_reader_preserves_duration_when_sampling(tmp_path):
    """A sampled clip must play slower, or it would appear sped up."""
    path = write_test_video(tmp_path / "long.mp4", frames=200, fps=30.0)
    with VideoReader(str(path), max_frames=50) as reader:
        source_duration = reader.total_frames / reader.source_fps
        output_duration = reader.planned_frames / reader.output_fps
    assert output_duration == pytest.approx(source_duration, rel=0.02)


def test_reader_rejects_a_file_that_is_not_a_video(tmp_path):
    from prelabel.errors import UnsupportedMedia

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    with pytest.raises(UnsupportedMedia):
        VideoReader(str(broken))


# --- codec ------------------------------------------------------------------


def test_preferred_codec_reports_what_it_can_actually_encode(tmp_path):
    """``browser_playable`` must reflect the bytes written, not the request."""
    codec = preferred_codec()

    path = tmp_path / "probe.mp4"
    frames = [np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(5)]
    report = write_video(path, frames, fps=25.0)

    assert report.frames_written == 5
    assert path.exists() and path.stat().st_size > 0

    data = path.read_bytes()
    contains_h264 = any(tag in data for tag in BROWSER_SAFE_TAGS)
    assert contains_h264 == codec.browser_playable, (
        "the codec report disagrees with what was written: "
        f"browser_playable={codec.browser_playable}, H.264 tag present={contains_h264}"
    )


def test_written_video_reads_back_with_every_frame(tmp_path):
    path = tmp_path / "roundtrip.mp4"
    frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (10, 60, 110, 160, 210)]
    write_video(path, frames, fps=25.0)

    capture = cv2.VideoCapture(str(path))
    try:
        read_back = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            read_back.append(frame)
    finally:
        capture.release()
    assert len(read_back) == len(frames)


def test_write_video_resizes_mismatched_frames(tmp_path):
    """A frame of the wrong size must be resized, not silently dropped."""
    path = tmp_path / "mixed.mp4"
    frames = [
        np.zeros((48, 64, 3), dtype=np.uint8),
        np.zeros((96, 128, 3), dtype=np.uint8),  # different size
        np.zeros((48, 64, 3), dtype=np.uint8),
    ]
    report = write_video(path, frames, fps=25.0)
    assert report.frames_written == 3
    assert report.size == (64, 48)


def test_write_video_with_no_frames_reports_empty(tmp_path):
    report = write_video(tmp_path / "empty.mp4", iter([]), fps=25.0)
    assert report.is_empty


# --- the fallback machine ---------------------------------------------------


@pytest.fixture
def without_h264(monkeypatch):
    """Pretend this machine's OpenCV has no H.264 encoder.

    Most Linux installs of ``opencv-python`` are exactly this, so it is the
    configuration CI runs in and a developer on Windows never sees.
    """
    import prelabel.media.video as video_module

    monkeypatch.setattr(video_module, "CANDIDATE_FOURCC", ("mp4v",))
    video_module.preferred_codec.cache_clear()
    yield
    video_module.preferred_codec.cache_clear()


def test_codec_note_is_header_safe(without_h264):
    """The note travels in an HTTP header, and headers are latin-1.

    A typographic dash in this string made every video request fail with a 500
    on machines without H.264 — a whole endpoint lost to one character.
    """
    note = preferred_codec().note
    assert not preferred_codec().browser_playable
    note.encode("latin-1")  # raises if a stray non-ASCII character creeps back in


def test_every_codec_note_is_header_safe():
    from prelabel.media.video import Codec

    for codec in (Codec("avc1", True), Codec("mp4v", False), Codec("hvc1", True)):
        codec.note.encode("latin-1")


def test_video_endpoint_works_without_h264(client, loaded, tmp_path, without_h264):
    """End to end on the fallback: a 500 here is the bug this pins down."""
    from tests.helpers import write_test_video

    source = write_test_video(tmp_path / "fallback.mp4", frames=8)
    response = client.post(
        "/api/predict/video",
        files={"file": ("fallback.mp4", source.read_bytes(), "video/mp4")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Prelabel-Browser-Playable"] == "0"
    assert response.headers["X-Prelabel-Codec-Note"]
    assert len(response.content) > 0
