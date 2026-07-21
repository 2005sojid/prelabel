"""Video reading and writing.

Two details here are worth more attention than they usually get, because getting
either wrong fails *silently* — the user gets a result that looks fine and isn't.

**Sampling, not truncating.** A frame cap has to be spread across the whole clip.
Reading frames until the cap is hit instead returns the first N frames, so a
60-second video quietly comes back as 30 seconds of footage with no error
anywhere. :class:`VideoReader` picks evenly spaced frames and uses the full
budget.

**A codec browsers can actually play.** OpenCV's usual ``mp4v`` fourcc writes
MPEG-4 Part 2, which Chrome and Firefox refuse to decode in a ``<video>``
element — the download succeeds and the player stays blank. We probe for a real
H.264 encoder once and report honestly when only a fallback is available.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..errors import UnsupportedMedia

log = logging.getLogger("prelabel.media.video")

#: Fourcc codes worth asking for, best first. What OpenCV actually produces for
#: each is decided by the FFmpeg build underneath, which is why we check the
#: result rather than trusting the request.
CANDIDATE_FOURCC: tuple[str, ...] = ("avc1", "h264", "H264", "x264", "mp4v")

#: Last resort. Always available, but MPEG-4 Part 2 is not playable in Chrome or
#: Firefox, so we say so rather than pretend otherwise.
FALLBACK_FOURCC = "mp4v"

#: Sample-entry tags that mean "a browser can play this". Found by inspecting the
#: written container: the only evidence that is not a guess.
BROWSER_SAFE_TAGS: tuple[bytes, ...] = (b"avc1", b"avcC", b"hvc1", b"hev1")

#: A working encoder writes a real container; a failed one leaves a stub.
_MIN_PROBE_BYTES = 256

_DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class Codec:
    """The video codec this machine can actually encode with."""

    fourcc: str
    browser_playable: bool

    @property
    def note(self) -> str:
        """Plain-language explanation, safe to put in an HTTP header.

        Deliberately ASCII: this string is returned as ``X-Prelabel-Codec-Note``,
        and HTTP header values are latin-1. A single typographic dash in here
        made the whole video endpoint fail with a 500 on any machine that lacked
        H.264 — which is most Linux installs of ``opencv-python``.
        """
        if self.browser_playable:
            return f"H.264 ({self.fourcc})"
        return (
            f"{self.fourcc} (MPEG-4 Part 2) plays in Safari and desktop players, "
            "but not in Chrome or Firefox. Install an OpenCV build with H.264 "
            "support (e.g. ffmpeg with libopenh264) for in-browser playback."
        )


@contextmanager
def _quiet_opencv() -> Iterator[None]:
    """Silence OpenCV's C++ logger for the duration of a block.

    Probing deliberately asks for codecs that may be missing, and each miss
    prints a red ``ERROR`` line from FFmpeg. Those look like a broken install
    when they are in fact the probe doing its job.
    """
    try:
        from cv2.utils import logging as cv_logging

        previous = cv_logging.getLogLevel()
        cv_logging.setLogLevel(cv_logging.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001 - older OpenCV without the logging helper
        yield
        return
    try:
        yield
    finally:
        cv_logging.setLogLevel(previous)


def _probe_fourcc(fourcc: str) -> bytes | None:
    """Encode a few frames and report what actually landed in the container.

    Returns the codec tag found in the output (``b"avc1"``, ``b"mp4v"``, …), or
    ``None`` if nothing usable was written.

    Checking the *output* matters because neither half of the request is
    trustworthy: ``VideoWriter.isOpened()`` can return True and then write an
    empty file when no encoder is present, and FFmpeg silently substitutes a
    different codec for an unsupported tag. The bytes on disk are the only
    evidence that does not lie.
    """
    with tempfile.TemporaryDirectory(prefix="pl-codec-") as tmp:
        probe = Path(tmp) / "probe.mp4"
        writer = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*fourcc), 25.0, (64, 64))
        try:
            if not writer.isOpened():
                return None
            # Noise, not black: an all-zero clip can compress to almost nothing
            # and blur the "did anything get written?" signal.
            rng = np.random.default_rng(0)
            for _ in range(5):
                writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        finally:
            writer.release()

        if not probe.exists() or probe.stat().st_size < _MIN_PROBE_BYTES:
            return None
        return _container_tag(probe.read_bytes())


def _container_tag(data: bytes) -> bytes | None:
    """The video sample-entry tag present in an MP4's bytes, if recognised."""
    for tag in (*BROWSER_SAFE_TAGS, b"mp4v"):
        if tag in data:
            return tag
    return None


@lru_cache(maxsize=1)
def h264_transcoder() -> str | None:
    """Path to an ``ffmpeg`` binary that can encode H.264, if one is installed.

    OpenCV's pip wheels bundle their own FFmpeg built *without* an H.264 encoder
    — licensing, not oversight — so on Linux (and inside the Docker image)
    ``cv2.VideoWriter`` can only produce MPEG-4 Part 2, which no browser plays.
    The system ``ffmpeg`` package usually does have libx264, and re-encoding a
    finished clip with it is cheap next to running a model over every frame.
    """
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    try:
        encoders = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Could not query ffmpeg encoders: %s", exc)
        return None

    if "libx264" in encoders:
        return binary
    log.debug("ffmpeg found at %s but it has no libx264", binary)
    return None


def transcode_to_h264(source: Path, timeout: int = 600) -> bool:
    """Re-encode ``source`` in place as H.264. True if it worked.

    Failure is never fatal: the original file is left exactly as it was, and the
    caller keeps reporting the codec it actually has.
    """
    binary = h264_transcoder()
    if binary is None:
        return False

    destination = source.with_name(f"{source.stem}_h264{source.suffix}")
    command = [
        binary, "-y", "-loglevel", "error",
        "-i", str(source),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        # Browsers need 4:2:0; libx264 would otherwise keep the source's format
        # and some players show nothing at all.
        "-pix_fmt", "yuv420p",
        # Put the index at the front so playback can start before the whole file
        # has arrived.
        "-movflags", "+faststart",
        str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("H.264 transcode failed to run: %s", exc)
        destination.unlink(missing_ok=True)
        return False

    if result.returncode != 0 or not destination.exists() or destination.stat().st_size < _MIN_PROBE_BYTES:
        log.warning("H.264 transcode failed: %s", (result.stderr or "")[:300])
        destination.unlink(missing_ok=True)
        return False

    destination.replace(source)
    return True


@lru_cache(maxsize=1)
def preferred_codec() -> Codec:
    """Best available MP4 codec, probed once and cached for the process."""
    fallback: Codec | None = None

    with _quiet_opencv():
        for fourcc in CANDIDATE_FOURCC:
            tag = _probe_fourcc(fourcc)
            if tag is None:
                continue
            if tag in BROWSER_SAFE_TAGS:
                log.info("Video codec: %s → %s (browser-playable)", fourcc, tag.decode())
                return Codec(fourcc, browser_playable=True)
            if fallback is None:
                fallback = Codec(fourcc, browser_playable=False)

    if fallback is not None:
        log.warning(
            "No H.264 encoder available to OpenCV — using %s, which Chrome and Firefox "
            "cannot play back in a <video> element.",
            fallback.fourcc,
        )
        return fallback

    log.error("No usable MP4 encoder found; falling back to %s and hoping for the best", FALLBACK_FOURCC)
    return Codec(FALLBACK_FOURCC, browser_playable=False)


@lru_cache(maxsize=1)
def effective_codec() -> Codec:
    """What a rendered clip will *actually* be, transcoding included.

    :func:`preferred_codec` answers "what can OpenCV write", which is not the
    same question once an ffmpeg binary can re-encode the result. Reporting the
    probe alone would tell the UI a clip is unplayable when it is about to be
    made playable.
    """
    codec = preferred_codec()
    if codec.browser_playable or h264_transcoder() is None:
        return codec
    return Codec("avc1", browser_playable=True)


def sampling_plan(total_frames: int, max_frames: int) -> int:
    """How many frames to emit from a clip of ``total_frames``.

    Returns ``total_frames`` when no cap applies, otherwise the cap. The point of
    a separate function is that the *count* and the *positions* stay consistent:
    :meth:`VideoReader.frames` spreads exactly this many frames across the whole
    clip.
    """
    if total_frames <= 0:
        return max(0, max_frames)
    if max_frames <= 0:
        return total_frames
    return min(total_frames, max_frames)


class VideoReader:
    """Reads a video, sampling evenly when a frame cap applies.

    Use as a context manager::

        with VideoReader(path, max_frames=900) as reader:
            for frame in reader.frames():
                ...
    """

    def __init__(self, path: str, max_frames: int = 0) -> None:
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            self._capture.release()
            raise UnsupportedMedia(f"Could not open video: {Path(path).name}")

        self.path = path
        self.max_frames = max(0, int(max_frames))
        self.source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0) or _DEFAULT_FPS
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        #: How many frames we intend to emit. Zero total means the container did
        #: not report a frame count (some streams and codecs do not).
        self.planned_frames = sampling_plan(self.total_frames, self.max_frames)
        self.frames_emitted = 0
        #: True when the source length was unknown *and* the cap cut us short, so
        #: the output really is only the beginning of the clip.
        self.truncated = False

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info) -> None:  # noqa: ANN002
        self.close()

    def close(self) -> None:
        self._capture.release()

    # -- properties ----------------------------------------------------------

    @property
    def length_known(self) -> bool:
        return self.total_frames > 0

    @property
    def is_sampled(self) -> bool:
        """True when we are dropping frames to honour the cap."""
        return self.length_known and self.planned_frames < self.total_frames

    @property
    def output_fps(self) -> float:
        """Playback rate that preserves the original duration.

        Emitting ``planned`` of ``total`` frames over the same wall-clock span
        means the output must run proportionally slower, otherwise a sampled clip
        plays back sped up.
        """
        if not self.length_known or self.planned_frames <= 0:
            return self.source_fps
        return self.source_fps * self.planned_frames / self.total_frames

    # -- iteration -----------------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames, evenly spaced across the clip.

        When the source length is known, frame *i* of the output is taken from
        position ``i * total / planned`` — so the emitted frames span the entire
        video and there are exactly ``planned`` of them. When the length is
        unknown we cannot space anything, so we read sequentially and set
        :attr:`truncated` if the cap stops us early.
        """
        planned, total = self.planned_frames, self.total_frames
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break

            if self.length_known and planned > 0:
                if index >= (self.frames_emitted * total) // planned:
                    yield frame
                    self.frames_emitted += 1
                    if self.frames_emitted >= planned:
                        break
            else:
                if self.max_frames and self.frames_emitted >= self.max_frames:
                    self.truncated = True
                    break
                yield frame
                self.frames_emitted += 1
            index += 1


@dataclass(frozen=True)
class WriteReport:
    """What actually came out of :func:`write_video`."""

    path: Path
    codec: Codec
    frames_written: int
    fps: float
    size: tuple[int, int]

    @property
    def is_empty(self) -> bool:
        return self.frames_written == 0


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
    size: tuple[int, int] | None = None,
) -> WriteReport:
    """Encode ``frames`` to an MP4 at ``path``.

    ``size`` is ``(width, height)``; when omitted it is taken from the first
    frame. Frames whose size differs from the writer's are resized rather than
    silently dropped by the encoder.
    """
    destination = Path(path)
    codec = preferred_codec()
    iterator = iter(frames)

    try:
        first = next(iterator)
    except StopIteration:
        return WriteReport(destination, codec, 0, fps, size or (0, 0))

    width, height = size or (first.shape[1], first.shape[0])
    fps = float(fps) if fps and fps > 0 else _DEFAULT_FPS

    # Quiet: FFmpeg logs a red ERROR for each encoder it tries and rejects before
    # settling on one that works. `isOpened()` below is the check that matters.
    with _quiet_opencv():
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec.fourcc), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise UnsupportedMedia(f"Could not open a video writer for {destination.name}")

    written = 0
    try:
        for frame in _sized(first, iterator, width, height):
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    # OpenCV could only manage MPEG-4 Part 2, but a system ffmpeg may still be
    # able to turn it into something a browser will play. Cheap next to the
    # inference that produced these frames.
    if written and not codec.browser_playable and transcode_to_h264(destination):
        log.info("Re-encoded %s to H.264 with ffmpeg", destination.name)
        codec = Codec("avc1", browser_playable=True)

    return WriteReport(destination, codec, written, fps, (width, height))


def _sized(
    first: np.ndarray,
    rest: Iterator[np.ndarray],
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Yield every frame at exactly ``width`` x ``height``."""
    for frame in (first, *rest):
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        yield frame
