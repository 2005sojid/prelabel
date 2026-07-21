"""The one loaded model, and the rules for replacing it safely.

Prelabel holds a single model in memory (it is a local, single-user tool, not
a multi-tenant service). :class:`ModelRegistry` owns that model together with the
lock guarding it, so no other module touches a mutable global.

Two properties matter here, and both are things the naive version gets wrong:

**A failed load never costs you the working model.** The replacement engine is
built *first*; only once it is running do we swap it in and tear the old one
down. An incomplete upload, a corrupt file or an unsupported format leaves the
previously loaded model exactly as it was.

**Checking and using the engine is one atomic step.** ``registry.engine()`` takes
the lock, verifies a model is present, and yields it — so a concurrent
``/api/model`` cannot swap the engine out between the check and the call. This
matters because the inference routes are plain ``def`` handlers, which FastAPI
runs in a thread pool, so requests really do overlap.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import config
from .engines import build_engine
from .engines.base import BaseEngine
from .errors import ModelChanged, ModelLoadError, NoModelLoaded

log = logging.getLogger("prelabel.state")

#: Prefix for the per-load directory holding a model's files. Each load gets its
#: own slot so the previous model's files stay untouched (and, on Windows,
#: unlocked) until its engine has actually been released.
SLOT_PREFIX = "active-"


def normalize_device(device: str | None) -> str | None:
    """Map a UI choice ('cpu' / 'gpu' / 'cuda' / 'cuda:1') to a backend string."""
    if not device:
        return None
    value = device.strip().lower()
    if value in ("gpu", "cuda"):
        return "cuda"
    if value == "cpu":
        return "cpu"
    return value


class ModelRegistry:
    """Holds the active engine and serialises access to it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: BaseEngine | None = None
        self._slot: Path | None = None
        self._target: Path | None = None
        self._imgsz: int | None = None
        self._device_pref: str | None = None
        #: Bumped on every change of the active model — see :attr:`generation`.
        self._generation = 0

    # -- inspection ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._engine is not None

    @property
    def device_preference(self) -> str | None:
        with self._lock:
            return self._device_pref

    @property
    def supports_tracking(self) -> bool:
        """Whether the loaded model can follow objects across video frames."""
        with self._lock:
            return bool(self._engine is not None and getattr(self._engine, "supports_tracking", False))

    def info(self) -> dict[str, Any] | None:
        """Model summary, or ``None`` when nothing is loaded."""
        with self._lock:
            return self._engine.info().to_dict() if self._engine else None

    def require_info(self) -> dict[str, Any]:
        with self._lock:
            if self._engine is None:
                raise NoModelLoaded()
            return self._engine.info().to_dict()

    def trainable_checkpoint(self) -> Path | None:
        """The active model's weights file, if it is one we can fine-tune from.

        Only a PyTorch Ultralytics checkpoint can be trained: an exported ONNX or
        OpenVINO graph has no optimiser state or trainable parameters in the form
        ``model.train`` needs. Returns the ``.pt`` path when the loaded model
        qualifies, and ``None`` otherwise — so the caller can refuse with a clear
        reason instead of letting Ultralytics fail deep inside a training run.
        """
        with self._lock:
            if self._engine is None or self._target is None:
                return None
            info = self._engine.info()
            if info.format != "PyTorch" or info.backend != "ultralytics":
                return None
            return self._target

    # -- access --------------------------------------------------------------

    @property
    def generation(self) -> int:
        """Counter bumped whenever the active model changes.

        Lets a long-running job (rendering a video frame by frame) notice that
        the model was swapped underneath it, instead of silently annotating the
        rest of the clip with a different model.
        """
        with self._lock:
            return self._generation

    @contextmanager
    def engine(self, expect_generation: int | None = None) -> Iterator[BaseEngine]:
        """Yield the loaded engine while holding the lock.

        Raises :class:`~app.errors.NoModelLoaded` if there is none, so callers
        never have to check first — which is what makes check-then-use atomic.

        Pass ``expect_generation`` when a job spans many calls and must be sure
        it is still talking to the same model throughout.
        """
        with self._lock:
            if self._engine is None:
                raise NoModelLoaded()
            if expect_generation is not None and expect_generation != self._generation:
                raise ModelChanged()
            yield self._engine

    # -- slots ---------------------------------------------------------------

    def new_slot(self) -> Path:
        """Create an empty directory to hold one model's files."""
        slot = config.MODELS_DIR / f"{SLOT_PREFIX}{uuid.uuid4().hex[:8]}"
        slot.mkdir(parents=True, exist_ok=True)
        return slot

    @staticmethod
    def discard_slot(slot: Path | None) -> None:
        if slot is not None and slot.exists():
            shutil.rmtree(slot, ignore_errors=True)

    # -- loading -------------------------------------------------------------

    def load(
        self,
        target: Path,
        slot: Path,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Build an engine for ``target`` and make it the active model.

        ``slot`` is the directory owning ``target``'s files; it is adopted on
        success and removed on failure. The currently loaded model is only
        released once the replacement is running.
        """
        requested_device = normalize_device(device) if device else self._device_pref
        engine = self._build(str(target), imgsz, requested_device)

        with self._lock:
            previous_engine, previous_slot = self._engine, self._slot
            self._engine = engine
            self._slot = slot
            self._target = target
            self._imgsz = imgsz
            self._device_pref = requested_device
            self._generation += 1

        # Outside the lock: releasing the old engine can be slow (CUDA teardown)
        # and nothing else needs to wait for it.
        self._retire(previous_engine, previous_slot, keep=slot)
        info = engine.info().to_dict()
        log.info("Model ready: %s", info)
        return info

    def reload_on_device(self, device: str) -> dict[str, Any]:
        """Rebuild the current model on a different device.

        The files are still on disk, so this is a rebuild rather than a
        re-upload. The image size chosen at load time is preserved — switching
        device must not silently change how the model runs.
        """
        with self._lock:
            target, slot, imgsz = self._target, self._slot, self._imgsz
        if target is None or slot is None:
            raise NoModelLoaded("No model loaded to switch device for")

        requested = normalize_device(device)
        engine = self._build(str(target), imgsz, requested)

        with self._lock:
            previous_engine = self._engine
            self._engine = engine
            self._device_pref = requested
            self._generation += 1

        # Same slot — release the engine but keep the files.
        self._retire(previous_engine, None, keep=slot)
        info = engine.info().to_dict()
        log.info("Device switched to %s; model reloaded", requested or "auto")
        return info

    def unload(self) -> None:
        """Release the model and free its resources."""
        with self._lock:
            engine, slot = self._engine, self._slot
            self._engine = self._slot = self._target = None
            self._imgsz = self._device_pref = None
            self._generation += 1
        self._retire(engine, slot, keep=None)

    # -- internals -----------------------------------------------------------

    def _build(self, target: str, imgsz: int | None, device: str | None) -> BaseEngine:
        """Construct an engine, leaving any currently loaded one untouched.

        Building before releasing means both models are briefly resident, which
        costs peak memory — on a GPU with little headroom, enough to matter.

        We accept that cost rather than freeing the incumbent first, because the
        alternative gives up the one guarantee this class exists to provide: a
        load that fails must never leave the user with nothing. Releasing first
        and retrying would recover the memory, but a retry that *also* fails
        loses the working model — the exact failure being defended against.

        When a swap genuinely does not fit in memory, the load fails with the
        driver's own message and the previous model keeps working. Freeing it is
        then a deliberate act: ``DELETE /api/model``, or "Unload model" in the UI.
        """
        try:
            engine = build_engine(target, imgsz=imgsz, device=device)
        except Exception as error:  # noqa: BLE001 - re-raised as ModelLoadError
            log.exception("Model load failed")
            raise ModelLoadError(f"Could not load model: {error}{self._memory_hint()}") from error

        # Constructing an engine is not proof that it works. Some backends accept
        # a garbage file and only fall over when asked something real, so we ask
        # now — while failing is still free — rather than after the swap, when it
        # would leave the registry holding a broken engine.
        try:
            engine.info()
        except Exception as error:  # noqa: BLE001
            log.exception("Model loaded but could not describe itself")
            engine.close()
            raise ModelLoadError(f"Could not load model: the file is not a usable model ({error})") from error

        return engine

    def _memory_hint(self) -> str:
        if not self.is_loaded:
            return ""
        return " (the previously loaded model is still active — unload it first if memory is tight)"

    @staticmethod
    def _retire(engine: BaseEngine | None, slot: Path | None, keep: Path | None) -> None:
        """Close a replaced engine and delete its files, in that order.

        The ordering is load-bearing on Windows, where a loaded OpenVINO ``.bin``
        stays locked until its handles are freed.
        """
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the new load
                log.exception("Error while releasing the previous engine")
        if slot is not None and (keep is None or slot.resolve() != keep.resolve()):
            ModelRegistry.discard_slot(slot)


def clear_stale_slots() -> None:
    """Delete model slots left behind by a previous run.

    Nothing is loaded at startup, so every ``active-*`` directory is by
    definition orphaned — as is anything left in ``pending/``.
    """
    if not config.MODELS_DIR.exists():
        return
    for entry in config.MODELS_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith(SLOT_PREFIX):
            shutil.rmtree(entry, ignore_errors=True)
    shutil.rmtree(config.PENDING_DIR, ignore_errors=True)
