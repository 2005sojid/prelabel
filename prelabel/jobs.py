"""Running a project in the background.

A folder of ten thousand images takes minutes to hours. That cannot happen inside
a request, so a project run is a job: it starts, reports progress, can be
cancelled, and survives the browser tab being closed — the results are in the
database either way.

**One run at a time.** There is a single model behind a single lock, so two
concurrent runs would not be faster; they would just take turns and make both
progress bars misleading. A second start is refused with an explanation rather
than silently queued.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from . import comparison, config, datasets, training
from .engines.tiling import predict_tiled
from .errors import Conflict, NotFound
from .state import ModelRegistry
from .store import Item, Store
from .training import TrainingSettings

log = logging.getLogger("prelabel.jobs")


@dataclass
class RunSettings:
    """Per-project inference options, stored so a resumed run behaves the same."""

    conf: float = config.PROJECT_CONF_FLOOR
    #: Class ids to keep; empty means every class.
    classes: list[int] = field(default_factory=list)
    #: Slice large images instead of downscaling them (see engines/tiling.py).
    tiled: bool = False
    tile_size: int = 0            # 0 = the model's own input size
    tile_overlap: float = 0.2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RunSettings:
        data = data or {}
        return cls(
            conf=float(data.get("conf", config.PROJECT_CONF_FLOOR)),
            classes=[int(c) for c in data.get("classes", []) or []],
            tiled=bool(data.get("tiled", False)),
            tile_size=int(data.get("tile_size", 0) or 0),
            tile_overlap=float(data.get("tile_overlap", 0.2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conf": self.conf,
            "classes": self.classes,
            "tiled": self.tiled,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
        }

    @property
    def class_filter(self) -> list[int] | None:
        return self.classes or None


class ProjectRunner:
    """Owns the worker thread that processes projects."""

    def __init__(self, store: Store, registry: ModelRegistry) -> None:
        self._store = store
        self._registry = registry
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active: str | None = None
        self._cancel = threading.Event()

    # -- control -------------------------------------------------------------

    @property
    def active_project(self) -> str | None:
        with self._lock:
            return self._active

    def is_running(self, project_id: str) -> bool:
        return self.active_project == project_id

    def start(self, project_id: str, settings: RunSettings) -> None:
        """Begin (or resume) a run. Raises if another project is already going."""
        project = self._store.get_project(project_id)
        if project is None:
            raise NotFound(f"No project '{project_id}'")
        if not self._registry.is_loaded:
            raise Conflict("Load a model before running a project")

        with self._lock:
            if self._active is not None:
                raise Conflict(
                    f"Project '{self._active}' is still running. "
                    "Wait for it or cancel it before starting another."
                )
            self._active = project_id
            self._cancel.clear()
            # Flip the status here, not in the worker: a client that polls
            # immediately after this call would otherwise read the *previous*
            # run's "done" and conclude the new one had already finished.
            self._store.update_project(project_id, status="running", detail="")
            self._thread = threading.Thread(
                target=self._run, args=(project_id, settings), name=f"project-{project_id}", daemon=True
            )
            self._thread.start()

    def cancel(self, project_id: str) -> bool:
        """Ask a running project to stop. Returns False if it was not running."""
        if not self.is_running(project_id):
            return False
        self._cancel.set()
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop any run and wait briefly for the worker to notice."""
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- the run -------------------------------------------------------------

    def _run(self, project_id: str, settings: RunSettings) -> None:
        store = self._store
        try:
            project = store.get_project(project_id)
            if project is None:
                return

            root = datasets.resolve_dataset_dir(project.source_dir)
            store.update_project(
                project_id,
                settings=settings.to_dict(),
                model=self._registry.info() or {},
            )
            generation = self._registry.generation

            processed = self._process_all(project_id, root, settings, generation)

            if self._cancel.is_set():
                store.update_project(project_id, status="cancelled",
                                     detail=f"Stopped after {processed} images")
                log.info("Project %s cancelled after %d images", project_id, processed)
            else:
                store.update_project(project_id, status="done", detail="")
                log.info("Project %s finished: %d images", project_id, processed)

            # New results invalidate any stored comparison. Recompute it here so
            # the diff is ready when the user looks, rather than stale or absent.
            if store.baseline_size(project_id):
                comparison.refresh(store, project_id)

        except Exception as exc:  # noqa: BLE001 - recorded on the project, not lost
            log.exception("Project %s failed", project_id)
            store.update_project(project_id, status="failed", detail=str(exc)[:500])
        finally:
            with self._lock:
                self._active = None
                self._thread = None

    def _process_all(self, project_id: str, root: Path, settings: RunSettings, generation: int) -> int:
        processed = 0
        batch_size = max(1, config.PROJECT_BATCH_SIZE)

        while not self._cancel.is_set():
            pending = self._store.pending_items(project_id, batch_size)
            if not pending:
                break
            processed += self._process_batch(project_id, root, pending, settings, generation)
        return processed

    def _process_batch(
        self,
        project_id: str,
        root: Path,
        items: list[Item],
        settings: RunSettings,
        generation: int,
    ) -> int:
        """Read, infer and store one batch. Per-image failures stay per-image."""
        loaded: list[tuple[Item, Any]] = []
        for item in items:
            try:
                path = datasets.resolve_item_path(root, item.rel_path)
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("not a readable image")
                loaded.append((item, image))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                self._store.save_error(project_id, item.id, str(exc))

        if not loaded:
            return len(items)

        try:
            results = self._infer(loaded, settings, generation)
        except Exception as exc:  # noqa: BLE001
            # A whole-batch failure (model unloaded, device lost) is not the
            # images' fault, so mark them and let the run's own handler decide.
            for item, _ in loaded:
                self._store.save_error(project_id, item.id, f"batch failed: {exc}")
            raise

        for (item, image), result in zip(loaded, results, strict=True):
            height, width = image.shape[:2]
            self._store.save_result(
                project_id, item.id,
                width=width, height=height,
                task=result.task,
                inference_ms=result.timings.inference_ms,
                review_priority=result.review_priority,
                detections=[d.to_dict() for d in result.detections],
            )
        return len(items)

    def _infer(self, loaded: list[tuple[Item, Any]], settings: RunSettings, generation: int) -> list:
        """One batched pass, or one sliced pass per image when tiling is on."""
        images = [image for _, image in loaded]
        with self._registry.engine(expect_generation=generation) as engine:
            if settings.tiled:
                # Slicing already turns one image into a batch of tiles, so
                # batching across images on top of that would only add latency.
                return [
                    predict_tiled(
                        engine, image,
                        conf=settings.conf,
                        classes=settings.class_filter,
                        tile=settings.tile_size or None,
                        overlap=settings.tile_overlap,
                    )
                    for image in images
                ]
            return engine.predict_batch(images, conf=settings.conf, classes=settings.class_filter)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingRunner:
    """Owns the worker thread that fine-tunes a model on a project's labels.

    A sibling of :class:`ProjectRunner`: same single-flight worker, same "survives
    the tab being closed" contract, but the job is training rather than inference.
    The two share one GPU, so an application only lets one of them run at a time —
    enforced where the requests come in, in :mod:`prelabel.api.projects`.
    """

    def __init__(self, store: Store, registry: ModelRegistry) -> None:
        self._store = store
        self._registry = registry
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active: str | None = None
        self._cancel = threading.Event()

    @property
    def active_project(self) -> str | None:
        with self._lock:
            return self._active

    def is_training(self, project_id: str) -> bool:
        return self.active_project == project_id

    def start(self, project_id: str, settings: TrainingSettings) -> dict[str, Any]:
        """Begin a fine-tune run. Raises if the model can't be trained or one is going."""
        project = self._store.get_project(project_id)
        if project is None:
            raise NotFound(f"No project '{project_id}'")

        checkpoint = self._registry.trainable_checkpoint()
        if checkpoint is None:
            raise Conflict(
                "Load a PyTorch (.pt) model to fine-tune from — an exported "
                "ONNX/OpenVINO model has no trainable weights."
            )
        info = self._registry.require_info()
        if info.get("task") != "detect":
            raise Conflict(
                f"Retraining currently supports detection models; this one is "
                f"'{info.get('task')}'."
            )

        with self._lock:
            if self._active is not None:
                raise Conflict(
                    f"A training run for '{self._active}' is still going. "
                    "Wait for it or cancel it first."
                )
            self._active = project_id
            self._cancel.clear()
            base_names = {int(k): str(v) for k, v in (info.get("class_names") or {}).items()}
            state = {
                "status": "running",
                "detail": "Preparing the training set…",
                "source": settings.source,
                "settings": settings.to_dict(),
                "epoch": 0,
                "epochs": settings.epochs,
                "base_model": info.get("name"),
                "metrics": {},
                "started_at": _now(),
                "finished_at": "",
                "weights": "",
            }
            self._store.update_project(project_id, training=state)
            self._thread = threading.Thread(
                target=self._run,
                args=(project_id, settings, checkpoint, base_names, dict(state)),
                name=f"train-{project_id}",
                daemon=True,
            )
            self._thread.start()
        return state

    def cancel(self, project_id: str) -> bool:
        if not self.is_training(project_id):
            return False
        self._cancel.set()
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _run(
        self,
        project_id: str,
        settings: TrainingSettings,
        checkpoint: Path,
        base_names: dict[int, str],
        state: dict[str, Any],
    ) -> None:
        run_id = uuid.uuid4().hex[:8]
        run_root = config.TRAINING_DIR / project_id / run_id
        data_dir = config.TRAINING_DIR / project_id / f"{run_id}-data"
        store = self._store

        def save(**fields: Any) -> None:
            state.update(fields)
            store.update_project(project_id, training=dict(state))

        try:
            project = store.get_project(project_id)
            if project is None:
                return

            data_dir.mkdir(parents=True, exist_ok=True)
            dataset = training.build_yolo_dataset(store, project, data_dir, settings, base_names)
            save(detail="Training…", dataset=dataset.to_dict())

            # Train from a copy, so the run is safe even if the user unloads the
            # model (which deletes its slot) while training is in flight.
            base = data_dir / "base.pt"
            shutil.copy2(checkpoint, base)

            def on_progress(epoch: int, total: int, metrics: dict[str, float]) -> None:
                save(epoch=epoch, epochs=total, metrics=metrics or state.get("metrics", {}))

            result = training.train_yolo(
                base, dataset.data_yaml, run_root, settings,
                on_progress=on_progress, should_cancel=self._cancel.is_set,
            )

            if self._cancel.is_set():
                save(status="cancelled", detail=f"Stopped after {state.get('epoch', 0)} epochs",
                     weights=str(result.weights), finished_at=_now())
                log.info("Training %s cancelled", project_id)
            else:
                save(status="done", detail="", weights=str(result.weights),
                     metrics=result.metrics or state.get("metrics", {}), finished_at=_now())
                log.info("Training %s finished: %s", project_id, result.metrics)

        except Exception as exc:  # noqa: BLE001 - recorded on the project, not lost
            log.exception("Training %s failed", project_id)
            save(status="failed", detail=str(exc)[:500], finished_at=_now())
        finally:
            # The copied images are scratch; the weights live under run_root.
            shutil.rmtree(data_dir, ignore_errors=True)
            with self._lock:
                self._active = None
                self._thread = None
