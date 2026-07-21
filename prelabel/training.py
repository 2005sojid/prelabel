"""Fine-tuning a model on a project's corrected labels.

This is the far end of the loop the rest of the tool sets up: a model pre-labels
a folder, you review the least-confident images and correct them in CVAT, import
the corrections back — and then teach the model what it got wrong, so the next
pass over the next folder is better. Compare the retrained model against the old
one with the diff, and you can see whether it actually improved.

Two things here are easy to get wrong, and both are handled deliberately:

**Training on raw predictions teaches nothing.** A model fine-tuned on its own
output just reinforces what it already believes. The labels have to be
*corrections* — which, in this tool's workflow, is what the ``current`` set holds
after you import a corrected COCO over the model's guesses. Nothing here can
prove a set was reviewed, so the honest contract is: this trains on whatever set
you point it at, and it is on you to point it at corrected labels.

**The dataset is built from stored annotations, not re-inferred.** The boxes are
already in the database in pixel coordinates; training just needs them in YOLO's
normalised form, in the layout Ultralytics expects, with a class list that keeps
the base model's head intact when the corrections are a subset of its classes.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, datasets
from .store import Item, Project, Store

log = logging.getLogger("prelabel.training")

#: Which stored set to train on. ``current`` is the working annotations — the
#: corrections, once imported over the model's output; ``baseline`` is the frozen
#: comparison set, useful when the corrections were imported there instead.
SOURCES = ("current", "baseline")


@dataclass
class TrainingSettings:
    """Knobs for one fine-tune run, all with defaults sane for a small set."""

    epochs: int = config.TRAIN_EPOCHS
    imgsz: int = config.TRAIN_IMGSZ
    batch: int = config.TRAIN_BATCH
    val_fraction: float = config.TRAIN_VAL_FRACTION
    min_conf: float = config.TRAIN_MIN_CONF
    #: Which stored set is the ground truth to learn from.
    source: str = "current"
    #: Training device; ``None`` lets Ultralytics pick the GPU if there is one.
    device: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingSettings:
        data = data or {}
        source = str(data.get("source", "current"))
        return cls(
            # Bounds keep a typo ("epochs": 100000) from turning into a run that
            # never ends, and a fractional split from collapsing to all-or-nothing.
            epochs=_clamp(int(data.get("epochs", config.TRAIN_EPOCHS)), 1, 1000),
            imgsz=_round32(_clamp(int(data.get("imgsz", config.TRAIN_IMGSZ)), 32, 1280)),
            batch=_clamp(int(data.get("batch", config.TRAIN_BATCH)), 1, 128),
            val_fraction=_clampf(float(data.get("val_fraction", config.TRAIN_VAL_FRACTION)), 0.05, 0.5),
            min_conf=_clampf(float(data.get("min_conf", config.TRAIN_MIN_CONF)), 0.0, 1.0),
            source=source if source in SOURCES else "current",
            device=(str(data["device"]) if data.get("device") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "val_fraction": self.val_fraction,
            "min_conf": self.min_conf,
            "source": self.source,
        }


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _clampf(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _round32(value: int) -> int:
    """Round an image size to a multiple of 32, which the model's stride needs."""
    return max(32, round(value / 32) * 32)


# --- building the dataset ---------------------------------------------------


@dataclass
class DatasetSummary:
    """What :func:`build_yolo_dataset` produced, for the UI and the training call."""

    data_yaml: Path
    images: int
    boxes: int
    train: int
    val: int
    skipped_empty: int
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "boxes": self.boxes,
            "train": self.train,
            "val": self.val,
            "skipped_empty": self.skipped_empty,
            "classes": self.classes,
            "num_classes": len(self.classes),
        }


def _source_detections(item: Item, source: str) -> list[dict]:
    return item.baseline if source == "baseline" else item.detections


def _class_names(store: Store, project: Project, base_class_names: dict[int, str] | None,
                 settings: TrainingSettings) -> list[str]:
    """The class list for ``data.yaml``.

    Starts from the base model's own classes, in id order, so fine-tuning a
    subset of them leaves the detection head untouched. Any class a correction
    introduces that the model did not have is appended, growing the head rather
    than silently dropping the label.
    """
    names: list[str] = []
    seen: set[str] = set()
    for _, name in sorted((base_class_names or {}).items()):
        if str(name) not in seen:
            names.append(str(name))
            seen.add(str(name))

    for item in store.iter_done_items(project.id):
        for detection in _source_detections(item, settings.source):
            if float(detection.get("confidence", 1.0)) < settings.min_conf:
                continue
            name = str(detection.get("class_name", "object"))
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _split(rel_path: str, val_fraction: float) -> str:
    """Assign an image to train or val, deterministically by path.

    Hashing the path rather than shuffling means the same image lands in the same
    split every run, so a retrain's val metrics are comparable to the last one's.
    """
    digest = int(hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (digest % 1000) / 1000.0 < val_fraction else "train"


def _yolo_lines(item: Item, source: str, min_conf: float, index: dict[str, int]) -> list[str]:
    """One YOLO label line per box: ``class cx cy w h``, all normalised 0–1."""
    width, height = item.width or 0, item.height or 0
    if width <= 0 or height <= 0:
        return []
    lines: list[str] = []
    for detection in _source_detections(item, source):
        if float(detection.get("confidence", 1.0)) < min_conf:
            continue
        box = detection.get("box")
        if not box or len(box) < 4:
            continue
        name = str(detection.get("class_name", "object"))
        class_id = index.get(name)
        if class_id is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        cx = _unit((x1 + x2) / 2 / width)
        cy = _unit((y1 + y2) / 2 / height)
        bw = _unit((x2 - x1) / width)
        bh = _unit((y2 - y1) / height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _labelled_items(store: Store, project: Project, settings: TrainingSettings,
                    index: dict[str, int]) -> Iterator[tuple[Item, list[str]]]:
    """Yield each done item that has at least one usable box, with its label lines.

    Only labelled images go into the set. Empty images make good background
    examples, but including every unlabelled image in a large project would swamp
    the few that were corrected — so v1 trains on what has labels and reports the
    rest as skipped.
    """
    for item in store.iter_done_items(project.id):
        lines = _yolo_lines(item, settings.source, settings.min_conf, index)
        if lines:
            yield item, lines


def build_yolo_dataset(
    store: Store,
    project: Project,
    dest: Path,
    settings: TrainingSettings,
    base_class_names: dict[int, str] | None = None,
) -> DatasetSummary:
    """Materialise a project's labelled images as a YOLO detection dataset.

    Writes ``images/{train,val}`` and ``labels/{train,val}`` under ``dest`` plus a
    ``data.yaml``, and returns what went in. Raises ``ValueError`` when there is
    nothing to learn from, so the caller fails with a reason rather than handing
    Ultralytics an empty directory.
    """
    names = _class_names(store, project, base_class_names, settings)
    if not names:
        raise ValueError("Nothing to train on: no annotations in the chosen set.")
    index = {name: position for position, name in enumerate(names)}

    root = datasets.resolve_dataset_dir(project.source_dir)
    for split in ("train", "val"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    total_boxes = 0
    used_names: set[str] = set()
    written: list[tuple[str, str, list[str]]] = []  # (split, stem, lines)
    seen_stems: set[str] = set()

    for item, lines in _labelled_items(store, project, settings, index):
        try:
            source_path = datasets.resolve_item_path(root, item.rel_path)
        except Exception as exc:  # noqa: BLE001 - a vanished image must not fail the build
            log.warning("Skipping %s while building training set: %s", item.rel_path, exc)
            continue

        stem = _unique_stem(item.rel_path, seen_stems)
        split = _split(item.rel_path, settings.val_fraction)
        _copy_pair(source_path, dest, split, stem, lines)
        written.append((split, stem, lines))
        counts[split] += 1
        total_boxes += len(lines)
        for line in lines:
            used_names.add(names[int(line.split(" ", 1)[0])])

    if not written:
        raise ValueError("Nothing to train on: no labelled images with readable files.")

    _guarantee_val(dest, written, counts)

    data_yaml = _write_data_yaml(dest, names)
    summary = DatasetSummary(
        data_yaml=data_yaml,
        images=len(written),
        boxes=total_boxes,
        train=counts["train"],
        val=counts["val"],
        skipped_empty=0,
        classes=sorted(used_names),
    )
    log.info(
        "Built training set for %s: %d images (%d train / %d val), %d boxes, %d classes",
        project.id, summary.images, summary.train, summary.val, summary.boxes, len(names),
    )
    return summary


def _unique_stem(rel_path: str, seen: set[str]) -> str:
    """A file stem unique within the dataset, so the image and its label match.

    Two subfolders can each hold ``frame_0001.jpg``; flattened into one directory
    the second would overwrite the first — and its label file with it.
    """
    stem = Path(rel_path).stem
    candidate = stem
    if candidate.lower() in seen:
        candidate = rel_path.replace("/", "__").replace("\\", "__")
        candidate = Path(candidate).stem or stem
    suffix = 1
    base = candidate
    while candidate.lower() in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate.lower())
    return candidate


def _copy_pair(source: Path, dest: Path, split: str, stem: str, lines: list[str]) -> None:
    image_name = f"{stem}{source.suffix.lower() or '.jpg'}"
    shutil.copy2(source, dest / "images" / split / image_name)
    (dest / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _guarantee_val(dest: Path, written: list[tuple[str, str, list[str]]], counts: dict[str, int]) -> None:
    """Make sure val is non-empty, so training can actually measure the model.

    With a handful of images the hash split can put everything on one side.
    Rather than fail, borrow one image into val (duplicating it when there is
    only a single image, since a val set of one is better than none).
    """
    if counts["val"] > 0:
        return
    _, stem, _ = written[-1]
    image = _image_file(dest, "train", stem)
    if image is None:
        return
    shutil.copy2(image, dest / "images" / "val" / image.name)
    shutil.copy2(dest / "labels" / "train" / f"{stem}.txt", dest / "labels" / "val" / f"{stem}.txt")
    counts["val"] += 1
    if counts["train"] > 1:
        # With more than one training image, move it rather than duplicate; with
        # only one, keep it in both, since a val set of one still beats none.
        (dest / "images" / "train" / image.name).unlink(missing_ok=True)
        (dest / "labels" / "train" / f"{stem}.txt").unlink(missing_ok=True)
        counts["train"] -= 1


def _image_file(dest: Path, split: str, stem: str) -> Path | None:
    matches = list((dest / "images" / split).glob(f"{stem}.*"))
    return matches[0] if matches else None


def _write_data_yaml(dest: Path, names: list[str]) -> Path:
    """Write Ultralytics' dataset descriptor.

    Written by hand rather than with PyYAML: the document is a fixed shape and
    class names are quoted to survive a stray colon or leading digit.
    """
    lines = [
        f"path: {dest.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {i}: {_quote(name)}" for i, name in enumerate(names))
    path = dest / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _quote(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# --- running the fine-tune --------------------------------------------------


@dataclass
class TrainingResult:
    weights: Path
    metrics: dict[str, float]


METRIC_KEYS = {
    "map50": "metrics/mAP50(B)",
    "map": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def _metrics_from_dict(source: dict | None) -> dict[str, float]:
    if not isinstance(source, dict):
        return {}
    out: dict[str, float] = {}
    for short, key in METRIC_KEYS.items():
        value = source.get(key)
        if value is not None:
            out[short] = round(float(value), 4)
    return out


def _metrics_from_results(results: Any) -> dict[str, float]:
    """Pull the headline metrics out of an Ultralytics results object."""
    from_dict = _metrics_from_dict(getattr(results, "results_dict", None))
    if from_dict:
        return from_dict
    box = getattr(results, "box", None)
    if box is None:
        return {}
    pairs = {"map50": "map50", "map": "map", "precision": "mp", "recall": "mr"}
    return {
        short: round(float(getattr(box, attr)), 4)
        for short, attr in pairs.items()
        if getattr(box, attr, None) is not None
    }


def train_yolo(
    weights: Path,
    data_yaml: Path,
    run_dir: Path,
    settings: TrainingSettings,
    on_progress: Callable[[int, int, dict[str, float]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Fine-tune ``weights`` on ``data_yaml`` and return the best checkpoint.

    Progress and cancellation ride on Ultralytics' epoch callback: after each
    epoch it reports the validation metrics, and if the caller asks to stop, it
    sets the trainer's own stop flag so training ends cleanly at the next epoch
    boundary rather than being killed mid-step.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))

    if on_progress or should_cancel:
        def _on_epoch_end(trainer: Any) -> None:
            total = int(getattr(trainer, "epochs", settings.epochs))
            # Ultralytics fires this once more during the closing validation, with
            # the epoch already past the last — clamp so the UI never shows "2/1".
            epoch = min(int(getattr(trainer, "epoch", 0)) + 1, total)
            if on_progress:
                on_progress(epoch, total, _metrics_from_dict(getattr(trainer, "metrics", None)))
            if should_cancel and should_cancel():
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", _on_epoch_end)

    results = model.train(
        data=str(data_yaml),
        epochs=settings.epochs,
        imgsz=settings.imgsz,
        batch=settings.batch,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=False,
        plots=False,
        deterministic=True,
        seed=0,
        # A dataloader worker subprocess on Windows can hang or fail to pickle the
        # loader; the set is small enough that the main thread is plenty.
        workers=0,
        device=settings.device,
    )

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # A cancelled run may only have written last.pt.
        alt = run_dir / "weights" / "last.pt"
        if alt.exists():
            best = alt
        else:
            raise RuntimeError("Training produced no weights file")

    return TrainingResult(weights=best, metrics=_metrics_from_results(results))
