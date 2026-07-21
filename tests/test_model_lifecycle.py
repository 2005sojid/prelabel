"""Loading, replacing and releasing the model.

The behaviour under test is the one the naive implementation gets wrong: an
upload that cannot be loaded must not cost you the model you already had.
Uploading half an OpenVINO model, an oversized file or a corrupt one all used to
close the running engine *before* discovering the new files were unusable,
leaving the server with nothing.
"""

from __future__ import annotations

import pytest

from prelabel import config, state
from prelabel.errors import ModelLoadError, NoModelLoaded
from prelabel.state import ModelRegistry, clear_stale_slots, normalize_device
from tests.helpers import StubEngine


@pytest.fixture
def fake_builder(monkeypatch):
    """Replace ``build_engine`` with a controllable double.

    Returns a recorder whose ``fail`` flag makes the next build raise, and whose
    ``built`` list records every engine produced.
    """

    class Builder:
        def __init__(self):
            self.fail = False
            self.built = []
            self.calls = []

        def __call__(self, target, imgsz=None, device=None):
            self.calls.append({"target": target, "imgsz": imgsz, "device": device})
            if self.fail:
                raise RuntimeError("unsupported format")
            engine = StubEngine(name=f"engine-{len(self.built)}")
            self.built.append(engine)
            return engine

    builder = Builder()
    monkeypatch.setattr(state, "build_engine", builder)
    return builder


def _slot_with(registry, filename="model.pt"):
    slot = registry.new_slot()
    (slot / filename).write_bytes(b"weights")
    return slot


# --- normalisation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [("cpu", "cpu"), ("gpu", "cuda"), ("cuda", "cuda"), ("CUDA:1", "cuda:1"), ("", None), (None, None)],
)
def test_normalize_device(given, expected):
    assert normalize_device(given) == expected


# --- registry ---------------------------------------------------------------


def test_load_makes_the_model_available(registry, fake_builder):
    slot = _slot_with(registry)
    info = registry.load(slot / "model.pt", slot)

    assert registry.is_loaded
    assert info["name"] == "engine-0"
    with registry.engine() as engine:
        assert engine is fake_builder.built[0]


def test_engine_context_raises_when_nothing_is_loaded(registry):
    with pytest.raises(NoModelLoaded), registry.engine():
        pass


def test_a_failed_load_keeps_the_previous_model(registry, fake_builder):
    """The regression this whole design exists for."""
    first_slot = _slot_with(registry, "good.pt")
    registry.load(first_slot / "good.pt", first_slot)
    original = fake_builder.built[0]

    fake_builder.fail = True
    second_slot = _slot_with(registry, "bad.pt")
    with pytest.raises(ModelLoadError):
        registry.load(second_slot / "bad.pt", second_slot)

    assert registry.is_loaded, "the working model was lost by a failed load"
    with registry.engine() as engine:
        assert engine is original
    assert original.closed is False


def test_a_successful_load_releases_the_previous_engine_and_its_files(registry, fake_builder):
    first_slot = _slot_with(registry, "one.pt")
    registry.load(first_slot / "one.pt", first_slot)
    first_engine = fake_builder.built[0]

    second_slot = _slot_with(registry, "two.pt")
    registry.load(second_slot / "two.pt", second_slot)

    assert first_engine.closed is True
    assert not first_slot.exists(), "the replaced model's files were left on disk"
    assert second_slot.exists()


def test_load_does_not_retry_by_freeing_the_incumbent(registry, monkeypatch):
    """A failed load must not be "rescued" by releasing the working model.

    Retrying after freeing the incumbent would recover memory, but a retry that
    also fails leaves the user with nothing — the exact outcome this class
    exists to prevent. One attempt, and the previous model survives.
    """
    attempts = {"count": 0}

    def builder(target, imgsz=None, device=None):
        attempts["count"] += 1
        if attempts["count"] >= 2:
            raise RuntimeError("CUDA out of memory")
        return StubEngine(name="incumbent")

    monkeypatch.setattr(state, "build_engine", builder)

    first_slot = _slot_with(registry, "one.pt")
    registry.load(first_slot / "one.pt", first_slot)

    second_slot = _slot_with(registry, "two.pt")
    with pytest.raises(ModelLoadError) as failure:
        registry.load(second_slot / "two.pt", second_slot)

    assert attempts["count"] == 2, "the load must be attempted exactly once"
    assert registry.is_loaded, "the working model was released to make room for a failed load"
    assert "unload it first" in str(failure.value), "the error should point at the way out"


def test_device_switch_preserves_the_image_size(registry, fake_builder):
    """Switching device must not silently change how the model runs."""
    slot = _slot_with(registry)
    registry.load(slot / "model.pt", slot, imgsz=768, device="cpu")

    registry.reload_on_device("cuda")

    assert fake_builder.calls[-1]["imgsz"] == 768
    assert fake_builder.calls[-1]["device"] == "cuda"
    assert registry.device_preference == "cuda"


def test_device_switch_without_a_model_is_rejected(registry, fake_builder):
    with pytest.raises(NoModelLoaded):
        registry.reload_on_device("cuda")


def test_unload_releases_everything(registry, fake_builder):
    slot = _slot_with(registry)
    registry.load(slot / "model.pt", slot)
    engine = fake_builder.built[0]

    registry.unload()

    assert registry.is_loaded is False
    assert engine.closed is True
    assert not slot.exists()


def test_clear_stale_slots_removes_orphans(registry):
    """Nothing is loaded at startup, so every slot on disk is by definition dead."""
    slot = _slot_with(registry)
    config.PENDING_DIR.mkdir(parents=True, exist_ok=True)
    (config.PENDING_DIR / "leftover.xml").write_bytes(b"x")

    clear_stale_slots()

    assert not slot.exists()
    assert not config.PENDING_DIR.exists()


def test_generation_changes_whenever_the_model_does(registry, fake_builder):
    """A long job can pin itself to one model with this counter."""
    start = registry.generation

    slot = _slot_with(registry)
    registry.load(slot / "model.pt", slot)
    after_load = registry.generation
    assert after_load != start

    registry.reload_on_device("cuda")
    after_switch = registry.generation
    assert after_switch != after_load

    registry.unload()
    assert registry.generation != after_switch


def test_engine_rejects_a_stale_generation(registry, fake_builder):
    """The guard that stops a video being annotated by two different models."""
    from prelabel.errors import ModelChanged

    slot = _slot_with(registry, "one.pt")
    registry.load(slot / "one.pt", slot)
    pinned = registry.generation

    with registry.engine(expect_generation=pinned):
        pass  # same model — fine

    replacement = _slot_with(registry, "two.pt")
    registry.load(replacement / "two.pt", replacement)

    with pytest.raises(ModelChanged), registry.engine(expect_generation=pinned):
        pass


def test_registries_are_independent(fake_builder):
    """Two applications must not share a model — the factory exists for this."""
    first, second = ModelRegistry(), ModelRegistry()
    slot = first.new_slot()
    (slot / "m.pt").write_bytes(b"w")
    first.load(slot / "m.pt", slot)

    assert first.is_loaded
    assert second.is_loaded is False
