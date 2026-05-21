"""Detector backends. Add new ones by subclassing Detector + registering here."""

from .base import Detection, Detector


def load_detector(name: str, config: dict) -> Detector:
    name = (name or "").lower()
    if name == "tensorrt":
        from .tensorrt_detector import TensorRTDetector
        return TensorRTDetector(config)
    if name == "fake":
        from .fake_detector import FakeDetector
        return FakeDetector(config.get("fake_detector", {}))
    raise ValueError(f"unknown detector backend: {name!r} (expected 'tensorrt' or 'fake')")


__all__ = ["Detection", "Detector", "load_detector"]
