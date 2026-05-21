"""Detector ABC. Keep the surface tiny so swapping engines stays cheap."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def warmup(self) -> None:
        """One or two throwaway inferences so the first prod frame doesn't pay cold-start."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR HxWx3 frame and return vehicle detections."""

    def close(self) -> None:
        pass
