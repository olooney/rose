from __future__ import annotations

from abc import ABC, abstractmethod

from ..typing import RoseModel, XRose, yRose


class Approach(ABC):
    code: str
    title: str

    @abstractmethod
    def train(self, X: XRose, y: yRose) -> RoseModel: ...
