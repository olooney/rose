from __future__ import annotations

from pygam import LinearGAM, s

from ..typing import RoseModel, XRose, yRose
from .base import Approach


class GAM(Approach):
    code: str = "gam"
    title: str = "GAM"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        model = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4))  # type: ignore[arg-type]
        model.fit(X, y)
        return model