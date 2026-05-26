from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..typing import XRose, yRose
from .base import Approach


class BincountModel:
    def __init__(self) -> None:
        self.coefficients: yRose | None = None

    @staticmethod
    def bincount(X: XRose) -> npt.NDArray[np.int_]:
        return np.array([np.bincount(row, minlength=7)[1:7] for row in X])

    def train(self, X: XRose, y: yRose) -> None:
        C = self.bincount(X)
        self.coefficients, *_ = np.linalg.lstsq(C, y, rcond=None)

    def predict(self, X: XRose) -> yRose:
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")

        return self.bincount(X) @ self.coefficients


class Bincount(Approach):
    code: str = "lr_bincount"
    title: str = "Bincount Pivot"

    def train(self, X: XRose, y: yRose):
        model = BincountModel()
        model.train(X, y)
        return model
