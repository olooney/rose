from __future__ import annotations

from typing import Annotated, Protocol

import numpy as np
import numpy.typing as npt

type XRose = Annotated[
    npt.NDArray[np.int_],
    "shape=(N, 5)",
]

type yRose = npt.NDArray[np.float64]


class RoseModel(Protocol):
    def predict(self, X: XRose) -> yRose: ...
