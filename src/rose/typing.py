from __future__ import annotations

from typing import Annotated, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

XRose: TypeAlias = Annotated[
    npt.NDArray[np.int_],
    "shape=(N, 5)",
]
yRose = npt.NDArray[np.float64]


class RoseModel(Protocol):
    def predict(self, X: XRose) -> yRose: ...
