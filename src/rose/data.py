from __future__ import annotations

import numpy as np

from .typing import XRose, yRose

TRAIN_SEED = 37
TEST_SEED = 42


def sample_sizes(max_N: int) -> list[int]:
    if max_N < 1:
        return []

    sizes = list(range(1, min(max_N, 10) + 1))

    step = 10
    start = 20

    while start <= max_N:
        stop = min(max_N, step * 10)
        sizes.extend(range(start, stop + 1, step))

        step *= 10
        start *= 10

    return sizes


def training_data(
    N: int = 10,
    seed: int = TRAIN_SEED,
) -> tuple[XRose, yRose]:
    np.random.seed(seed)
    X = np.random.randint(1, 7, size=(N, 5))
    X_petals = np.where(X == 3, 2, 0) + np.where(X == 5, 4, 0)
    y = X_petals.sum(axis=1).astype(float)
    return X, y