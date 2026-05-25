from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from .approach.base import Approach
from .data import TEST_SEED, TRAIN_SEED, training_data


def plot_learning(
    results: pd.DataFrame,
    max_train_N: int = 1000,
) -> None:
    approach = str(results["approach"].iloc[0])

    plt.figure(figsize=(6, 4))
    plt.plot(
        results["N"],
        results["test_accuracy"],
        marker="o",
        linewidth=1,
        alpha=0.6,
        markeredgewidth=0,
    )

    plt.xscale("log")
    plt.xlim(1, max_train_N)

    plt.xlabel("$N$ Training Examples")
    plt.ylabel("Test Accuracy")
    plt.title(f"{approach}: Accuracy vs. Training Set Size")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, which="both", alpha=0.3)

    plt.show()


def plot_scatter(
    approach: Approach,
    *,
    N: int,
    test_N: int = 200,
) -> None:
    X_train, y_train = training_data(N, seed=TRAIN_SEED)
    X_test, y_test = training_data(test_N, seed=TEST_SEED)

    model = approach.train(X_train, y_train)
    y_hat = model.predict(X_test).astype(float)

    y_round = np.rint(y_hat)

    rmse = root_mean_squared_error(y_test, y_hat)
    accuracy = np.mean(y_round == y_test)
    accuracy_pm1 = np.mean(np.abs(y_round - y_test) <= 1)

    plt.figure(figsize=(4, 4))

    alpha = min(1.0, 20 / test_N)

    label = (
        f"RMSE = {rmse:.3f}\n"
        f"Acc = {100 * accuracy:.1f}%\n"
        f"Acc±1 = {100 * accuracy_pm1:.1f}%"
    )

    plt.scatter(
        y_test,
        y_hat,
        alpha=alpha,
        edgecolors="none",
        label=label,
    )

    plt.axline(
        (0, 0),
        slope=1,
        linestyle="--",
        color="gray",
        alpha=0.5,
        lw=1,
    )

    plt.xlim(0, 20)
    plt.ylim(0, 20)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.legend(loc="lower right")

    plt.title(f"{approach.title}: Actual vs. Predicted, $N={N}$")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)

    plt.show()