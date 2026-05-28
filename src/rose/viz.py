from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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
    plt.xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])

    plt.xlabel("$N$ Training Examples")
    plt.ylabel("Test Accuracy")
    plt.title(f"{approach}: Accuracy vs. Training Set Size")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, which="both", alpha=0.3)

    plt.show()


def plot_summary_learning(
    results: pd.DataFrame | str | Path,
    max_train_N: int | None = None,
) -> None:
    if isinstance(results, pd.DataFrame):
        results_df = results.copy()
    else:
        results_df = pd.read_parquet(Path(results))

    if results_df.empty:
        raise ValueError("results must contain at least one row")

    if max_train_N is None:
        max_train_N = int(results_df["N"].max())

    dashed_approaches = {
        "Naïve Linear Regression",
        "LR With Categorical Features",
        "GAM",
    }
    approach_colors = {
        "Naïve Linear Regression": "#8A8F98",
        "LR With Categorical Features": "#D18F1D",
        "Bincount Pivot": "#2D9D78",
        "GAM": "#D04F4F",
        "GB Tree": "#6E5AE6",
        "Neural Network": "#2A9DBB",
        "DeepSet": "#C65AA0",
    }
    legend_entries: list[tuple[int, str, Line2D]] = []

    plt.figure(figsize=(8.92, 8.92 * 7 / 11))

    for approach, group in results_df.groupby("approach", sort=False):
        group = group.sort_values("N")

        max_accuracy = float(group["test_accuracy"].max())
        first_max_row = group[group["test_accuracy"] == max_accuracy].iloc[0]
        natural_min_max_n = int(first_max_row["N"])
        min_max_n = natural_min_max_n
        legend_label = f"{approach} ($N={min_max_n}$)"
        sort_n = min_max_n
        if str(approach) == "Naïve Linear Regression":
            min_max_n = max_train_N
            first_max_row = group[group["N"] == min_max_n].iloc[0]
            legend_label = str(approach)
            sort_n = natural_min_max_n
        plotted_group = group[group["N"] <= min_max_n]

        (line,) = plt.plot(
            plotted_group["N"],
            plotted_group["test_accuracy"],
            linewidth=1.3,
            alpha=0.75,
            color=approach_colors.get(str(approach)),
            linestyle="--" if str(approach) in dashed_approaches else "-",
            label=legend_label,
        )
        legend_entries.append((sort_n, str(approach), line))
        plt.scatter(
            first_max_row["N"],
            first_max_row["test_accuracy"],
            color=line.get_color(),
            s=28,
            alpha=0.9,
            zorder=3,
        )

    plt.xscale("log")
    plt.xlim(1, max_train_N)

    plt.xlabel("$N$ Training Examples")
    plt.ylabel("Test Accuracy")
    plt.title("Approach Comparison: Accuracy vs. Training Set Size")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, which="both", alpha=0.3)
    legend_entries.sort(key=lambda entry: (entry[0], entry[1]))
    plt.legend(
        handles=[line for _, _, line in legend_entries],
        labels=[line.get_label() for _, _, line in legend_entries],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    plt.show()


def response_surface_grid(
    predictor,
    *,
    feature_count: int,
    fixed_values: tuple[float, ...],
    x_index: int = 0,
    y_index: int = 1,
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_values is None:
        x_values = np.arange(0, 7.01, 0.1)
    if y_values is None:
        y_values = np.arange(0, 7.01, 0.1)

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if len(fixed_values) != feature_count - 2:
        raise ValueError("fixed_values must have length feature_count - 2")

    X, Y = np.meshgrid(x_values, y_values)
    points = np.ones((X.size, feature_count), dtype=float)

    fixed_iter = iter(fixed_values)

    for feature_index in range(feature_count):
        if feature_index == x_index:
            points[:, feature_index] = X.ravel()
        elif feature_index == y_index:
            points[:, feature_index] = Y.ravel()
        else:
            points[:, feature_index] = next(fixed_iter)

    predict = predictor.predict if hasattr(predictor, "predict") else predictor
    Z = np.asarray(predict(points), dtype=float).reshape(X.shape)

    return X, Y, Z


def plot_response_surface(
    predictor,
    *,
    feature_count: int,
    fixed_values: tuple[float, ...],
    x_index: int = 0,
    y_index: int = 1,
    title: str = "Model Response Surface",
    colorscale: str = "Viridis",
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
) -> None:
    import plotly.graph_objects as go

    X, Y, Z = response_surface_grid(
        predictor,
        feature_count=feature_count,
        fixed_values=fixed_values,
        x_index=x_index,
        y_index=y_index,
        x_values=x_values,
        y_values=y_values,
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"x{x_index}",
            yaxis_title=f"x{y_index}",
            zaxis_title="prediction",
        ),
        width=900,
        height=700,
    )

    fig.show()


def plot_response_surface_matplotlib(
    predictor,
    *,
    feature_count: int,
    fixed_values: tuple[float, ...],
    x_index: int = 0,
    y_index: int = 1,
    title: str = "Model Response Surface",
    cmap: str = "viridis",
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
) -> None:
    X, Y, Z = response_surface_grid(
        predictor,
        feature_count=feature_count,
        fixed_values=fixed_values,
        x_index=x_index,
        y_index=y_index,
        x_values=x_values,
        y_values=y_values,
    )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
        shade=True,
    )

    ax.set_title(title)
    ax.set_xlabel(f"x{x_index}")
    ax.set_ylabel(f"x{y_index}")
    ax.set_zlabel("prediction")
    ax.view_init(elev=25, azim=-120)
    ax.set_box_aspect((1, 1, 0.8))

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        pad=0.1,
        label="prediction",
    )

    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Actual vs. Predicted",
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = root_mean_squared_error(y_true, y_pred)
    accuracy = np.mean(np.rint(y_pred) == y_true)
    accuracy_pm1 = np.mean(np.abs(np.rint(y_pred) - y_true) <= 1)

    plt.figure(figsize=(4, 4))

    alpha = min(1.0, 20 / max(len(y_true), 1))
    label = (
        f"RMSE = {rmse:.3f}\n"
        f"Acc = {100 * accuracy:.1f}%\n"
        f"Acc±1 = {100 * accuracy_pm1:.1f}%"
    )

    plt.scatter(
        y_true,
        y_pred,
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
    plt.title(title)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)

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

    plot_prediction_scatter(
        y_test,
        y_hat,
        title=f"{approach.title}: Actual vs. Predicted, $N={N}$",
    )
