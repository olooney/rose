from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _filter_trials(
    trials_df: pd.DataFrame,
    *,
    hyperparameters: list[str],
    target: str,
    greater_is_better: bool,
    remove_bottom: float,
) -> pd.DataFrame:
    df = trials_df[[*hyperparameters, target]].dropna()

    if df.empty:
        return df

    if greater_is_better:
        cutoff = df[target].quantile(remove_bottom)
        return df[df[target] >= cutoff]

    cutoff = df[target].quantile(1 - remove_bottom)
    return df[df[target] <= cutoff]


def plot_numeric_hyperparameter(
    trials_df: pd.DataFrame,
    hyperparameter: str,
    target: str = "value",
    *,
    log_x: bool = False,
    bandwidth: float | None = None,
    bandwidth_fraction: float = 0.1,
    greater_is_better: bool = False,
    remove_bottom: float = 0.1,
) -> None:
    df = _filter_trials(
        trials_df,
        hyperparameters=[hyperparameter],
        target=target,
        greater_is_better=greater_is_better,
        remove_bottom=remove_bottom,
    )

    if df.empty:
        return

    x = df[hyperparameter].to_numpy()
    y = df[target].to_numpy()
    x_smooth_space = np.log10(x) if log_x else x

    order = np.argsort(x_smooth_space)
    x = x[order]
    y = y[order]
    x_smooth_space = x_smooth_space[order]

    if bandwidth is None:
        span = x_smooth_space.max() - x_smooth_space.min()
        bandwidth = bandwidth_fraction * span if span > 0 else 1.0

    y_smooth = np.empty_like(y)

    for index, x_value in enumerate(x_smooth_space):
        weights = np.exp(-0.5 * ((x_smooth_space - x_value) / bandwidth) ** 2)
        y_smooth[index] = np.sum(weights * y) / np.sum(weights)

    plt.figure()
    plt.title(f"{hyperparameter} vs. {target}")
    plt.xlabel(hyperparameter)
    plt.ylabel(target)

    if log_x:
        plt.xscale("log")

    plt.scatter(x, y, alpha=0.4, s=20, edgecolors="none")
    plt.plot(x, y_smooth, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_categorical_hyperparameter(
    trials_df: pd.DataFrame,
    hyperparameter: str,
    target: str = "value",
    *,
    greater_is_better: bool = False,
    remove_bottom: float = 0.1,
) -> None:
    df = _filter_trials(
        trials_df,
        hyperparameters=[hyperparameter],
        target=target,
        greater_is_better=greater_is_better,
        remove_bottom=remove_bottom,
    )

    if df.empty:
        return

    grouped = df.groupby(hyperparameter)[target]
    categories = [str(category) for category in grouped.groups.keys()]
    values = [group.to_numpy() for _, group in grouped]
    counts = grouped.size().to_numpy()

    plt.figure()
    plt.title(f"{hyperparameter} vs. {target}")
    plt.xlabel(hyperparameter)
    plt.ylabel(target)
    plt.boxplot(
        values,
        labels=categories,
        widths=counts / counts.max(),
        showmeans=False,
    )
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_hyperparameter_trials_3d(
    trials_df: pd.DataFrame,
    x_hyperparameter: str,
    y_hyperparameter: str,
    *,
    target: str = "value",
    title: str = "Hyperparameter Trials",
    greater_is_better: bool = False,
    remove_bottom: float = 0.2,
) -> None:
    import plotly.graph_objects as go

    df = _filter_trials(
        trials_df,
        hyperparameters=[x_hyperparameter, y_hyperparameter],
        target=target,
        greater_is_better=greater_is_better,
        remove_bottom=remove_bottom,
    )

    if df.empty:
        return

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df[x_hyperparameter],
                y=df[y_hyperparameter],
                z=df[target],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df[target],
                    colorscale="RdYlGn",
                    reversescale=not greater_is_better,
                    opacity=0.75,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_hyperparameter,
            yaxis_title=y_hyperparameter,
            zaxis_title=target,
        ),
        width=900,
        height=700,
    )

    fig.show()
