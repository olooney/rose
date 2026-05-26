from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rose.approach import GAM, Bincount, DeepSet, GBTree, Linear, NeuralNetwork
from rose.data import sample_sizes, training_data
from rose.evaluation import build_approach_report
from rose.optimize import (
    plot_categorical_hyperparameter,
    plot_hyperparameter_trials_3d,
    plot_numeric_hyperparameter,
)
from rose.viz import (
    plot_learning,
    plot_prediction_scatter,
    plot_response_surface,
    plot_response_surface_matplotlib,
)

# headless matplotlib rendering
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _save_current_figure(output_path: Path) -> None:
    figure = plt.gcf()
    figure.savefig(output_path, dpi=80, bbox_inches="tight")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _headless_show(*args, **kwargs) -> None:
    return None


def test_sample_sizes_and_training_data_are_deterministic() -> None:
    assert sample_sizes(0) == []
    assert sample_sizes(9) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert sample_sizes(25) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    assert sample_sizes(105) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]

    X_1, y_1 = training_data(8, seed=123)
    X_2, y_2 = training_data(8, seed=123)

    assert X_1.shape == (8, 5)
    assert y_1.shape == (8,)
    assert np.array_equal(X_1, X_2)
    assert np.array_equal(y_1, y_2)
    assert np.all((X_1 >= 1) & (X_1 <= 6))

    expected = (np.where(X_1 == 3, 2, 0) + np.where(X_1 == 5, 4, 0)).sum(axis=1)
    assert np.array_equal(y_1, expected.astype(float))


@pytest.mark.parametrize("approach_cls", [Linear, Bincount])
def test_small_approaches_train_and_predict(approach_cls) -> None:
    X_train, y_train = training_data(40, seed=7)
    X_test, _ = training_data(12, seed=11)

    approach = approach_cls()
    model = approach.train(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (12,)
    assert np.isfinite(predictions).all()


def test_other_approach_imports() -> None:
    assert GAM is not None
    assert GBTree is not None
    assert NeuralNetwork is not None
    assert DeepSet is not None


def test_plotly_entry_points_import() -> None:
    assert plot_response_surface is not None
    assert plot_hyperparameter_trials_3d is not None


def test_matplotlib_visualizations_save_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "show", _headless_show)

    learning_results = pd.DataFrame(
        {
            "approach": ["Naive", "Naive", "Naive"],
            "N": [1, 2, 5],
            "test_accuracy": [0.2, 0.5, 0.9],
        }
    )
    plot_learning(learning_results, max_train_N=10)
    _save_current_figure(tmp_path / "learning.png")

    plot_prediction_scatter(
        np.array([0.0, 2.0, 4.0, 6.0]),
        np.array([0.1, 2.2, 3.8, 6.1]),
        title="scatter",
    )
    _save_current_figure(tmp_path / "prediction_scatter.png")

    def predictor(values: np.ndarray) -> np.ndarray:
        return values[:, 0] + values[:, 1]

    plot_response_surface_matplotlib(
        predictor,
        feature_count=5,
        fixed_values=(1.0, 2.0, 3.0),
        x_values=np.array([1.0, 2.0, 3.0]),
        y_values=np.array([1.0, 2.0, 3.0]),
        title="surface",
    )
    _save_current_figure(tmp_path / "surface.png")


def test_optimize_matplotlib_visualizations_save_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "show", _headless_show)

    trials = pd.DataFrame(
        {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "depth": [2, 3, 2, 4],
            "value": [0.9, 0.7, 0.5, 0.4],
        }
    )

    plot_numeric_hyperparameter(trials, "learning_rate", log_x=True)
    _save_current_figure(tmp_path / "numeric.png")

    plot_categorical_hyperparameter(trials, "depth")
    _save_current_figure(tmp_path / "categorical.png")


def test_build_approach_report_for_small_approaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "show", _headless_show)

    summary = build_approach_report(
        [Linear(), Bincount()],
        output_dir=tmp_path,
        scatter_N=10,
        verbose=0,
    )

    assert list(summary["approach"]) == [Linear.title, Bincount.title]
    assert (tmp_path / "summary.html").exists()
    assert (tmp_path / "summary.html").stat().st_size > 0
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "lr" / "learning.png").stat().st_size > 0
    assert (tmp_path / "lr_bincount" / "scatter.png").stat().st_size > 0
