from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .approach.base import Approach
from .data import TEST_SEED, TRAIN_SEED, sample_sizes, training_data
from .typing import yRose
from .viz import plot_learning, plot_scatter


@dataclass
class ReportApproachContext:
    code: str
    title: str
    learning_plot: str
    scatter_plot: str
    metrics: dict[str, str]


def evaluate_predictions(y: yRose, y_hat: yRose) -> dict[str, float]:
    y_hat_rounded = np.rint(y_hat)

    return {
        "rmse": float(root_mean_squared_error(y, y_hat)),
        "mae": float(mean_absolute_error(y, y_hat)),
        "accuracy": float(np.mean(y_hat_rounded == y)),
    }


def evaluate_approach(
    approach: Approach,
    *,
    max_train_N: int = 1_000,
    test_N: int = 10_000,
    verbose: int = 0,
) -> pd.DataFrame:
    X_train_full, y_train_full = training_data(max_train_N, seed=TRAIN_SEED)
    X_test, y_test = training_data(test_N, seed=TEST_SEED)

    rows = []
    perfect = 0

    for N in sample_sizes(max_train_N):
        X_train = X_train_full[:N]
        y_train = y_train_full[:N]

        model = approach.train(X_train, y_train)

        y_hat_train = model.predict(X_train).astype(float)
        y_hat_test = model.predict(X_test).astype(float)

        train_metrics = evaluate_predictions(y_train, y_hat_train)
        test_metrics = evaluate_predictions(y_test, y_hat_test)

        rows.append({
            "approach": approach.title,
            "N": N,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_accuracy": train_metrics["accuracy"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_accuracy": test_metrics["accuracy"],
        })

        if verbose >= 1:
            print(".", end="")

        perfect = perfect + 1 if test_metrics["accuracy"] == 1.0 else 0
        if perfect >= 2:
            if verbose >= 1:
                print(f"stopping early at N={N}", end="")
            break

    print()

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for approach, group in results.groupby("approach", sort=False):
        group = group.sort_values("N")

        perfect = group[group["test_accuracy"] == 1.0]
        min_N_perfect = None if perfect.empty else int(perfect["N"].iloc[0])

        at_20 = group[group["N"] >= 20]
        row_20 = group.iloc[-1] if at_20.empty else at_20.iloc[0]

        rows.append({
            "approach": approach,
            "min_N_for_100pct_accuracy": min_N_perfect,
            "accuracy_at_N_20": float(row_20["test_accuracy"]),
            "rmse_at_N_20": float(row_20["test_rmse"]),
            "max_accuracy_achieved": float(group["test_accuracy"].max()),
        })

    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    formatted = summary.rename(columns={
        "approach": "Approach",
        "min_N_for_100pct_accuracy": "N Needed to Solve",
        "accuracy_at_N_20": "Accuracy at N=20",
        "rmse_at_N_20": "RMSE at N=20",
        "max_accuracy_achieved": "Max Accuracy",
    }).copy()

    formatted["N Needed to Solve"] = formatted["N Needed to Solve"].map(_format_int)
    formatted["Accuracy at N=20"] = formatted["Accuracy at N=20"].map(_format_percent)
    formatted["RMSE at N=20"] = formatted["RMSE at N=20"].map(_format_number)
    formatted["Max Accuracy"] = formatted["Max Accuracy"].map(_format_percent)

    return formatted


def build_approach_report(
    approaches: list[Approach],
    *,
    output_dir: str | Path | None = None,
    scatter_N: int = 20,
    verbose: int = 1,
) -> pd.DataFrame:
    output_path = Path(output_dir) if output_dir is not None else default_report_dir()

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)

    results = []
    approach_contexts: list[ReportApproachContext] = []

    for approach in approaches:
        if verbose >= 1:
            print(f"Trying {approach.title} approach...")

        approach_dir = output_path / approach.code
        approach_dir.mkdir()

        df = evaluate_approach(approach, verbose=verbose)
        results.append(df)

        df.to_csv(approach_dir / "results.csv", index=False)
        df.to_parquet(approach_dir / "results.parquet", index=False)

        learning_path = approach_dir / "learning.png"
        scatter_path = approach_dir / "scatter.png"

        _capture_plot(plot_learning, learning_path, df)
        _capture_plot(plot_scatter, scatter_path, approach, N=scatter_N)

        approach_contexts.append(
            ReportApproachContext(
                code=approach.code,
                title=approach.title,
                learning_plot=f"{approach.code}/learning.png",
                scatter_plot=f"{approach.code}/scatter.png",
                metrics={},
            )
        )

    results_df = pd.concat(results, ignore_index=True)
    summary = summarize_results(results_df)

    results_df.to_csv(output_path / "results.csv", index=False)
    results_df.to_parquet(output_path / "results.parquet", index=False)
    summary.to_csv(output_path / "summary.csv", index=False)
    summary.to_parquet(output_path / "summary.parquet", index=False)

    summary_by_title = summary.set_index("approach")

    for context in approach_contexts:
        row = summary_by_title.loc[context.title]
        context.metrics = {
            "Min N for 100% Accuracy": _format_int(row["min_N_for_100pct_accuracy"]),
            "Accuracy at N=20": _format_percent(row["accuracy_at_N_20"]),
            "RMSE at N=20": _format_number(row["rmse_at_N_20"]),
            "Max Accuracy Achieved": _format_percent(row["max_accuracy_achieved"]),
        }

    html = _template_environment().get_template("summary.html").render(
        summary_table=_format_summary_table(summary),
        approaches=approach_contexts,
    )
    (output_path / "summary.html").write_text(html, encoding="utf-8")

    return summary


def default_report_dir() -> Path:
    return _project_root() / "docs" / "approaches"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capture_plot(plot_function, path: Path, *args, **kwargs) -> None:
    original_show = plt.show

    try:
        plt.close("all")
        plt.show = lambda *a, **k: None

        plot_function(*args, **kwargs)

        fig = plt.gcf()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    finally:
        plt.show = original_show


def _template_environment() -> Environment:
    return Environment(
        loader=PackageLoader("rose", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _format_percent(x: object) -> str:
    return "-" if pd.isna(x) else f"{100 * float(x):.2f}%"


def _format_number(x: object) -> str:
    return "-" if pd.isna(x) else f"{float(x):.2f}"


def _format_int(x: object) -> str:
    return "-" if pd.isna(x) else f"{int(x)}"


def _format_summary_table(summary: pd.DataFrame) -> str:
    return format_summary(summary).to_html(
        index=False,
        classes="table table-striped table-sm align-middle",
        border=0,
        escape=False,
    )