from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



from sklearn.ensemble import HistGradientBoostingRegressor
from pygam import LinearGAM, s

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

TRAIN_SEED = 37
TEST_SEED = 42

XRose: TypeAlias = Annotated[
    npt.NDArray[np.int_],
    "shape=(N, 5)",
]
yRose = npt.NDArray[np.float_]


class RoseModel(Protocol):
    def predict(self, X: XRose) -> yRose:
        ...


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
    seed: int = 37,
) -> tuple[XRose, yRose]:
    np.random.seed(seed)
    X = np.random.randint(1, 7, size=(N, 5))
    X_petals = np.where(X == 3, 2, 0) + np.where(X == 5, 4, 0)
    y = X_petals.sum(axis=1).astype(float)
    return X, y


class Approach(ABC):
    code: str
    title: str

    @abstractmethod
    def train(self, X: XRose, y: yRose) -> RoseModel:
        ...


class Linear(Approach):
    code: str = "lr"
    title: str = "Naïve Linear Regression"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        model = LinearRegression()
        model.fit(X, y)
        return model


class LinearCategoricalFeatures(Approach):
    code: str = "lr_categorical_features"
    title: str = "LR With Categorical Features"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        categorical_encoder = OneHotEncoder(
            categories=[np.arange(1, 7)] * 5,
        )

        model = Pipeline(
            steps=[
                (
                    "one_hot",
                    ColumnTransformer(
                        transformers=[
                            ("dice_faces", categorical_encoder, [0, 1, 2, 3, 4]),
                        ],
                        remainder="drop",
                    ),
                ),
                ("linear", LinearRegression()),
            ]
        )
        model.fit(X, y)
        return model


class BincountModel:
    def __init__(self):
        self.coefficients: yRose | None = None

    @staticmethod
    def bincount(X: XRose) -> npt.NDArray[np.int_]:
        return np.array([
            np.bincount(row, minlength=7)[1:7]
            for row in X
        ])

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

    def train(self, X: XRose, y: yRose) -> RoseModel:
        model = BincountModel()
        model.train(X, y)
        return model


class GAM(Approach):
    code: str = "gam"
    title: str = "GAM"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        model = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4))
        model.fit(X, y)
        return model


class GBTree(Approach):
    code: str = "gb_tree"
    title: str = "GB Tree"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        model = HistGradientBoostingRegressor(
            max_iter=1268,
            learning_rate=0.16379302966980017,
            max_leaf_nodes=3,
            max_depth=2,
            min_samples_leaf=6,
            l2_regularization=1.5335407616629294e-08,
            max_bins=93,
            random_state=37,
        )
        model.fit(X, y)
        return model


class TorchModelWrapper:
    def __init__(self, model: nn.Module, transform=None):
        self.model = model
        self.transform = transform

    def predict(self, X: XRose) -> yRose:
        self.model.eval()

        if self.transform is not None:
            X = self.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            y_hat = self.model(X_tensor).squeeze(1)

        return y_hat.numpy().astype(float)


class NeuralNetwork(Approach):
    code: str = "fcnn"
    title: str = "Neural Network"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        torch.manual_seed(37)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # optimal hyperparameters
        hidden_1 = 500
        hidden_2 = 150
        batch_size = 256
        learning_rate = 0.075
        activation = nn.ReLU

        # only use batch normalization if we have enough rows.
        use_batch = X_tensor.shape[0] >= 3
        
        model = nn.Sequential(
            nn.Linear(5, hidden_1),
            *([nn.BatchNorm1d(hidden_1)] * use_batch),
            activation(),
            nn.Linear(hidden_1, hidden_2),
            *([nn.BatchNorm1d(hidden_2)] * use_batch),
            activation(),
            nn.Linear(hidden_2, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0)

        model.apply(init_weights)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for _ in range(1000):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        return TorchModelWrapper(model)


class DeepSetNN(nn.Module):
    def __init__(self, input_dim, phi_hidden_dim, phi_dim, rho_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Linear(phi_hidden_dim, phi_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_dim),
            nn.ReLU(),
            nn.Linear(rho_dim, 1),
        )

    def forward(self, x):
        phi_x = self.phi(x)
        sum_phi = phi_x.sum(dim=1)
        return self.rho(sum_phi)


class DeepSet(Approach):
    code: str = "deepset"
    title: str = "DeepSet"

    def train(self, X: XRose, y: yRose) -> RoseModel:
        torch.manual_seed(37)

        X_onehot = np.eye(6)[X - 1]
        X_tensor = torch.tensor(X_onehot, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        model = DeepSetNN(input_dim=6, phi_hidden_dim=8, phi_dim=1, rho_dim=8)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for _ in range(2000):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return TorchModelWrapper(model, transform=lambda X: np.eye(6)[X - 1])


def all_approaches() -> list[Approach]:
    return [
        Linear(),
        LinearCategoricalFeatures(),
        Bincount(),
        GAM(),
        GBTree(),
        NeuralNetwork(),
        DeepSet(),
    ]


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

        # early stopping
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

        if at_20.empty:
            row_20 = group.iloc[-1]
        else:
            row_20 = at_20.iloc[0]

        rows.append({
            "approach": approach,
            "min_N_for_100pct_accuracy": min_N_perfect,
            "accuracy_at_N_20": float(row_20["test_accuracy"]),
            "rmse_at_N_20": float(row_20["test_rmse"]),
            "max_accuracy_achieved": float(group["test_accuracy"].max()),
        })

    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> None:
    formatted = summary.rename(columns={
        "approach": "Approach",
        "min_N_for_100pct_accuracy": "N Needed to Solve",
        "accuracy_at_N_20": "Accuracy at N=20",
        "rmse_at_N_20": "RMSE at N=20",
        "max_accuracy_achieved": "Max Accuracy",
    }).copy()

    def fmt_int(x):
        return "-" if pd.isna(x) else f"{int(x)}"

    def fmt_pct(x):
        return "-" if pd.isna(x) else f"{100 * x:.2f}%"

    def fmt_rmse(x):
        return "-" if pd.isna(x) else f"{x:.2f}"

    formatted["N Needed to Solve"] = formatted["N Needed to Solve"].map(fmt_int)
    formatted["Accuracy at N=20"] = formatted["Accuracy at N=20"].map(fmt_pct)
    formatted["RMSE at N=20"] = formatted["RMSE at N=20"].map(fmt_rmse)
    formatted["Max Accuracy"] = formatted["Max Accuracy"].map(fmt_pct)

    return formatted



def plot_learning(
    results: pd.DataFrame,
    max_train_N=1000,
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
    # recalculate for given N
    X_train, y_train = training_data(N, seed=TRAIN_SEED)
    X_test, y_test = training_data(test_N, seed=TEST_SEED)

    model = approach.train(X_train, y_train)
    y_hat = model.predict(X_test).astype(float)

    y_round = np.rint(y_hat)

    rmse = root_mean_squared_error(y_test, y_hat)
    accuracy = np.mean(y_round == y_test)
    accuracy_pm1 = np.mean(np.abs(y_round - y_test) <= 1)

    # scatter plot
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

    # max theoretical range
    plt.xlim(0, 20)
    plt.ylim(0, 20)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.legend(loc="lower right")

    plt.title(f"{approach.title}: Actual vs. Predicted, $N={N}$")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)

    plt.show()

