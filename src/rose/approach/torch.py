from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..typing import XRose, yRose
from .base import Approach


class TorchModelWrapper:
    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[XRose], np.ndarray] | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.transform = transform

    def predict(self, X: XRose) -> yRose:
        self.model.eval()

        if self.transform is not None:
            X = self.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_hat = self.model(X_tensor).squeeze(1)

        return y_hat.cpu().numpy().astype(float)


class NeuralNetwork(Approach):
    code: str = "fcnn"
    title: str = "Neural Network"

    def train(self, X: XRose, y: yRose):
        torch.manual_seed(37)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        hidden_1 = 500
        hidden_2 = 150
        batch_size = 256
        learning_rate = 0.075
        activation = nn.ReLU

        use_batch = X_tensor.shape[0] >= 3

        model = nn.Sequential(
            nn.Linear(5, hidden_1),
            *([nn.BatchNorm1d(hidden_1)] * use_batch),
            activation(),
            nn.Linear(hidden_1, hidden_2),
            *([nn.BatchNorm1d(hidden_2)] * use_batch),
            activation(),
            nn.Linear(hidden_2, 1),
        ).to(device)

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0)

        model.apply(init_weights)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for _ in range(1000):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        return TorchModelWrapper(model)


class DeepSetNN(nn.Module):
    def __init__(self, input_dim: int, phi_hidden_dim: int, phi_dim: int, rho_dim: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi_x = self.phi(x)
        sum_phi = phi_x.sum(dim=1)
        return self.rho(sum_phi)


class DeepSet(Approach):
    code: str = "deepset"
    title: str = "DeepSet"

    def train(self, X: XRose, y: yRose):
        torch.manual_seed(37)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_onehot = np.eye(6)[X - 1]
        X_tensor = torch.tensor(X_onehot, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        model = DeepSetNN(input_dim=6, phi_hidden_dim=8, phi_dim=1, rho_dim=8).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for _ in range(2000):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return TorchModelWrapper(model, transform=lambda X: np.eye(6)[X - 1])