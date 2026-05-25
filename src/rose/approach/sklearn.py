from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..typing import RoseModel, XRose, yRose
from .base import Approach


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
        return model  # type: ignore[return-value]


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