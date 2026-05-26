from .base import Approach
from .numpy import Bincount
from .pygam import GAM
from .sklearn import GBTree, Linear, LinearCategoricalFeatures
from .torch import DeepSet, NeuralNetwork


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


__all__ = [
    "Approach",
    "Bincount",
    "DeepSet",
    "GAM",
    "GBTree",
    "Linear",
    "LinearCategoricalFeatures",
    "NeuralNetwork",
    "all_approaches",
]
