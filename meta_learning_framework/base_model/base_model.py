from abc import ABC, abstractmethod
from typing import Any

from meta_learning_framework.types import Instances, Targets
from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator, ABC):

    """
    Abstract class to wrap a base model (classifier or regressor).
    It implements its interface methods.

    Parameters
    ----------
        name : str
            Model name (required to differentiate models);
        model : Any
            Actually machine learning model.

    Raises
    ---------
        TypeError
            If provided name is not from string type.
    """

    def __init__(self, name: str, model: Any = None) -> None:

        self._validate_name(name)
        self.name = name
        self.model = model

    @staticmethod
    def _validate_name(name: str) -> bool:

        if isinstance(name, str) and name is not None:
            return True

        raise TypeError(f"Parameter name must be of type string, {type(name)} was passed.")

    @abstractmethod
    def fit(self, X: Instances, y: Targets) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: Instances) -> Targets:
        raise NotImplementedError()
