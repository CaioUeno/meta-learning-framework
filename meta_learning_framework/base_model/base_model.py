from abc import abstractmethod
from typing import Any

from meta_learning_framework.types import Instances, Targets
from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def __init__(self, name: str, model: Any = None) -> None:

        self._validate_name(name)
        self.name = name
        self.model = model

    @staticmethod
    def _validate_name(name: str) -> bool:

        if isinstance(name, str) and name is not None:
            return True

        raise ValueError(f"")

    @abstractmethod
    def fit(self, X: Instances, y: Targets) -> None:
        pass

    @abstractmethod
    def predict(self, X: Instances) -> Targets:
        pass
