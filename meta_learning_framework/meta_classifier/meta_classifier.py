from abc import abstractmethod
from typing import Any

from meta_learning_framework.types import Instances, Targets
from sklearn.base import BaseEstimator


class MetaClassifier(BaseEstimator):
    def __init__(self, model: Any) -> None:

        self.model = model

    @abstractmethod
    def fit(self, X: Instances, y: Targets) -> None:
        pass

    @abstractmethod
    def predict(self, X: Instances) -> Targets:
        pass
