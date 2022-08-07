from abc import ABC, abstractmethod
from typing import Any

from meta_learning_framework.types import Instances, Targets
from sklearn.base import BaseEstimator


class MetaClassifier(BaseEstimator, ABC):

    """
    Abstract class to implement the meta classifier interface.

    Parameters
    ----------
        model : Any
            Machine learning model to use as meta classifier.
    """
    
    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def fit(self, X: Instances, y: Targets) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: Instances) -> Targets:
        raise NotImplementedError()
