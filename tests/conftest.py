from typing import Any
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from meta_learning_framework.base_model.base_model import BaseModel
from meta_learning_framework.types import Instances, Targets


@pytest.fixture
def classifier():
    return DummyClassifier(strategy="most_frequent")


@pytest.fixture
def sample():
    X = np.random.random((10, 5))
    y = np.random.randint(low=0, high=1, size=10)
    return (X, y)

@pytest.fixture
def base_classifier(classifier):
    
    class DC(BaseModel):

        def __init__(self, name: str, model: Any = None) -> None:
            super().__init__(name, model)

        def fit(self, X: Instances, y: Targets) -> None:
            self.model.fit(X, y)

        def predict(self, X: Instances) -> Targets:
            return self.model.predict(X)
