from typing import Any

import numpy as np
import pytest
from meta_learning_framework import BaseModel
from meta_learning_framework.types import Instances, Targets
from sklearn.dummy import DummyClassifier, DummyRegressor


@pytest.fixture(scope="class")
def diff_tasks_models():
    return [DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")]


@pytest.fixture(scope="class")
def not_str_objects():
    return [-21, 0.7, {"name": "dummy"}, ["dummy"], {"dummy"}]


@pytest.fixture(scope="function")
def data_sample():

    X = np.random.random((10, 5))
    y = np.random.randint(low=0, high=1, size=10)

    return (X, y)


@pytest.fixture(scope="module")
def concrete_base_model_class():

    # define a simple concrete class for BaseModel
    class Dummy(BaseModel):
        def __init__(self, name: str, model: Any = None) -> None:
            super().__init__(name, model)

        def fit(self, X: Instances, y: Targets) -> None:
            self.model.fit(X, y)

        def predict(self, X: Instances) -> Targets:
            return self.model.predict(X)

    return Dummy
