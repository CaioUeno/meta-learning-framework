import statistics
from typing import Any

import numpy as np
import pytest
from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.meta_classifier import MetaClassifier
from meta_learning_framework.types import Instances, Target, Targets
from meta_learning_framework.utils import Combiner, ErrorMeasurer
from meta_learning_framework.utils.selector import Selector


@pytest.fixture
def classification_sample():
    X = np.random.random((10, 5))
    y = np.random.randint(low=0, high=1, size=10)
    return (X, y)


@pytest.fixture
def regression_sample():
    X = np.random.random((10, 5))
    y = np.random.random(10)
    return (X, y)


@pytest.fixture
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


@pytest.fixture
def concrete_meta_clasifier_class():

    # define a simple concrete class for MetaClassifier
    class MC(MetaClassifier):
        def __init__(self, model: Any = None) -> None:
            super().__init__(model)

        def fit(self, X: Instances, y: Targets) -> None:
            self.model.fit(X, y)

        def predict(self, X: Instances) -> Targets:
            return self.model.predict(X)

    return MC


@pytest.fixture
def concrete_combiner():

    # define a simple concrete class for Combiner
    class SimpleCombiner(Combiner):
        def combine(self, base_preds: Targets) -> Target:
            return statistics.mode(base_preds)

    return SimpleCombiner


@pytest.fixture
def concrete_error_measurer():

    # define a simple concrete class for ErrorMeasurer
    class EM(ErrorMeasurer):
        def measure(self, base_preds: Targets, y_true: Targets) -> Targets:
            return np.asarray(base_preds != y_true).astype(int)

    return EM

@pytest.fixture
def concrete_selector():

    # define a simple concrete class for Selector
    class SimpleSelector(Selector):
        def select(self, base_errors: Targets) -> Targets:
            return np.asarray(base_errors == 0).astype(int)

    return SimpleSelector

@pytest.fixture
def useless_class():

    class UselessClass(object):
        pass

    return UselessClass