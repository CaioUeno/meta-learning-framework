import statistics
from typing import Any

import numpy as np
import pytest
from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.meta import MetaClassifier
from meta_learning_framework.types import Instances, Target, Targets
from meta_learning_framework.utils import Combiner, ErrorMeasurer
from meta_learning_framework.utils.selector import Selector
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


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


@pytest.fixture(scope="module")
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


# classification task


@pytest.fixture(scope="class")
def classification_data_sample():

    X = np.random.random((10, 5))
    y = np.random.randint(low=0, high=1, size=10)

    return (X, y)


@pytest.fixture(scope="class")
def concrete_classification_error_measurer():
    class EM(ErrorMeasurer):
        def measure(self, base_preds: Targets, y_true: Targets) -> Targets:
            return np.asarray(base_preds != y_true).astype(int)

    return EM


@pytest.fixture
def concrete_classification_selector():
    class SimpleSelector(Selector):
        def select(self, base_errors: Targets) -> Targets:
            return np.asarray(base_errors == 0).astype(int)

    return SimpleSelector


@pytest.fixture(scope="class")
def concrete_classification_combiner():
    class SimpleCombiner(Combiner):
        def combine(self, base_preds: Targets) -> Target:
            return statistics.mode(base_preds)

    return SimpleCombiner


@pytest.fixture(scope="class")
def mle_classification_init_params(
    concrete_meta_clasifier_class,
    concrete_base_model_class,
    concrete_classification_combiner,
    concrete_classification_error_measurer,
    concrete_classification_selector,
):

    return {
        "meta_classifier": concrete_meta_clasifier_class(
            model=KNeighborsClassifier(n_neighbors=3)
        ),
        "base_models": [
            concrete_base_model_class(
                name="Decision Tree", model=DecisionTreeClassifier()
            ),
            concrete_base_model_class(
                name="Nearest Neighbors", model=KNeighborsClassifier()
            ),
            concrete_base_model_class(name="Naive Bayes", model=GaussianNB()),
        ],
        "combiner": concrete_classification_combiner(),
        "error_measurer": concrete_classification_error_measurer(),
        "selector": concrete_classification_selector(),
    }


# regression task


@pytest.fixture(scope="class")
def concrete_regression_error_measurer():
    class EM(ErrorMeasurer):
        def measure(self, base_preds: Targets, y_true: Targets) -> Targets:
            return np.asarray(np.abs(base_preds - y_true))

    return EM


@pytest.fixture
def concrete_regression_selector():
    class SimpleSelector(Selector):
        def select(self, base_errors: Targets) -> Targets:
            return np.ravel(np.argmin(base_errors == 0))[0].astype(int)

    return SimpleSelector


@pytest.fixture(scope="class")
def concrete_regression_combiner():
    class SimpleCombiner(Combiner):
        def combine(self, base_preds: Targets) -> Target:
            return np.mean(base_preds)

    return SimpleCombiner
