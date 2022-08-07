import numpy as np
import pytest
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
