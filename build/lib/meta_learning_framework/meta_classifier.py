from sklearn.base import BaseEstimator
from abc import abstractmethod


class MetaClassifier(BaseEstimator):

    """
    Abstract class to define methods a meta classifier has to have to be used in this framework.

    Arguments:
        model: model.
    """

    def __init__(self, model, **kwargs):

        self.model = model

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_one(self, x):
        pass
