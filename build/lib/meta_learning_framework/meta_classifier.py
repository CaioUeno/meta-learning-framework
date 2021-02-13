from abc import abstractmethod
from sklearn.base import BaseEstimator


class MetaClassifier(BaseEstimator):

    """
    Abstract class to define methods a meta classifier has to have to be used in this framework.

    Arguments:
        model: model.
    """

    def __init__(self, model):

        self.model = model

    @abstractmethod
    def fit(self, X, y):
        """
        Note: Check fit_meta_models method on meta_learning_model.py to see y's shape.
        """
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_one(self, x):
        pass
