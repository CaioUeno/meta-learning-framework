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

        """
        Note: y will have shape (n_instances, n_base_models), even if it is only a multi-class task (not a multi-label).
        Keep this in mind while implementing this method. 
        Suggestion: if you want a "one-number label" (0, 1, 2, ...) and not a one-hot encoding,
        simply apply numpy.argmax(y) and then you have a numerical class.
        """

        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_one(self, x):
        pass
