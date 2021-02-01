from abc import abstractmethod
from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):

    """
    Abstract class to define methods a base model has to have to be used in this framework.

    Arguments:
        model: base model.
        name (str): name for this model - necessary to differenciate each base model.
    """

    def __init__(self, model, name: str):

        self.model = model

        if not name:
            raise TypeError("You must pass a name for every base model.")
        else:
            self.name = name
        
        # self.classes_ = 0

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_one(self, x):
        pass
