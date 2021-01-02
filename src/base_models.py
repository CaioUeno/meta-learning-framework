import numpy as np
from sktime.classification.base import BaseClassifier

class LocalClassifier(BaseClassifier):
    
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):

        # if it is only one instance
        if isinstance(X, np.ndarray):
            return self.model.predict(X[0].values.reshape(1, 1, -1))

        # otherwise is a whole dataset
        return self.model.predict(X)

        