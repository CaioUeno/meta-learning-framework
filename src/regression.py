from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_boston
from metamodel import MetaLearningModel
from base_models import BaseModel
from meta_nn import MetaClassifier

from naive_ensemble import NaiveEnsemble

import numpy as np

X, y = load_boston(return_X_y=True)

class LocalEstimator(BaseModel):

    def __init__(self, model, name: str):

        super().__init__(model, name)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one(self, x):
        return self.model.predict([x])

class LocalMetaClassifier(MetaClassifier):

    def __init__(self, model):

        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one(self, x):
        return self.model.predict([x])


bm = [LocalEstimator(DecisionTreeRegressor(max_depth=2), 'One'),
      LocalEstimator(DecisionTreeRegressor(max_depth=4), 'Two'),
      LocalEstimator(DecisionTreeRegressor(max_depth=8), 'Three')]

mm = MetaLearningModel(LocalMetaClassifier(DecisionTreeClassifier()), bm, "regression", "score")

# a, b = mm.analysis(X, y)
# print(a)
# print(b)


# fit and predict methods
mm.fit(X, y, cv=10, dynamic_shrink=True)
meta_preds = mm.predict(X)
print(np.mean(abs(meta_preds - y)))

ne = NaiveEnsemble(bm, "regression")
ne.fit(X, y)
ne_preds = ne.predict(X)
print(np.mean(abs(ne_preds - y)))