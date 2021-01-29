from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
from metamodel import MetaLearningModel
from sklearn.base import BaseEstimator

X, y = load_boston(return_X_y=True)


class DDDD(BaseEstimator):
    def __init__(self, a):

        self.model = DecisionTreeRegressor(max_depth=12)
        self.name = a
        self.a = a

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one(self, x):
        return self.model.predict([x])


class mets(BaseEstimator):
    def __init__(self):

        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one(self, x):
        return self.model.predict([x])


bm = [DDDD("q"), DDDD("r"), DDDD("t")]

mm = MetaLearningModel(mets(), bm, "regression", "score")

# fit and predict methods
mm.fit(X, y, cv=10, dynamic_shrink=False)
meta_preds = mm.predict(X)
print(meta_preds - y)
