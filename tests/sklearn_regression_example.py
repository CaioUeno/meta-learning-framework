from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split

from meta_learning_framework.base_models import BaseModel
from meta_learning_framework.meta_classifier import MetaClassifier
from meta_learning_framework.metamodel import MetaLearningModel
from meta_learning_framework.naive_ensemble import NaiveEnsemble

import numpy as np


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

if __name__ == "__main__":

    # X, y = load_boston(return_X_y=True)
    # X, y = load_diabetes(return_X_y=True)
    X, y = fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
        
    bm = [LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), 'Linear'),
          LocalEstimator(SGDRegressor(random_state=11), 'SGDR'),
          LocalEstimator(SVR(), 'SVR'),
          LocalEstimator(KNeighborsRegressor(n_jobs=-1), '5NN'),]

    mm = MetaLearningModel(LocalMetaClassifier(RandomForestClassifier()), bm, "regression", "score")

    a, b = mm.analysis(X_train, y_train)
    print(a)
    print(b)

    # fit and predict methods
    mm.fit(X_train, y_train, cv=10, dynamic_shrink=True)
    meta_preds = mm.predict(X_test)
    print(np.mean(np.square(abs(meta_preds - y_test))))

    bm = [LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), 'Linear'),
          LocalEstimator(SGDRegressor(random_state=11), 'SGDR'),
          LocalEstimator(SVR(), 'SVR'),
          LocalEstimator(KNeighborsRegressor(n_jobs=-1), '5NN'),]

    ne = NaiveEnsemble(bm, "regression")
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)
    print(np.mean(np.square(abs(ne_preds - y_test))))

    ne_preds = ne.individual_predict(X_test)
    for key in ne_preds.keys():
        print(key)
        print(np.mean(np.square(abs(ne_preds[key] - y_test))))