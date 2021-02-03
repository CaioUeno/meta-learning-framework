from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    # sklearn regression datasets - choose one
    X, y = load_boston(return_X_y=True)
    # X, y = load_diabetes(return_X_y=True)
    # X, y = fetch_california_housing(return_X_y=True)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13
    )

    # list of base regressors
    bm = [
        LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), "Linear"),
        LocalEstimator(SVR(), "SVR"),
        LocalEstimator(KNeighborsRegressor(n_neighbors=3, n_jobs=-1), "3NN"),
        LocalEstimator(
            AdaBoostRegressor(n_estimators=100, random_state=11), "AdaBoost"
        ),
    ]

    # meta learning framework initialization
    mm = MetaLearningModel(
        LocalMetaClassifier(RandomForestClassifier()), bm, "regression", "score"
    )

    # run an analysis on correlation between regressors
    pred_corr, error_corr = mm.analysis(X_train, y_train)

    print("Predictions correlation:")
    print(pred_corr)
    print("Errors correlation:")
    print(error_corr)

    # fit and predict methods
    mm.fit(X_train, y_train, cv=10, dynamic_shrink=True)
    meta_preds = mm.predict(X_test)

    # evaluation
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, meta_preds)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, meta_preds)}")
    print(f"R2: {r2_score(y_test, meta_preds)}")

    # naive ensemble for comparison

    # reinitialize list of base classifiers
    bm = [
        LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), "Linear"),
        LocalEstimator(SVR(), "SVR"),
        LocalEstimator(KNeighborsRegressor(n_neighbors=3, n_jobs=-1), "3NN"),
        LocalEstimator(
            AdaBoostRegressor(n_estimators=100, random_state=11), "AdaBoost"
        ),
    ]

    # naive ensemble object
    ne = NaiveEnsemble(bm, "regression")

    # fit and predict methods
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)

    # evaluation
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, ne_preds)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, ne_preds)}")
    print(f"R2: {r2_score(y_test, ne_preds)}")

    # evaluate base models individual performance
    print("individual performance report:")
    individual_preds = ne.individual_predict(X_test)

    for model_name in individual_preds.keys():

        print(
            model_name
            + " - MAE: "
            + str(mean_absolute_error(y_test, individual_preds[model_name]))
            + " / MSE: "
            + str(mean_squared_error(y_test, individual_preds[model_name]))
            + "/ R2 : "
            + str(r2_score(y_test, individual_preds[model_name]))
        )
