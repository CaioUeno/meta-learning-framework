from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

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

    # create a artificial time series
    time_series = np.sin(np.linspace(0, 100, 10000)) + np.linspace(0, 100, 10000) + (np.random.rand(10000) / 2)
    
    WINDOW_SIZE = 15
    
    # for simplicity, use this tensorflow function to structure the time series dataset
    dataset = timeseries_dataset_from_array(time_series[:-WINDOW_SIZE], time_series[WINDOW_SIZE:],
                                            sequence_length=WINDOW_SIZE, batch_size=10000-WINDOW_SIZE, shuffle=False)
    
    for (batch_of_sequences, batch_of_targets) in dataset:
        X, y = np.array(batch_of_sequences), np.array(batch_of_targets)

    # split into train and test sets - easy for time series
    train_index = 6000
    X_train, y_train = X[:train_index], y[:train_index]
    X_test, y_test = X[train_index:], y[train_index:]

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

    mm.fit(X_train, y_train, cv=TimeSeriesSplit().split(X_train))
    meta_preds = mm.predict(X_test)

    # evaluation
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, meta_preds)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, meta_preds)}")
    print(f"R2: {r2_score(y_test, meta_preds)}")

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



    