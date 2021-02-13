from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.meta_classifier import MetaClassifier
from meta_learning_framework.meta_learning_model import MetaLearningModel
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

        super().__init__(model)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one(self, x):
        return self.model.predict([x])


def my_error(pred: np.array, target: np.array) -> float:

    """
    Sum the absolute error given a prediction and a target (both 2-D).

    Arguments:
        pred (np.ndarray): array which contains one prediction.
        target (np.ndarray): array which contains the target (ground thruth) for the given prediction.

    Returns:
        abs_error (float): sum of absolute error.
    """

    error = abs(pred - target).sum()

    return error


def my_min_combiner(preds: np.array) -> np.array:

    """
    Return the lowest value for each output dimension.

    Arguments:
        preds (np.ndarray): array which contains predictions.

    Returns:
        mins (np.ndarray): array which contains lowest value for each output dimension.
    """
    mins = np.min(preds, axis=0)

    return mins


if __name__ == "__main__":

    # create a artificial time series
    time_series = np.random.uniform(0, 1, 10000)

    WINDOW_SIZE = 25

    # for simplicity, use this tensorflow function to structure the time series dataset
    dataset = timeseries_dataset_from_array(
        time_series[:-WINDOW_SIZE],
        time_series[WINDOW_SIZE:],
        sequence_length=WINDOW_SIZE,
        batch_size=10000 - WINDOW_SIZE,
        shuffle=False,
    )

    # transform Dataset object into array
    for (batch_of_sequences, batch_of_targets) in dataset:

        X = np.array(batch_of_sequences)
        y = np.array(batch_of_targets)

        # simulating a multi output task
        y = np.concatenate(
            [np.expand_dims(y, axis=-1), np.expand_dims(y * 2, axis=-1)], axis=-1
        )

    # split into train and test sets - easy for time series
    train_index = 6000
    X_train, y_train = X[:train_index], y[:train_index]
    X_test, y_test = X[train_index:], y[train_index:]

    # list of base regressors
    bm = [
        LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), "Linear"),
        LocalEstimator(KNeighborsRegressor(n_neighbors=3, n_jobs=-1), "3NN"),
        LocalEstimator(RandomForestRegressor(n_jobs=-1), "RF"),
    ]

    # meta learning framework initialization
    mm = MetaLearningModel(
        LocalMetaClassifier(RandomForestClassifier()),
        bm,
        "regression",
        "score",
        error_measure=my_error,
        combiner=my_min_combiner,
    )

    mm.fit(X_train, y_train, cv=TimeSeriesSplit().split(X_train))
    meta_preds = mm.predict(X_test)

    # evaluation
    print(f"MAE : {abs(meta_preds-y_test).sum(axis=-1).mean():.2f}")

    # naive ensemble for comparison

    # reinitialize list of base classifiers
    bm = [
        LocalEstimator(LinearRegression(normalize=True, n_jobs=-1), "Linear"),
        LocalEstimator(KNeighborsRegressor(n_neighbors=3, n_jobs=-1), "3NN"),
        LocalEstimator(RandomForestRegressor(n_jobs=-1), "RF"),
    ]

    # naive ensemble object
    ne = NaiveEnsemble(bm, "regression", combiner=my_min_combiner)

    # fit and predict methods
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)

    # evaluation
    print(f"MAE : {abs(ne_preds-y_test).sum(axis=-1).mean():.2f}")
