import time
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class NaiveEnsemble(object):

    """

    """

    def __init__(self, models):

        self.models = models

    def fit(self, X, y):

        self.fit_time = {
            "Fit-" + self.models[i].name: time.time() for i in range(len(self.models))
        }
        for idx, model in enumerate(self.models):
            model.fit(X, y)
            self.fit_time["Fit-" + self.models[idx].name] = (
                time.time() - self.fit_time["Fit-" + self.models[idx].name]
            )

    def predict(self, X):

        predictions = np.zeros((X.shape[0], len(self.models)))
        self.prediction_time = time.time()
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(X)
        self.prediction_time = time.time() - self.prediction_time

        return np.array([statistics.mode(prediction) for prediction in predictions])

    def save_performance_metrics(self, path, y_true, y_pred):

        """
            Save performance metrics into a .csv file given by path.

            Arguments:
                path (str): file's path (.csv). 
        """

        self.performance_metrics = pd.DataFrame()
        self.performance_metrics["metric"] = [
            "accuracy",
            "precision",
            "recall",
            "f1-score",
        ]
        self.performance_metrics["value"] = [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="micro"),
            recall_score(y_true, y_pred, average="micro"),
            f1_score(y_true, y_pred, average="micro"),
        ]
        self.performance_metrics.to_csv(path, index=False)

    def save_time_metrics(self, path):

        """
            Save time metrics into a .csv file given by path.

            Arguments:
                path (str): file's path (.csv). 
        """

        self.time_metrics = pd.concat(
            [
                pd.DataFrame([self.fit_time]).T,
                pd.DataFrame([{"Prediction": self.prediction_time}]).T,
            ]
        )
        self.time_metrics.rename(columns={0: "Time (secs)"}, inplace=True)
        self.time_metrics.to_csv(path)
