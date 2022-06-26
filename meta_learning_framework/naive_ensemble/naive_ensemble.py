import numpy as np
import pandas as pd
import statistics
from time import time
from tqdm import tqdm
import warnings


class NaiveEnsemble(object):

    """
    Class that implements a bagging ensemble method.

    Arguments:
        models (list): base models list.
        task (str): one option between classification and regression.
        combiner (function): function that combines outputs.
    """

    def __init__(self, models: list, task: str, combiner: "<function>" = None):

        self.models = models
        self.task = task

        # define the combiner function if it was not passed as an argument
        if not combiner:
            self.combiner = statistics.mode if task == "classification" else np.mean
            warnings.warn(
                "You did not pass a combiner function, then it will use a standard one for the selected task."
            )
        else:
            self.combiner = combiner

        self.fit_time = {}
        self.prediction_time = 0

    def fit(self, X, y, verbose=True) -> None:

        """
        It fits base models.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            y (pd.Series, pd.DataFrame or np.ndarray): labels for each instance on X. It has shape (n_instances, ...) as well.
            verbose (boolean): flag to show or not detail information during the process.
        """

        if verbose:
            print("Starting fitting base models:")

        # estimate fit time - start
        self.fit_time = {
            "Fit-" + self.models[i].name: time() for i in range(len(self.models))
        }
        for idx, model in (
            tqdm(enumerate(self.models), total=len(self.models)) if verbose else enumerate(self.models)
        ):

            model.fit(X, y)

            # estimate fit time - end
            self.fit_time["Fit-" + self.models[idx].name] = (
                time() - self.fit_time["Fit-" + self.models[idx].name]
            )

        if verbose:
            print("All base models fitted and ready to prediction.")

    def predict(self, X, verbose=True) -> np.ndarray:

        """
        Iterate over base models to predict X, and combine their output using the combiner function.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            verbose (boolean): flag to show or not detail information during the process.

        Returns:
            predictions (np.ndarray): an array that contains a label for each instance, using the combiner function.
        """

        if verbose:
            print("Starting base models predict:")

        # estimate prediction time - start
        self.prediction_time = time()

        predictions = {}

        for idx, model in (
            tqdm(enumerate(self.models), total=len(self.models)) if verbose else enumerate(self.models)
        ):
            predictions[idx] = model.predict(X)

        # estimate prediction time - end
        self.prediction_time = time() - self.prediction_time

        # combine each base model prediction
        predictions = np.array(
            [
                self.combiner(
                    [
                        predictions[base_model][instance]
                        for base_model in predictions.keys()
                    ]
                )
                for instance in range(X.shape[0])
            ]
        )

        return predictions

    def individual_predict(self, X, verbose=True) -> dict:

        """
        Iterate over base models to predict X individually.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            verbose (boolean): flag to show or not detail information during the process.

        Returns:
            predictions (dict): a dictionary which contains the predictions for each base model using its name as key.
        """

        if verbose:
            print("Starting base models predict:")

        # estimate prediction time - start
        self.prediction_time = time()

        predictions = {}

        for idx, model in (
            tqdm(enumerate(self.models), total=len(self.models)) if verbose else enumerate(self.models)
        ):
            predictions[model.name] = model.predict(X)

        # estimate prediction time - end
        self.prediction_time = time() - self.prediction_time

        return predictions

    def save_time_metrics_csv(self, filename) -> bool:

        """
        Save time metrics into a .csv file given by filename.

        Arguments:
            filename (str): filename (.csv).
        """

        # check extension on filename
        if not filename[-4:] == ".csv":
            warnings.warn(
                "You did not pass the .csv estension, then it will be autocompleted."
            )
            filename = filename + ".csv"

        # each base model fit and prediction time
        self.time_metrics = pd.concat(
            [
                pd.DataFrame([self.fit_time]).T,
                pd.DataFrame([{"Prediction": self.prediction_time}]).T,
            ]
        )

        # renaming column
        self.time_metrics.rename(columns={0: "Time (secs)"}, inplace=True)

        # saving
        self.time_metrics.to_csv(filename)

        return True
