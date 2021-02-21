import numpy as np
import pandas as pd
import statistics
from time import time
from tqdm import tqdm
from types import GeneratorType
import warnings

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# own library
from meta_learning_framework.utils import (
    absolute_error,
    proba_mean_error,
    minimum_error,
)


class MetaLearningModel(object):

    """
    Class for a Meta-Learning model. It is a model that applies a machine learning technique
    to better ensemble base models.

    Arguments:
        meta_model (classifier): classifier to select base models.
        base_models (list): base models list.
        task (str): one option between classification and regression.
        mode (str): one option between binary and score.
        multi_label (bool): if the given base estimator as the meta_model supports a multiclass classification task - presume it does not.
        combiner (function): function that combines outputs.
        error_measure (function): function that calculates the error between the true and predicted label.
        selector (function): function that implements the condition to a base model be selected given its error (error_measure).
    """

    def __init__(
        self,
        meta_model,
        base_models: list,
        task: str,
        mode: str,
        multi_label: bool = False,
        combiner: "<function>" = None,
        error_measure: "<function>" = None,
        selector=minimum_error,
    ):

        if self.__check_args(
            meta_model, base_models, task, mode, combiner, error_measure, selector
        ):

            self.meta_models = meta_model
            self.base_models = base_models
            self.task = task
            self.mode = mode
            self.multi_label = multi_label
            self.selector = selector

        # attributes that will appear inside methods
        self.y_shape = None
        self.X_meta_models = None
        self.y_meta_models = None
        self.prediction_base_models_used = None
        self.cross_validation_time = {}
        self.partial_fit_time = {}
        self.fit_time = {}
        self.meta_fit_time = 0
        self.prediction_time = 0

    def __check_args(
        self,
        meta_model,
        base_models: list,
        task: str,
        mode: str,
        combiner: "<function>",
        error_measure: "<function>",
        selector: "<function>",
    ) -> bool:

        """
        Check some arguments to initialize the class properly. Also, it does
        some inferences to undefined variables - if so.

        Arguments:
            meta_model (classifier): base estimator to select base models.
            base_models (list): base models list.
            task (str): one option between classification and regression.
            mode (str): one option between binary and score.
            combiner (function): function that combines outputs.
            error_measure (function): function that calculate the error between the true and predicted label.
            selector (function): function that implements the condition to a base model be selected given its error (error_measure).

        Returns:
            True (boolean): all checks passed.
        """

        # check task argument
        if task not in ["classification", "regression"]:
            raise ValueError("Must choose a task: classification or regression.")

        # regression only works with score mode
        if task == "regression" and mode == "binary":
            mode = "score"
            warnings.warn(
                "Regression tasks only support score mode. Using score instead of binary mode."
            )

        # check if meta model and base models have the necessary methods for the task
        if (
            "fit" not in dir(meta_model)
            or "predict" not in dir(meta_model)
            or "predict_one" not in dir(meta_model)
        ):
            raise TypeError(
                "The meta_model must have method fit(X, y), predict(X) and predict_one(X)."
            )

        for base_model in base_models:

            if (
                "fit" not in dir(base_model)
                or "predict" not in dir(base_model)
                or "predict_one" not in dir(base_model)
            ):
                raise TypeError(
                    "Al base models must have method fit(X, y), predict(X) and predict_one(X)."
                )

            # for this specific combination, base models must have a predict method that returns the probabilities
            if (
                task == "classification"
                and mode == "score"
                and "predict_proba" not in dir(base_model)
            ):
                raise TypeError(
                    "If classification task and score mode chosen then base models must have method predict_proba(X)."
                )

        # define the error measure function if it was not passed as an argument
        if not error_measure:

            # classification task and binary mode does not use a error measure function
            if task == "classification" and mode == "binary":
                self.error_measure = None

            elif task == "classification" and mode == "score":
                self.error_measure = proba_mean_error

            # regression (only works with score mode)
            else:
                self.error_measure = absolute_error

            warnings.warn(
                "You did not pass a error_measure function, then it will use a standard one for the selected task and mode."
            )

        else:
            self.error_measure = error_measure

        # define the combiner function if it was not passed as an argument
        if not combiner:

            # standard functions for classification and regression
            self.combiner = statistics.mode if task == "classification" else np.mean
            warnings.warn(
                "You did not pass a combiner function, then it will use a standard one for the selected task."
            )
        else:
            self.combiner = combiner

        # everything is all right!
        return True

    def fit(self, X, y, cv=10, refit=True, verbose=True, dynamic_shrink=True) -> None:

        """
        First, it creates the meta model's training set using a cross-validation method (check it for more information).
        Then, after targets are checked, it trains both levels - base models and meta model(s).

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            y (pd.Series, pd.DataFrame or np.ndarray): targets for each instance on X. It has shape (n_instances, ...) as well.
            cv (int, float or generator): if int check cross_val_predict sklearn function for more information,
                                          if float means the validation portion of X to be used to create the meta model training set,
                                          and, finally, if generator it will use it as a generator of train/test indexes.
            refit (boolean): flag to refit base models after the cross-validation step - only applied if cv is a float.
            verbose (boolean): flag to show or not detail information during the process.
            dynamic_shrink (boolean): flag to remove or not unselected base models in meta model training set creation.
        """

        # check X and y type
        if not isinstance(X, (pd.DataFrame, np.ndarray)) or not isinstance(
            y, (pd.core.series.Series, pd.DataFrame, np.ndarray)
        ):
            raise TypeError(
                "X must be as type pd.DataFrame or np.ndarray, and y must be as type pd.core.series.Series, pd.DataFrame or np.ndarray."
            )

        # check cv type and value
        if not isinstance(cv, (int, float, GeneratorType)):
            raise TypeError("cv must be as type int, float or generator.")

        if isinstance(cv, float) and not (cv > 0 and cv < 1):
            raise TypeError("If cv passed as float, then it must be in (0, 1) range.")

        # only accept refit argument if cv passed as float
        refit = refit if isinstance(cv, float) else True

        # store y shape
        self.y_shape = None if len(y.shape) == 1 else y.shape[1:]

        # create meta model training set
        X_meta_models, y_meta_models = self.cross_validation(
            X, y, cv=cv, verbose=verbose
        )

        if verbose:
            print("Cross-validation done and meta model training set created.")

        # check meta model y (targets)
        y_meta_models = self.check_targets(y_meta_models, dynamic_shrink)

        if verbose:
            print("Meta model targets checked.")

        # store meta model's training set
        self.X_meta_models, self.y_meta_models = (
            X_meta_models,
            y_meta_models,
        )

        if refit:
            # fit both levels - meta model and base models
            self.fit_both_levels((X, y), (X_meta_models, y_meta_models))

        else:
            self.fit_meta_models(X_meta_models, y_meta_models)

        if verbose:
            print("Base models and meta model(s) fitted and ready to prediction.")

    def predict(self, X, verbose=True) -> np.ndarray:

        """
        The meta model predicts for each instance which base models are going to be selected.
        Then, it combines only selected base models' predictions.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            verbose (boolean): flag to show or not detail information during the process.

        Returns:
            predictions (np.ndarray): an array that contains a label for each instance, using the combiner function.
        """

        # check X type
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be as type pd.DataFrame or np.ndarray.")

        if self.task == "classification" or not self.y_shape:
            predictions = np.zeros(len(X))

        # regression task can output more than one value, so it need to infer the predictions array shape
        else:
            shape = list(self.y_shape)
            shape.insert(0, len(X))
            predictions = np.zeros(shape)

        # store how many base models were selected for each instance (metric)
        self.prediction_base_models_used = np.zeros(len(X))

        # estimate prediction time - start
        self.prediction_time = time()

        # iterate over instances
        for idx, x in tqdm(enumerate(X), total=len(X)) if verbose else enumerate(X):

            # meta model prediction tells which base models to use
            selected_base_models = self.predict_meta_models(x)

            # used base models
            self.prediction_base_models_used[idx] = np.sum(selected_base_models)

            # if none base model was selected, then select all of them (similar to the bagging approach)
            if not np.any(selected_base_models):
                selected_base_models[:] = 1

            # combine selected base models predictions in one final prediction
            final_prediction = self.combiner(
                self.predict_base_models(x, selected_base_models)
            )

            predictions[idx] = final_prediction

        # estimate prediction time - end
        self.prediction_time = time() - self.prediction_time

        return predictions

    def fit_both_levels(self, X_y_base_models: tuple, X_y_meta_models: tuple) -> None:

        """
        It fits base models and meta models.

        Arguments:
            X_y_base_models (tuple): training set of base models (X, y).
            X_y_meta_models (tuple): training set of meta model (X, y).
        """

        X_base_models, y_base_models = X_y_base_models
        self.fit_base_models(X_base_models, y_base_models)

        X_meta_models, y_meta_models = X_y_meta_models
        self.fit_meta_models(X_meta_models, y_meta_models)

    def fit_base_models(self, X, y) -> None:

        """
        It fits base models.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            y (pd.Series, pd.DataFrame or np.ndarray): targets for each instance on X. It has shape (n_instances, ...) as well.
        """

        # estimate fit time for each base model - start
        self.fit_time = {
            "Fit-" + self.base_models[i].name: time()
            for i in range(len(self.base_models))
        }

        for idx, base_model in enumerate(self.base_models):

            base_model.fit(X, y)

            # estimate fit time for each base model - end
            self.fit_time["Fit-" + self.base_models[idx].name] = (
                time() - self.fit_time["Fit-" + self.base_models[idx].name]
            )

    def predict_base_models(
        self, x: np.ndarray, selected_base_models: np.ndarray
    ) -> list:

        """
        Given a instance x, and a mask selected_base_models - it contains which base models were
        selected for this instance - returns a list containing selected base models' predictions.

        Arguments:
            x (np.ndarray): a single instance.
            selected_base_models (np.ndarray): array with length n_base_models, which contains which base models were selected to predict this instance.

        Returns:
            predictions (list): A list which contains only the ** selected ** base models predictions for the given instance.
        """

        predictions = []
        for idx, base_model in enumerate(self.base_models):
            if selected_base_models[idx] == 1:
                predictions.append(base_model.predict_one(x))

        return predictions

    def fit_meta_models(self, X, y) -> None:

        """
        It fits meta models.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            y (pd.Series, pd.DataFrame or np.ndarray): meta labels for each instance on X. It has shape (n_instances, n_base_models).
        """

        # save training time for meta classifier - start
        self.meta_fit_time = time()

        # meta model that supports a multi-label task
        if self.multi_label == True:
            self.meta_models.fit(X, y)

        else:

            # single meta model for a multi-class task
            if self.n_meta_models == 1:
                self.meta_models.fit(X, np.argmax(y, axis=1))

            # a set of binary meta models that do not support a multi-label task but simulate it
            else:
                for idx, meta_model in enumerate(self.meta_models):
                    meta_model.fit(X, y[:, idx])

        # save training time for meta classifier - end
        self.meta_fit_time = time() - self.meta_fit_time

    def predict_meta_models(self, x: np.ndarray) -> np.ndarray:

        """
        Given a instance x, it returns the meta models prediction for it.

        Arguments:
            x (np.ndarray): a single instance.

        Returns:
            predictions (np.ndarray): Array with length n_base_models indicating which base models are going to be used.
        """

        # single meta model that supports a multi-label task
        if self.multi_label == True:
            predictions = self.meta_models.predict_one(x)

        else:

            # single meta model for a multi-class task
            if self.n_meta_models == 1:
                predictions = np.zeros(len(self.base_models))
                predictions[self.meta_models.predict_one(x)] = 1

            # a set of binary meta models that do not support a multi-label task but simulate it
            else:
                predictions = np.zeros(len(self.meta_models))
                for idx, meta_model in enumerate(self.meta_models):
                    predictions[idx] = meta_model.predict_one(x)

        return predictions

    def cross_validation(self, X, y, cv, verbose: bool) -> tuple:

        """
        Cross-validation/Partial fit for each base model.
        It is actually creating a training set for the meta model (X, y).
        X is the instances as they are, but y is which base models should be used to predict each instance.

        It has three different flows for each possible cv type:
        1) int: full cross-validation method using cv folds.
        2) float: partial cross-validation method using cv as a validation portion.
        3) generator: custom cross-validation method using cv as a generator of train and test indexes.

        It has three possible flows to decide how to select base models:
        1) classification task and binary mode: It checks if the base models labeled correctly or not
        each instance - it is selected if so (Soft criterion).
        2) classification task and score mode: Apply the error measure function at the score distribuition for
        each instance and the selector function as well.
        3) regression (only works with score mode): Same as 2) but applied to the numerical output.

        Arguments:
            X (pd.DataFrame or np.ndarray): an object with shape (n_instances, ...).
            y (pd.Series, pd.DataFrame or np.ndarray): targets for each instance on X. It has shape (n_instances, ...) as well.
            cv (int, float or generator): if int check cross_val_predict sklearn function for more information,
                                          if float means the validation portion of X to be used to create the meta model training set,
                                          and, finally, if generator it will use it as a generator of train/test indexes.
            verbose (boolean): flag to show or not detail information during the process.

        Returns:
            X (pd.DataFrame or np.ndarray): same as input - or a portion of it.
            y (pd.Series, pd.DataFrame or np.ndarray): Array which contains for each instance which base models were selected.
                                                       It has shape (n_instances, n_base_models).
        """

        if isinstance(cv, int):

            """
            Full cross-validation method using cv folds. For more information check sklearn cross_val_predict function.
            """

            if verbose:
                print("Starting cross-validation:")

            # save training time for each base model - start
            self.cross_validation_time = {
                "CV-" + self.base_models[i].name: time()
                for i in range(len(self.base_models))
            }

            # cross validation for each base model and store its prediction
            base_models_predictions = {}
            for idx, base_model in (
                tqdm(enumerate(self.base_models), total=len(self.base_models))
                if verbose
                else enumerate(self.base_models)
            ):

                base_models_predictions[idx] = cross_val_predict(
                    base_model, X, y, cv=cv, method=self.__adapt_method()
                )

                # save training time for each base model - end
                self.cross_validation_time["CV-" + self.base_models[idx].name] = (
                    time()
                    - self.cross_validation_time["CV-" + self.base_models[idx].name]
                )

            # instances that were predicted on this step
            X_meta_models = X.copy()
            y_true = y.copy()

        # float in [0, 1] range
        elif isinstance(cv, float):

            """
            Partial cross-validation method using cv as a validation portion.
            """

            if verbose:
                print("Starting partial-fit:")

            # save training time for each base model - start
            self.partial_fit_time = {
                "PF-" + self.base_models[i].name: time()
                for i in range(len(self.base_models))
            }

            # split data using cv as a validation set fraction
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=cv, random_state=0
            )

            # fit base models using a data fraction only once and store their prediction
            base_models_predictions = {}
            for idx, base_model in (
                tqdm(enumerate(self.base_models), total=len(self.base_models))
                if verbose
                else enumerate(self.base_models)
            ):

                base_model.fit(X_train, y_train)

                # save training time for each base model - end
                self.partial_fit_time["PF-" + self.base_models[idx].name] = (
                    time() - self.partial_fit_time["PF-" + self.base_models[idx].name]
                )

                # decide which prediction method to use
                if self.task == "classification" and self.mode == "score":
                    base_models_predictions[idx] = base_model.predict_proba(X_valid)
                else:
                    base_models_predictions[idx] = base_model.predict(X_valid)

            # instances that were predicted on this step
            X_meta_models = X_valid.copy()
            y_true = y_valid.copy()

        else:

            """
            Custom cross-validation method using cv as a generator of train and test indexes splits.
            This option does not store the training time for each base model due to a double for loop (it prioritize performance).
            """

            if verbose:
                print("Starting custom cross-validation:")

            # get instance shape
            X_shape = list(X.shape[1:])
            X_shape.insert(0, 0)

            # variable to store instances that are going to be predicted on this step
            X_meta_models = np.zeros(tuple(X_shape))

            y_shape = list(self.y_shape)
            y_shape.insert(0, 0)
            y_true = np.zeros(y_shape) if self.y_shape else np.zeros((0))

            base_models_predictions = {
                idx: np.zeros(y_shape) if self.y_shape else np.zeros((0))
                for idx, base_model in enumerate(self.base_models)
            }

            # iterate over train/test indexes on generator
            for idx, (train_index, test_index) in (
                tqdm(enumerate(cv)) if verbose else enumerate(cv)
            ):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # increase the set over iterations - both X and y
                X_meta_models = np.concatenate([X_meta_models, X_test])
                y_true = np.concatenate([y_true, y_test])

                # fit all base models
                self.fit_base_models(X_train, y_train)

                for idx, base_model in enumerate(self.base_models):

                    # prediction step
                    # decide which prediction method to use
                    if self.task == "classification" and self.mode == "score":
                        y_pred = base_model.predict_proba(X_test)
                    else:
                        y_pred = base_model.predict(X_test)

                    y_pred = (
                        np.expand_dims(y_pred, axis=0) if len(y_pred) == 1 else y_pred
                    )

                    # increase y_pred over iterations as well
                    base_models_predictions[idx] = np.concatenate(
                        [base_models_predictions[idx], y_pred]
                    )

        # now, extract information of correctness from y_pred for each base model
        # and create meta model training set

        # decide which approach to follow based on task and mode
        if self.mode == "binary":

            # binary multi-label task dataset - if prediction is equal to true label, then base model is selected
            y_target_meta_models = np.zeros((y_true.shape[0], len(self.base_models)))

            for idx, base_model in enumerate(self.base_models):
                y_target_meta_models[:, idx] = (
                    base_models_predictions[idx] == y_true
                ).astype(int)

            return (X_meta_models, y_target_meta_models)

        elif self.task == "classification" and self.mode == "score":

            ohe = OneHotEncoder(sparse=False)

            y_error_meta_models = np.zeros((y_true.shape[0], len(self.base_models)))

            # given an error measure function and a selector funtion, select "only" useful base models
            for idx, base_model in enumerate(self.base_models):
                y_error_meta_models[:, idx] = self.__measure_error(
                    base_models_predictions[idx], ohe.fit_transform(y_true.reshape(-1, 1))
                )

            return (X_meta_models, self.__selector(y_error_meta_models))

        # regression task that depends on an error measure and a selection function
        else:

            y_error_meta_models = np.zeros((y_true.shape[0], len(self.base_models)))

            # given an error measure function and a selector funtion, select "only" useful base models
            for idx, base_model in enumerate(self.base_models):
                y_error_meta_models[:, idx] = self.__measure_error(
                    base_models_predictions[idx],
                    y_true,
                )

            return (X_meta_models, self.__selector(y_error_meta_models))

    def __adapt_method(self) -> str:

        """
        Decide which method is the correct one based on task and mode to pass to cross_val_predict function.

        Returns:
            method name (string): method name to be used to predict {predict, predict_proba}.
        """

        if self.task == "regression" or (
            self.task == "classification" and self.mode == "binary"
        ):
            return "predict"
        else:
            return "predict_proba"

    def check_targets(
        self, y_meta_models: np.ndarray, dynamic_shrink: bool
    ) -> np.ndarray:

        """
        It checks if there is any base model that was not selected for any instance.
        Those base models can be removed, because they are not going to be useful on prediction.
        Also, this function infers if the meta model task will be a multi-label or not
        (If more than one base model were selected for any instance then it will be a multi-label task).

        Arguments:
            y_meta_models (np.ndarray): Array which contains for each instance which base models were selected. It has shape (n_instances, n_base_models).
            dynamic_shrink (bool): Flag to remove unused base models.

        Returns:
            treated y_meta_models (np.ndarray): Array which contains for each instance which base models were selected after checks.
        """

        # flag to remove base models that were not selected for any instance
        if dynamic_shrink:

            # sum over rows to return how many times each base model were selected
            sum_up = np.sum(y_meta_models, axis=0)

            # list of base models to remove
            models_to_remove = []

            for i in range(sum_up.shape[0]):

                # base model at index i were never selected
                if sum_up[i] == 0:
                    print(
                        "Base classifier "
                        + self.base_models[i].name
                        + " was never selected, thus it will be removed from set."
                    )
                    models_to_remove.append(i)

            # remove in reverse order to not harm the order
            for i in sorted(models_to_remove, reverse=True):
                del self.base_models[i]

            treated_y_meta_models = y_meta_models[:, sum_up != 0]

        else:
            treated_y_meta_models = y_meta_models

        # check if there is any instance that has more than one base model assigned to it
        # and if the meta model doesn't support a multi-label task.
        # Finally, infers how many meta models it will need based on those checks.
        if (
            np.any(np.sum(treated_y_meta_models, axis=1) > 1)
            and self.multi_label == False
        ):
            self.meta_models = [self.meta_models for _ in range(len(self.base_models))]
            self.n_meta_models = len(self.base_models)

        else:
            self.n_meta_models = 1

        return treated_y_meta_models

    def analysis(self, X, y, validation_split=0.2, random_state=None) -> tuple:

        """
        Depending on the task, it analyses how much the base models are "correlated"/similar.
        If classification, it returns two matrices: first one contains how many times two base models labeled
        correctly the same instance. And the second one contains how many times two base models labeled incorrectly
        the same instance but as the same wrong class.
        If regression, then it returns two correlation matrices: one is the correlation between the base models' output,
        and the other is the correlation between the base models' output **errors**.
        """

        # check X and y type
        if not isinstance(X, (pd.DataFrame, np.ndarray)) or not isinstance(
            y, (pd.core.series.Series, pd.DataFrame, np.ndarray)
        ):
            raise TypeError(
                "X must be as type pd.DataFrame or np.ndarray, and y must be as type pd.core.series.Series, pd.DataFrame or np.ndarray."
            )

        # split into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=validation_split, random_state=random_state
        )

        # store base models predictions
        predictions = {}
        for idx, base_model in enumerate(self.base_models):

            base_model.fit(X_train, y_train)
            predictions[base_model.name] = base_model.predict(X_valid)

        if self.task == "classification":

            accuracy_correlation = pd.DataFrame(
                columns=[base_model.name for base_model in self.base_models],
                index=[base_model.name for base_model in self.base_models],
            )
            mislabel_correlation = pd.DataFrame(
                columns=[base_model.name for base_model in self.base_models],
                index=[base_model.name for base_model in self.base_models],
            )

            for m_one in self.base_models:
                for m_two in self.base_models:

                    # how many times both labeled the same instances correctly
                    accuracy_correlation.loc[m_one.name, m_two.name] = (
                        (predictions[m_one.name] == predictions[m_two.name])
                        .astype(int)
                        .sum()
                    )

                    # how many times both labeled the same instances incorrectly but using the same wrong label
                    mislabel_correlation.loc[m_one.name, m_two.name] = (
                        (
                            (predictions[m_one.name] == predictions[m_two.name])
                            & (predictions[m_one.name] != y_valid)
                        )
                        .astype(int)
                        .sum()
                    )

            # normalize both matrices
            accuracy_correlation /= len(y_valid)
            mislabel_correlation /= len(y_valid)

            return (accuracy_correlation, mislabel_correlation)

        else:

            predictions_correlation_matrix = pd.DataFrame(predictions).corr()
            error_correlation_matrix = pd.DataFrame(
                {
                    k: self.__measure_error(predictions[k], y_valid)
                    for k in predictions.keys()
                }
            ).corr()

            return (predictions_correlation_matrix, error_correlation_matrix)

    def __measure_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

        """
        Calculate the error for each instance given the error measure function.

        Arguments:
            y_pred (np.ndarray): array which contains the prediction for each instance.
            y_true (np.ndarray): array which contains the ground truth for each instance.

        Returns:
            y_error (np.ndarray): array which contains the error for each instance.
        """

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Meta Classifier training set: y_true {y_true.shape} and y_pred {y_pred.shape} do not have the same shape.")

        y_error = np.zeros((y_pred.shape[0]))
        for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
            y_error[idx] = self.error_measure(pred, true)

        return y_error

    def __selector(self, y_error_meta_models: np.ndarray) -> np.ndarray:

        """
        Apply the selector function for each error array.

        Arguments:
            y_error_meta_models (np.ndarray): Array which contains errors for each base model for each instance. It has shape (n_instances, n_base_models).

        Returns:
            target_meta_models (np.ndarray): Array which contains which base models were selected to be used for each instance. It has same shape as input.
        """

        target_meta_models = np.zeros(y_error_meta_models.shape)
        for i in range(y_error_meta_models.shape[0]):
            target_meta_models[i] = self.selector(y_error_meta_models[i])

        return target_meta_models

    def save_base_models_used(self, filename: str) -> bool:

        """
        Save a numpy array which contains for each instance how many base models were used.

        Arguments:
            filename (str): filename (.npy).
        """

        # check extension on filename
        if not filename[-4:] == ".npy":
            warnings.warn(
                "You did not pass the .npy extension, then it will be autocompleted."
            )
            filename = filename + ".npy"

        # saving
        np.save(filename, self.prediction_base_models_used)

        return True

    def save_time_metrics_csv(self, filename: str) -> bool:

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

        # each base model fit and cross-validation time
        self.time_metrics = pd.concat(
            [
                pd.DataFrame([self.fit_time]).T,
                pd.DataFrame([self.cross_validation_time]).T,
                pd.DataFrame([self.partial_fit_time]).T,
            ]
        )

        # meta model fit and prediction time
        self.time_metrics = pd.concat(
            [self.time_metrics, pd.DataFrame([{"MetaModel-Fit": self.meta_fit_time}]).T]
        )
        self.time_metrics = pd.concat(
            [self.time_metrics, pd.DataFrame([{"Prediction": self.prediction_time}]).T]
        )

        # renaming column
        self.time_metrics.rename(columns={0: "Time (secs)"}, inplace=True)

        # saving
        self.time_metrics.to_csv(filename)

        return True
