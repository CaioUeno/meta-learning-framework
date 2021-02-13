import numpy as np
from sklearn.model_selection import KFold


def minimum_error(errors: np.array) -> np.array:

    """
    It receives one array of errors with shape (num_base_models) which contains
    the errors for each base model for a given instance.
    Then it chooses the base model that has the lowest error.

    It returns an one-hot array where the respective index of the selected base model
    has an one, and the other positions are set to zero. Basically, it returns
    the meta model label for the given instance.

    Arguments:
        errors (np.ndarray): array which contains the errors for each base model for a given instance.

    Returns:
        label (np.ndarray): one-hot encoded array.
    """

    arg_minimum = np.argmin(errors)
    label = np.zeros(errors.shape)
    label[arg_minimum] = 1

    return label


def absolute_error(pred: np.array, target: np.array) -> np.array:

    """
    Calculate the absolute error given a prediction and a target.

    Arguments:
        pred (np.ndarray): array which contains one prediction.
        target (np.ndarray): array which contains the target (ground thruth) for the given prediction.

    Returns:
        abs_error (float): absolute error.
    """

    abs_error = abs(pred - target)

    return abs_error


def proba_mean_error(pred: np.array, target: np.array) -> np.array:

    """
    Calculate the mean error given a prediction of classes probabilities and a target (one-hot encoded).

    Arguments:
        pred (np.ndarray): array which contains one prediction with length n_classes.
        target (np.ndarray): array which contains the target (ground thruth) for the given prediction with length n_classes as well.

    Returns:
        prob_mean_error (float): probabilities mean error.
    """

    prob_mean_error = np.mean(abs(pred - target))

    return prob_mean_error
