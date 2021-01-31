import numpy as np


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


def absolute_error(preds: np.array, targets: np.array) -> np.array:

    """
    Calculate the absolute error given a prediction and a target.

    Arguments:
        preds (np.ndarray): array which contains predictions.
        targets (np.ndarray): array which contains targets (ground thruth).

    Returns:
        abs_error (float): absolute error.
    """

    abs_error = abs(preds - targets)

    return abs_error
