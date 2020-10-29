import numpy as np

def minimum_error(errors: np.array) -> np.array:

    arg_minimum = np.argmin(errors)
    label = np.zeros(errors.shape)
    label[arg_minimum] = 1
    return label

def mean_absolute_error(preds: np.array, targets: np.array) -> np.array:
    '''
    It receives two arrays: preds and targets, both with shape (num_instances, 1).
    If your base models predict more than one value, then you can implement a custom error measure,
    but it must return an array with shape (num_instances, 1).

    Returns an array with shape (num_instances, 1).
    '''
    
    return np.mean(abs(preds - targets), axis=1)