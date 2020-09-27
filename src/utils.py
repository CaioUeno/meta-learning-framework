import numpy as np

def minimum_error(errors):

    arg_minimum = np.argmin(errors)
    label = np.zeros(errors.shape)
    label[arg_minimum] = 1
    return label

def mean_absolute_error(pred, y):

    if len(y.shape) == 1:
        return np.sum(abs(pred - y))
    else:
        return np.sum(abs(pred - y), axis=1)