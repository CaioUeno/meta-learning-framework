import numpy as np

def minimum_error(errors):
    return np.argmin(errors)

def mean_absolute_error(pred, y):
    print(y.shape)
    if len(y.shape) == 1:
        return np.sum(abs(pred - y))
    else:
        return np.sum(abs(pred - y), axis=1)