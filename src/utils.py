import numpy as np

def minimum_error(errors: np.array) -> np.array:

    '''
        It receives one array of errors with shape (num_base_models) which contains
        the errors for each base model for a given instance.
        Then it chooses the base model that has the lowest error.

        It returns an one-hot array where the respective index of the selected base model
        has an one, and the other positions are set to zero. Basically, it returns
        the meta model label for the given instance.
        
        Returns an array with shape (num_base_models).
    '''

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