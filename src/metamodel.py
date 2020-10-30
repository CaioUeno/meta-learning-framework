from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelBinarizer
import statistics
import numpy as np
import warnings
from utils import mean_absolute_error, minimum_error

class MetaLearningModel(object):

    def __init__(self, meta_model, base_models: list, task: str, mode: str, combiner: '<function>'=None,
                 error_measure=mean_absolute_error, chooser=minimum_error):

        if self.check_args(meta_model, base_models, task, mode, combiner, error_measure, chooser):

            self.meta_models = meta_model
            self.base_models = base_models
            self.task = task
            self.mode = mode
            self.error_measure = error_measure
            self.chooser = chooser
            

    def check_args(self, meta_model, base_models: list, task: str, mode: str, combiner: '<function>',
                   error_measure: '<function>', chooser: '<function>'):

        if task not in ['classification', 'regression']:
            raise ValueError('Must choose a task: classification or regression.')

        if task == 'regression' and mode == 'binary':
            mode = 'score'
            warnings.warn('Regression tasks only support score mode. Using score instead of binary mode.')

        if 'fit' not in dir(meta_model) or 'predict' not in dir(meta_model):
            raise TypeError('meta_model must have method fit(X, y) and predict(X).')

        for base_model in base_models:
            
            if 'fit' not in dir(base_model) or 'predict' not in dir(base_model):
                raise TypeError('base models must have method fit(X, y) and predict(X).')

            if task == 'classification' and mode == 'score' and 'predict_proba' not in dir(base_model):
                raise TypeError('If classification task and score mode then base models must have method predict_proba(X).')

        if not combiner:
            self.combiner = statistics.mode if task == 'classification' else np.mean
            warnings.warn('You did not pass a combiner function, then it will use a standard one for the task.')
        else:
            self.combiner = combiner

        return True

    def fit(self, X, y, cv=10):

        '''
            First, it creates meta model's tranning set using a cross-validation method.
            Then, after targets are checked, it trains both levels - base models and meta model(s).
        '''

        X_meta_models, y_meta_models = self.__cross_validation(X, y, cv=cv)

        y_meta_models = self.__check_targets(y_meta_models)

        self.__fit_both_levels((X, y), (X_meta_models, y_meta_models))

    def predict(self, X):

        '''
             The meta model predicts for each instance which base models are going to be selected.
             Then, it combines selected base models' predictions.

             Returns a label for each istance, using a combiner function.
        '''

        predictions = np.zeros(len(X))
        for idx, x in enumerate(X):

            selected_base_models = self.__predict_meta_models(x)

            if not np.any(selected_base_models): # if none base model was selected, then select all of them (bagging method)
                selected_base_models[:] = 1
            
            final_prediction = self.combiner(self.__predict_base_models(x, selected_base_models))
            predictions[idx] = final_prediction

        return predictions

    def __fit_both_levels(self, X_y_base_models: tuple, X_y_meta_models: tuple):

        '''
            It fits base models and meta models. 
        '''

        X_base_models, y_base_models = X_y_base_models
        self.__fit_base_models(X_base_models, y_base_models)

        X_meta_models, y_meta_models = X_y_meta_models
        self.__fit_meta_models(X_meta_models, y_meta_models)

    def __fit_base_models(self, X, y):

        for base_model in self.base_models:
            base_model.fit(X, y)

    def __predict_base_models(self, x, selected_base_models):

        '''
            Given a instance x, and a mask selected_base_models - it contains which base models were
            selected for this instance - returns a list contaning selected base models' predictions.
        '''

        predictions = []
        for idx, base_model in enumerate(self.base_models):
            if selected_base_models[idx] == 1:
                predictions.append(base_model.predict([x]).ravel()[0])

        return predictions

    def __fit_meta_models(self, X, y):

        if self.n_meta_models == 1: # if for each instance there is only one possible class - base model chosen
            self.meta_models.fit(X, np.argmax(y, axis=1))

        else: # if it is a multi-label classification task
            for idx, meta_model in enumerate(self.meta_models):
                meta_model.fit(X, y[:, idx])

    def __predict_meta_models(self, x):

        if self.n_meta_models == 1:
            predictions = np.zeros(len(self.base_models))
            predictions[self.meta_models.predict([x])] = 1

        else:
            predictions = np.zeros(len(self.meta_models))
            for idx, meta_model in enumerate(self.meta_models):
                predictions[idx] = meta_model.predict([x])

        return predictions

    def __cross_validation(self, X, y, cv):

        '''
            Cross-validation for each base model given a cv. It has three possible flows:

            1) binary mode: Only for classification task. It checks if the base models labeled correctly or not
            every instance. It is actually creating a trainning set for the meta model. 
            
            2) classification score mode:

            3) regression (only works with score mode):

            returns X (same as input) and y target for meta model.
        '''

        self.base_models_predictions = {}
        for idx, base_model in enumerate(self.base_models):
            self.base_models_predictions[idx] = cross_val_predict(base_model, X, y, cv=cv, method=self.__adapt_method())

        if self.mode == 'binary':

            y_target_meta_models = np.zeros((y.shape[0], len(self.base_models)))
            for idx, base_model in enumerate(self.base_models):
                y_target_meta_models[:, idx] = (self.base_models_predictions[idx] == y).astype(int) 

            return X, y_target_meta_models

        elif self.task == 'classification' and self.mode == 'score':

            lb = LabelBinarizer()
    
            y_error_meta_models = np.zeros((y.shape[0], len(self.base_models)))
            for idx, base_model in enumerate(self.base_models):
                y_error_meta_models[:, idx] = self.error_measure(self.base_models_predictions[idx], lb.fit_transform(y))

            return X, self.__chooser(y_error_meta_models)

        else:
            y_error_meta_models = np.zeros((self.base_models_predictions[0].shape[0], len(self.base_models)))
            for idx, base_model in enumerate(self.base_models):
                y_error_meta_models[:, idx] = self.error_measure(self.base_models_predictions[idx], y[-y_error_meta_models.shape[0]:])

            return X[-y_error_meta_models.shape[0]:],  self.__chooser(y_error_meta_models)

    def __adapt_method(self):

        if self.task == 'regression' or (self.task == 'classification' and self.mode == 'binary'):
            return 'predict'
        else:
            return 'predict_proba'

    def __combiner(self, targets):
        
        if self.task == 'classification':
            return mode(targets)
        else:
            return np.mean(targets)

    def __check_targets(self, y_meta_models):

        '''
            It checks if there is any base model that was not selected for any instance.
            Those base models can be removed, because they are not going to be useful on prediction.
            

            Also, this function infers if the meta model task will be a multi-label or not 
            (If more than one base model were selected for any instance then it will be a multi-label task).
            The previous check ensure that this can be done.

            Returns treated y_meta_models.
        '''
        
        # sum over rows to return how many times each base model were selected
        sum_up = np.sum(y_meta_models, axis=0)

        for i in range(sum_up.shape[0]):

            if sum_up[i] == 0: # base model at index i were never selected
                self.base_models.remove(self.base_models[i])


        treated_y_meta_models = y_meta_models[:, sum_up != 0]
        
        # check if there is any instance that has more than one base model assigned to it.
        if np.any(np.sum(treated_y_meta_models, axis=1) > 1):
            self.meta_models = [self.meta_models for _ in range(len(self.base_models))]
            self.n_meta_models = len(self.base_models)

        else:
            self.n_meta_models = 1

        return treated_y_meta_models

    def __chooser(self, y_error_meta_models):

        '''
            Apply the chooser function for each error array.

            Returns the meta model target (label) for each instance.
        '''

        target_meta_models = np.zeros(y_error_meta_models.shape)
        for i in range(y_error_meta_models.shape[0]):
            target_meta_models[i] = self.chooser(y_error_meta_models[i])

        return target_meta_models