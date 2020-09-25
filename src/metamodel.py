from sklearn.model_selection import cross_val_predict
from statistics import mode
import numpy as np

class MetaLearningModel(object):

    def __init__(self, metamodel, base_models, task, mode):

        if self.check_args(metamodel, base_models, task, mode):
            self.metamodels = [metamodel for _ in range(len(base_models))]
            self.base_models = base_models
            self.task = task
            self.mode = mode

    def check_args(self, metamodel, base_models, task, mode):

        if task not in ['classification', 'regression']:
            raise ValueError('Must choose a task: classification or regression.')

        if task == 'regression' and mode == 'binary':
            mode = 'score'
            warnings.warn('Regression tasks only support score mode. Using score instead of binary mode.')

        if 'fit' not in dir(metamodel) or 'predict' not in dir(metamodel):
            raise TypeError('metamodel must have method fit(X, y) and predict(X).')

        for base_model in base_models:
            if 'fit' not in dir(base_model) or 'predict' not in dir(base_model):
                raise TypeError('base models must have method fit(X, y) and predict(X).')

        return True

    def fit_both(self, X_y_base_models, X_y_meta_model):

        X_base_models, y_base_models = X_y_base_models
        self.__fit_base_models(X_base_models, y_base_models)

        X_metamodel, y_metamodel = X_y_meta_model
        self.__fit_metamodels(X_metamodel, y_metamodel)

    def __fit_base_models(self, X, y):

        for base_model in self.base_models:
            base_model.fit(X, y)

    def __predict_base_models(self, X, selected_base_models):

        predictions = []
        for idx, base_model in enumerate(self.base_models):
            if selected_base_models[idx] == 1:
                predictions.append(base_model.predict(X).ravel()[0])

        return predictions

    def __fit_metamodels(self, X, y):

        for idx, metamodel in enumerate(self.metamodels):
            metamodel.fit(X, y[:, idx])

    def __predict_metamodels(self, X):

        predictions = np.zeros((len(X), len(self.metamodels)))
        for idx, metamodel in enumerate(self.metamodels):
            predictions[:, idx] = metamodel.predict(X)

        return predictions.ravel().tolist()

    def fit(self, X, y, n_folds):

        X_metamodels, y_metamodels = self.__cross_validation(X, y, n_folds=10)
        self.fit_both((X, y), (X_metamodels, y_metamodels))

    def predict(self, X):

        predictions = []
        for x in X:
            selected_base_models = self.__predict_metamodels([x])
            if not np.any(selected_base_models):
                selected_base_models[0] = 1
            # print(self.__predict_base_models([x], selected_base_models))
            final_prediction = self.__combiner(self.__predict_base_models([x], selected_base_models))
            predictions.append(final_prediction)

        return predictions

    def __cross_validation(self, X, y, n_folds):

        base_models_predictions = {}
        for idx, base_model in enumerate(self.base_models):
            base_models_predictions[idx] = cross_val_predict(base_model, X, y, cv=n_folds, method='predict_proba')

        if self.mode == 'binary':

            y_target_metamodels = np.zeros((y.shape[0], len(self.base_models)))
            for idx, base_model in enumerate(self.base_models):
                y_target_metamodels[:, idx] = (np.argmax(base_models_predictions[idx], axis=1) == y).astype(int)

            return X, y_target_metamodels

    def __adapt_method(self):

        if self.task == 'regression' or (self.task == 'classification' and self.mode == 'binary'):
            return 'predict'

        else:
            return 'predict_proba'

    def __combiner(self, labels):

        return mode(labels)
