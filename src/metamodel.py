from sklearn.model_selection import cross_val_predict
from statistics import mode
import numpy as np

class MetaLearningModel(object):

    def __init__(self, meta_model, base_models, task, mode):

        if self.check_args(meta_model, base_models, task, mode):
            self.meta_models = [meta_model for _ in range(len(base_models))]
            self.base_models = base_models
            self.task = task
            self.mode = mode

    def check_args(self, meta_model, base_models, task, mode):

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

        return True

    def fit(self, X, y, n_folds):

        X_meta_models, y_meta_models = self.__cross_validation(X, y, n_folds=10)
        self.fit_both_levels((X, y), (X_meta_models, y_meta_models))

    def predict(self, X):

        predictions = []
        for x in X:
            selected_base_models = self.__predict_meta_models([x])
            if not np.any(selected_base_models):
                selected_base_models[0] = 1
            # print(self.__predict_base_models([x], selected_base_models))
            final_prediction = self.__combiner(self.__predict_base_models([x], selected_base_models))
            predictions.append(final_prediction)

        return predictions

    def fit_both_levels(self, X_y_base_models, X_y_meta_models):

        X_base_models, y_base_models = X_y_base_models
        self.__fit_base_models(X_base_models, y_base_models)

        X_meta_models, y_meta_models = X_y_meta_models
        self.__fit_meta_models(X_meta_models, y_meta_models)

    def __fit_base_models(self, X, y):

        for base_model in self.base_models:
            base_model.fit(X, y)

    def __predict_base_models(self, X, selected_base_models):

        predictions = []
        for idx, base_model in enumerate(self.base_models):
            if selected_base_models[idx] == 1:
                predictions.append(base_model.predict(X).ravel()[0])

        return predictions

    def __fit_meta_models(self, X, y):

        for idx, meta_model in enumerate(self.meta_models):
            meta_model.fit(X, y[:, idx])

    def __predict_meta_models(self, X):

        predictions = np.zeros((len(X), len(self.meta_models)))
        for idx, meta_model in enumerate(self.meta_models):
            predictions[:, idx] = meta_model.predict(X)

        return predictions.ravel().tolist()

    def __cross_validation(self, X, y, n_folds):

        base_models_predictions = {}
        for idx, base_model in enumerate(self.base_models):
            base_models_predictions[idx] = cross_val_predict(base_model, X, y, cv=n_folds, method=self.__adapt_method())

        if self.mode == 'binary':

            y_target_meta_models = np.zeros((y.shape[0], len(self.base_models)))
            for idx, base_model in enumerate(self.base_models):
                y_target_meta_models[:, idx] = (base_models_predictions[idx] == y).astype(int)

            return X, y_target_meta_models

        else:
            y_target_meta_models = np.zeros((y.shape[0], len(self.base_models)))
            # classes = base_model.classes_.tolist()
            for idx, base_model in enumerate(self.base_models):
                y_target_meta_models[:, idx] = base_models_predictions[idx][[i for i in range(base_models_predictions[idx].shape[0])], y.tolist()]

            return X, np.argmax(y_target_meta_models, axis=1)


    def __adapt_method(self):

        if self.task == 'regression' or (self.task == 'classification' and self.mode == 'binary'):
            return 'predict'
        else:
            return 'predict_proba'

    def __combiner(self, labels):
        return mode(labels)
