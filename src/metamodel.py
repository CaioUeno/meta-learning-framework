class MetaLearningModel(object):

    def __init__(self, metamodel, base_models, task, mode):

        if self.check_args(metamodel, base_models, task, mode):
            self.metamodels = [metamodel for _ in range(len(base_models))]
            self.base_models = base_models
            self.task = task
            self.mode = mode

    def check_args(self, metamodel, base_models, task, mode):

        if task != 'classification' or task != 'regression':
            raise ValueError('Must choose a task: classification or regression.')

        if task == 'regression' and mode == 'binary':
            mode = 'score'
            warnings.warn('Regression tasks only support score mode. Using score instead of binary mode.')

        if 'fit' not in dir(metamodel) or 'predict' not in dir(metamodel):
            raise TypeError('metamodel must have method fit(X, y) and predict(X).')

        for base_model in base_models:
            if 'fit' not in dir(base_models) or 'predict' not in dir(base_models):
                raise TypeError('base models must have method fit(X, y) and predict(X).')

        return True

    def fit(self, X_y_base_models, X_y_meta_model):

        X_base_models, y_base_models = X_y_base_models
        for base_model in base_models:
            base_model.fit(X_base_models, y_base_models)

        X_metamodel, y_metamodel = X_y_meta_model
        y_metamodel = self.prepare_meta_target(y_metamodel)
        for idx, metamodel in enumerate(self.metamodels):
            metamodel.fit(X_metamodel, y_metamodel[idx])

    def __fit_base_models(self, X_y_base_models):

        X_base_models, y_base_models = X_y_base_models
        for base_model in base_models:
            base_model.fit(X_base_models, y_base_models)


    # def predict(self, X):
