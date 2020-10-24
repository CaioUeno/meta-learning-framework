
class NaiveEnsemble(object):

    def __init__(self, models):

        self.models = models

    def fit(self, X):

        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):

        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(X)