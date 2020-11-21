import time 

class NaiveEnsemble(object):

    def __init__(self, models):

        self.models = models

    def fit(self, X, y):

        self.fit_time = {i:time.time() for i in range(len(self.models))}
        for idx, model in enumerate(self.models):
            model.fit(X, y)
            self.fit_time[idx] = time.time() - self.fit_time[idx]
    
    def predict(self, X):

        predictions = np.zeros((X.shape[0], len(self.models)))
        self.prediction_time = time.time()
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(X)
        self.prediction_time = time.time() - self.prediction_time 