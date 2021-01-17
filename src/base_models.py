import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

def DTW(a, b):
    an = a.size
    bn = b.size
    w = np.ceil(an * 0.1)
    pointwise_distance = distance.cdist(a.reshape(-1,1), b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1, bn+1)) * np.inf)
    cumdist[0,0] = 0
    for ai in range(an):
        beg_win = int(np.max([0, ai-w]))
        end_win = int(np.min([ai+w, an]))
        for bi in range(beg_win, end_win):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    return cumdist[an, bn]

class TSKNN_ED(BaseClassifier):
    
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        self.name = 'TSKNN_ED'
    
    def fit(self, X, y):
        self.model.fit(np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y])

    def predict(self, X):
        return [str(i) for i in self.model.predict([s[0].values for s in X.values.tolist()])]

    def predict_one(self, x):
        return self.model.predict(x[0].values.reshape(1, -1))

class TSKNN_DTW(BaseClassifier):
    
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1, metric=DTW)
        self.name = 'TSKNN_DTW'
    
    def fit(self, X, y):
        self.model.fit(np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y])

    def predict(self, X):
        return [str(i) for i in self.model.predict([s[0].values for s in X.values.tolist()])]

    def predict_one(self, x):
        return self.model.predict(x[0].values.reshape(1, -1))

class LocalClassifierII(BaseClassifier):
    
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_one(self, x):
        return self.model.predict(pd.DataFrame({'dim_0':pd.Series(x)}))