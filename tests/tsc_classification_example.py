from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.classification.dictionary_based import (
    IndividualBOSS,
    BOSSEnsemble,
    MUSE,
    TemporalDictionaryEnsemble,
    IndividualTDE,
    WEASEL,
)
from sktime.classification.distance_based import ProximityForest, ProximityTree
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier

from base_models import BaseModel
from meta_nn import MetaClassifier
from metamodel import MetaLearningModel
from naive_ensemble import NaiveEnsemble
import os
import sys

import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.math as M

# Base Classifiers

class TSKNN_ED(BaseModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        self.name = "TSKNN_ED"

    def fit(self, X, y):
        self.model.fit(
            np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y]
        )

    def predict(self, X):
        return [
            int(i) for i in self.model.predict([s[0].values for s in X.values.tolist()])
        ]

    def predict_one(self, x):
        return int(self.model.predict(x[0].values.reshape(1, -1))[0])

class TSKNN_DTW(BaseModel):
    def __init__(self):

        def DTW(a, b):
            an = a.size
            bn = b.size
            w = np.ceil(an * 0.1)
            pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
            cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
            cumdist[0, 0] = 0
            for ai in range(an):
                beg_win = int(np.max([0, ai - w]))
                end_win = int(np.min([ai + w, an]))
                for bi in range(beg_win, end_win):
                    minimum_cost = np.min(
                        [cumdist[ai, bi + 1], cumdist[ai + 1, bi], cumdist[ai, bi]]
                    )
                    cumdist[ai + 1, bi + 1] = pointwise_distance[ai, bi] + minimum_cost
            return cumdist[an, bn]
        
        self.model = KNeighborsClassifier(n_neighbors=1, metric=DTW)
        self.name = "TSKNN_DTW"

    def fit(self, X, y):
        self.model.fit(
            np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y]
        )

    def predict(self, X):
        return [
            int(i) for i in self.model.predict([s[0].values for s in X.values.tolist()])
        ]

    def predict_one(self, x):
        return int(self.model.predict(x[0].values.reshape(1, -1))[0])

class LocalClassifier(BaseModel):
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return np.array([int(pred) for pred in self.model.predict(X)])

    def predict_one(self, x):
        return int(self.model.predict(pd.DataFrame({"dim_0": pd.Series(x)}))[0])

# Meta Classifier

class NeuralNetworkMetaClassifier(MetaClassifier):
    
    def __init__(self, in_shape, out_shape, lstm_cells, batch_size=4, epochs=10):

        inputs = Input(shape=(1, in_shape,))
        lstm = LSTM(lstm_cells)(inputs)
        out = Dense(out_shape, activation='sigmoid')(lstm)

        self.meta_clf = Model(inputs=inputs, outputs=out)

        def custom_loss(y_true, y_pred):

            abs_diff = K.abs(y_true - y_pred)
            mult = M.multiply(abs_diff, y_pred)
            custom_loss = abs_diff + mult
            return custom_loss
        
        self.meta_clf.compile(optimizer='rmsprop', loss=custom_loss)

        self.batch_size = batch_size
        self.epochs = epochs
    
    def fit(self, X, y):
        
        X_array = np.array(X['dim_0'].apply(lambda x: x.values).tolist())
        self.meta_clf.fit(X_array.reshape(X_array.shape[0], 1, X_array.shape[1]), y,
                         batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, X):

        pred = self.meta_clf.predict(X[0].values.reshape(1, 1, -1))[0].ravel()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def predict_one(self, x):

        x = x[0].values
        pred = self.meta_clf.predict(x.reshape(1, 1, x.shape[0]))[0]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

if __name__ == "__main__":

    # select dataset
    dataset_name = sys.argv[1]

    # train and test sets
    X_train, y_train = load_from_tsfile_to_dataframe(
        "../Univariate_ts/" + dataset_name + "/" + dataset_name + "_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
        "../Univariate_ts/" + dataset_name + "/" + dataset_name + "_TRAIN.ts"
    )

    # str label to int label
    y_train = np.array([int(y) for y in y_train])
    y_test = np.array([int(y) for y in y_test])

    # list of base classifiers
    bm_list = [
        LocalClassifier(IndividualBOSS(random_state=11), "IndividualBOSS"),
        LocalClassifier(BOSSEnsemble(random_state=11), "BOSSEnsemble"),
        LocalClassifier(MUSE(random_state=11), "MUSE"),
        LocalClassifier(TemporalDictionaryEnsemble(random_state=11), "TemporalDictionaryEnsemble"),
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(random_state=11), "WEASEL"),
        LocalClassifier(ProximityForest(random_state=11), "ProximityForest"),
        LocalClassifier(ProximityTree(random_state=11), "ProximityTree"),
        LocalClassifier(RandomIntervalSpectralForest(random_state=11), "RandomIntervalSpectralForest"),
        LocalClassifier(TimeSeriesForest(random_state=11), "TimeSeriesForest"),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # meta model
    input_shape = X_train.values[0][0].shape[0]
    MetaModel = NeuralNetworkMetaClassifier(input_shape, len(bm_list), 16)

    mm = MetaLearningModel(
        MetaModel,
        bm_list,
        "classification",
        "binary",
        multi_label=True,
    )

    # fit and predict methods
    mm.fit(X_train, y_train, cv=5, dynamic_shrink=False)
    meta_preds = mm.predict(X_test.values)

    # saving metrics
    mm.save_performance_metrics(
        "../Univariate_ts/" + dataset_name + "/MetaModel_performance_metrics.csv",
        y_test,
        meta_preds,
    )
    mm.save_time_metrics(
        "../Univariate_ts/" + dataset_name + "/MetaModel_time_metrics.csv"
    )
    mm.save_base_models_used(
        "../Univariate_ts/" + dataset_name + "/MetaModel_base_models_used.npy"
    )

    # ======= #
    # ======= #

    # naive ensemble

    # reinitialize base models
    bm_list = [
        LocalClassifier(IndividualBOSS(random_state=11), "IndividualBOSS"),
        LocalClassifier(BOSSEnsemble(random_state=11), "BOSSEnsemble"),
        LocalClassifier(MUSE(random_state=11), "MUSE"),
        LocalClassifier(TemporalDictionaryEnsemble(random_state=11), "TemporalDictionaryEnsemble"),
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(random_state=11), "WEASEL"),
        LocalClassifier(ProximityForest(random_state=11), "ProximityForest"),
        LocalClassifier(ProximityTree(random_state=11), "ProximityTree"),
        LocalClassifier(
            RandomIntervalSpectralForest(random_state=11), "RandomIntervalSpectralForest"
        ),
        LocalClassifier(TimeSeriesForest(random_state=11), "TimeSeriesForest"),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # naive ensemble object
    ne = NaiveEnsemble(bm_list, "classification")

    # fit and predict methods
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)

    # saving metrics
    ne.save_performance_metrics(
        "../Univariate_ts/" + dataset_name + "/NaiveEnsemble_performance_metrics.csv",
        y_test,
        ne_preds,
    )
    ne.save_time_metrics(
        "../Univariate_ts/" + dataset_name + "/NaiveEnsemble_time_metrics.csv"
    )
