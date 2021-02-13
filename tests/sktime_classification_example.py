# standard libs
import numpy as np
import pandas as pd

# meta learning framework classes and utils
from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.meta_classifier import MetaClassifier
from meta_learning_framework.meta_learning_model import MetaLearningModel
from meta_learning_framework.naive_ensemble import NaiveEnsemble
from meta_learning_framework.utils import proba_mean_error

# sktime utils and classifiers
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

# sklearn KNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

# metrics
from sklearn.metrics import classification_report, accuracy_score

# util function to implement DTW metric
from scipy.spatial import distance

# tensorflow to build a meta classifier
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.math as M

# to read dataset and mode from command line
import sys

# Base Classifiers


class TSKNN_ED(BaseModel):

    """
    KNeighborsClassifier using Euclidean Distance.
    """

    def __init__(self):

        self.model = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self.name = "TSKNN_ED"
        self.classes_ = (
            []
        )  # this attribute is necessary if you are going to use classification+score

    def fit(self, X, y):

        # filling this attribute (classification+score)
        self.classes_ = [c for c in set(y)]

        self.model.fit(
            np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y]
        )

    def predict(self, X):
        return self.model.predict([s[0].values for s in X.values.tolist()])

    def predict_one(self, x):
        return int(self.model.predict(x[0].values.reshape(1, -1))[0])

    def predict_proba(self, X):
        return self.model.predict_proba([s[0].values for s in X.values.tolist()])

    def predict_proba_one(self, x):
        return self.model.predict_proba(x[0].values.reshape(1, -1))


class TSKNN_DTW(BaseModel):

    """
    KNeighborsClassifier using Distance Time Warping (DTW).
    """

    def __init__(self):

        # DTW function between two time series a and b
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

        # define metric arugment as DTW
        self.model = KNeighborsClassifier(n_neighbors=1, metric=DTW, n_jobs=-1)
        self.name = "TSKNN_DTW"
        self.classes_ = (
            []
        )  # this attribute is necessary if you are going to use classification+score

    def fit(self, X, y):

        # filling this attribute (classification+score)
        self.classes_ = [c for c in set(y)]

        self.model.fit(
            np.array([s[0].values for s in X.values.tolist()]), [int(i) for i in y]
        )

    def predict(self, X):
        return self.model.predict([s[0].values for s in X.values.tolist()])

    def predict_one(self, x):
        return int(self.model.predict(x[0].values.reshape(1, -1))[0])

    def predict_proba(self, X):
        return self.model.predict_proba([s[0].values for s in X.values.tolist()])

    def predict_proba_one(self, x):
        return self.model.predict_proba(x[0].values.reshape(1, -1))


class LocalClassifier(BaseModel):

    """
    Class to encapsule sktime classifiers.
    """

    def __init__(self, model, name: str):
        super().__init__(model, name)

    def fit(self, X, y):

        # filling this attribute (classification+score)
        self.classes_ = [c for c in set(y)]

        self.model.fit(X, y)

    def predict(self, X):
        return np.array([int(pred) for pred in self.model.predict(X)])

    def predict_one(self, x):
        return int(self.model.predict(pd.DataFrame({"dim_0": pd.Series(x)}))[0])

    def predict_proba(self, X):
        return np.array([pred for pred in self.model.predict_proba(X)])

    def predict_proba_one(self, x):
        return self.model.predict_proba(pd.DataFrame({"dim_0": pd.Series(x)}))[0]


# Meta Classifier


class NeuralNetworkMetaClassifier(MetaClassifier):

    """
    Neural network based classifier. It supports a multi-label clasification task.
    """

    def __init__(self, in_shape, lstm_cells, batch_size=4, epochs=20):

        """
        Note the model itself is initialized only on fit method.
        """

        self.in_shape = in_shape
        self.lstm_cells = lstm_cells
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):

        inputs = Input(
            shape=(
                1,
                self.in_shape,
            )
        )
        lstm = LSTM(self.lstm_cells)(inputs)
        out = Dense(y.shape[1], activation="softmax")(lstm)

        self.meta_clf = Model(inputs=inputs, outputs=out)

        def custom_loss(y_true, y_pred):

            abs_diff = K.abs(y_true - y_pred)
            mult = M.multiply(abs_diff, y_pred)
            custom_loss = abs_diff + mult
            return custom_loss

        self.meta_clf.compile(
            optimizer="rmsprop", loss=custom_loss, metrics=["accuracy"]
        )

        X_array = np.array(X["dim_0"].apply(lambda x: x.values).tolist())

        self.meta_clf.fit(
            X_array.reshape(X_array.shape[0], 1, X_array.shape[1]),
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def predict(self, X):

        pred = self.meta_clf.predict(X[0].values.reshape(1, 1, -1))[0].ravel()

        # using 0.5 as threshold
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        return pred

    def predict_one(self, x):

        x = x[0].values
        pred = self.meta_clf.predict(x.reshape(1, 1, x.shape[0]))[0]

        # using 0.5 as threshold
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        return pred


if __name__ == "__main__":

    # read dataset and mode from command line
    dataset_name = sys.argv[1]
    mode = sys.argv[2]

    # train and test sets
    X_train, y_train = load_from_tsfile_to_dataframe(
        "Univariate_ts/" + dataset_name + "/" + dataset_name + "_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
        "Univariate_ts/" + dataset_name + "/" + dataset_name + "_TEST.ts"
    )

    # str label to int label
    y_train = np.array([int(y) for y in y_train])
    y_test = np.array([int(y) for y in y_test])

    # list of base classifiers
    bm_list = [
        LocalClassifier(IndividualBOSS(random_state=11), "IndividualBOSS"),
        LocalClassifier(BOSSEnsemble(random_state=11), "BOSSEnsemble"),
        LocalClassifier(MUSE(random_state=11), "MUSE"),
        LocalClassifier(
            TemporalDictionaryEnsemble(random_state=11), "TemporalDictionaryEnsemble"
        ),
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(n_jobs=-1, random_state=11), "WEASEL"),
        LocalClassifier(ProximityForest(n_jobs=-1, random_state=11), "ProximityForest"),
        LocalClassifier(ProximityTree(n_jobs=-1, random_state=11), "ProximityTree"),
        LocalClassifier(
            RandomIntervalSpectralForest(n_jobs=-1, random_state=11),
            "RandomIntervalSpectralForest",
        ),
        LocalClassifier(
            TimeSeriesForest(n_jobs=-1, random_state=11), "TimeSeriesForest"
        ),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # meta model initialization
    input_shape = X_train.values[0][0].shape[0]
    MetaModel = NeuralNetworkMetaClassifier(input_shape, 16)

    # meta learning framework initialization
    mm_framework = MetaLearningModel(
        MetaModel,
        bm_list,
        "classification",
        mode,
        multi_label=True,
    )

    # fit and predict methods
    mm_framework.fit(X_train, y_train, cv=0.4, dynamic_shrink=True)
    meta_preds = mm_framework.predict(X_test.values)

    # metrics
    print("Meta model report:")
    print(classification_report(y_test, meta_preds))

    # save time metrics and number of base classifiers used
    mm_framework.save_time_metrics_csv(
        "Univariate_ts/" + dataset_name + "/MetaModel_time_metrics.csv"
    )
    mm_framework.save_base_models_used(
        "Univariate_ts/" + dataset_name + "/MetaModel_base_models_used.npy"
    )

    # naive ensemble for comparison

    # reinitialize list of base classifiers
    bm_list = [
        LocalClassifier(IndividualBOSS(random_state=11), "IndividualBOSS"),
        LocalClassifier(BOSSEnsemble(random_state=11), "BOSSEnsemble"),
        LocalClassifier(MUSE(random_state=11), "MUSE"),
        LocalClassifier(
            TemporalDictionaryEnsemble(random_state=11), "TemporalDictionaryEnsemble"
        ),
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(n_jobs=-1, random_state=11), "WEASEL"),
        LocalClassifier(ProximityForest(n_jobs=-1, random_state=11), "ProximityForest"),
        LocalClassifier(ProximityTree(n_jobs=-1, random_state=11), "ProximityTree"),
        LocalClassifier(
            RandomIntervalSpectralForest(n_jobs=-1, random_state=11),
            "RandomIntervalSpectralForest",
        ),
        LocalClassifier(
            TimeSeriesForest(n_jobs=-1, random_state=11), "TimeSeriesForest"
        ),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # naive ensemble object
    ne = NaiveEnsemble(bm_list, "classification")

    # fit and predict methods
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)

    # metrics
    print("Naive ensemble report:")
    print(classification_report(y_test, ne_preds))

    ne.save_time_metrics_csv(
        "Univariate_ts/" + dataset_name + "/NaiveEnsemble_time_metrics.csv"
    )

    # evaluate base models individual performance
    print("individual performance report:")
    individual_preds = ne.individual_predict(X_test)

    for model_name in individual_preds.keys():
        print(
            model_name
            + " accuracy: "
            + str(accuracy_score(y_test, individual_preds[model_name]))
        )
