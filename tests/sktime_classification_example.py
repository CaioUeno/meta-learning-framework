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
    IndividualTDE,
    WEASEL,
)
from sktime.classification.interval_based import RandomIntervalSpectralForest

# sklearn KNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

# metrics
from sklearn.metrics import classification_report, accuracy_score

# util function to implement DTW metric
# from scipy.spatial import distance
from dtw import dtw

# tensorflow to build a meta classifier
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.math as M
from tensorflow import one_hot

# to read dataset and mode from command line
import sys

# Base Classifiers


class NNEuclideanDistance(BaseModel):

    """
    KNeighborsClassifier using Euclidean Distance.
    """

    def __init__(self, name: str):

        super().__init__(KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=5), name)

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


class NNDTW(BaseModel):

    """
    KNeighborsClassifier using Distance Time Warping (DTW).
    """

    def __init__(self, name: str):

        # DTW function between two time series a and b
        def DTW(a, b):
            return dtw(a, b, distance_only=True).distance
        
        super().__init__(KNeighborsClassifier(n_neighbors=1, metric=DTW, n_jobs=5), name)

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


# Meta Classifiers


class NeuralNetworkMetaClassifierBinary(MetaClassifier):

    """
    Neural network based classifier. It supports a multi-label clasification task.
    """

    def __init__(self, in_shape, lstm_cells, threshold=.5, batch_size=4, epochs=10):

        """
        Note the model itself is initialized only on fit method.
        """

        self.in_shape = in_shape
        self.lstm_cells = lstm_cells
        self.threshold = threshold
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):

        inputs = Input(
            shape=(
                1,
                self.in_shape,
            )
        )
        lstm_layer = LSTM(self.lstm_cells)(inputs)
        out = Dense(y.shape[1], kernel_regularizer="l1", activation="sigmoid")(lstm_layer)

        self.meta_clf = Model(inputs=inputs, outputs=out)

        def custom_loss(y_true, y_pred):

            abs_diff = K.abs(y_true - y_pred)
            mult = M.multiply(abs_diff, y_pred)
            custom_loss = abs_diff + mult

            return custom_loss

        self.meta_clf.compile(
            optimizer="rmsprop", loss=custom_loss
        )

        X_array = np.array(X["dim_0"].apply(lambda x: x.values).tolist())

        self.meta_clf.fit(
            X_array.reshape(X_array.shape[0], 1, X_array.shape[1]),
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def predict(self, X):

        pred = self.meta_clf.predict(X[0].values.reshape(1, 1, -1))

        pred[pred >= self.threshold] = 1
        pred[pred < self.threshold] = 0

        return pred

    def predict_one(self, x):

        x = x[0].values
        pred = self.meta_clf.predict(x.reshape(1, 1, x.shape[0]))[0]

        pred[pred >= self.threshold] = 1
        pred[pred < self.threshold] = 0

        return pred

class NeuralNetworkMetaClassifierScore(MetaClassifier):

    """
    Neural network based classifier. It supports only a multi-class clasification task.
    """

    def __init__(self, in_shape, lstm_cells, batch_size=4, epochs=10):

        """
        Note the model itself is initialized only on fit method.
        """

        self.in_shape = in_shape
        self.lstm_cells = lstm_cells
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):

        y = one_hot(y, depth=len(set(y)))

        inputs = Input(
            shape=(
                1,
                self.in_shape,
            )
        )
        lstm_layer = LSTM(self.lstm_cells, recurrent_regularizer="l1")(inputs)
        out = Dense(y.shape[1], kernel_regularizer="l1", activation="softmax")(lstm_layer)

        self.meta_clf = Model(inputs=inputs, outputs=out)

        self.meta_clf.compile(
            optimizer="rmsprop", loss="binary_crossentropy"
        )

        X_array = np.array(X["dim_0"].apply(lambda x: x.values).tolist())

        self.meta_clf.fit(
            X_array.reshape(X_array.shape[0], 1, X_array.shape[1]),
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def predict(self, X):

        pred = self.meta_clf.predict(X[0].values.reshape(1, 1, -1))

        return pred.argmax(axis=1)

    def predict_one(self, x):

        x = x[0].values
        pred = self.meta_clf.predict(x.reshape(1, 1, x.shape[0]))[0]

        return pred.argmax()


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
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(n_jobs=5, random_state=11), "WEASEL"),
        LocalClassifier(
            RandomIntervalSpectralForest(n_jobs=5, random_state=11),
            "RandomIntervalSpectralForest",
        ),
        NNDTW("DTW"),
        NNEuclideanDistance("EuclideanDistance"),
    ]

    # meta model initialization
    input_shape = X_train.values[0][0].shape[0]

    if mode == "binary":
        MetaModel = NeuralNetworkMetaClassifierBinary(input_shape, 4, .75)

    else:
        MetaModel = NeuralNetworkMetaClassifierScore(input_shape, 4)

    # meta learning framework initialization
    mm_framework = MetaLearningModel(
        MetaModel,
        bm_list,
        "classification",
        mode,
        multi_label=True if mode == "binary" else False,
    )

    # fit and predict methods
    mm_framework.fit(X_train, y_train, cv=.3, dynamic_shrink=True)
    meta_preds = mm_framework.predict(X_test.values)

    # metrics
    print("Meta model report:")
    print(classification_report(y_test, meta_preds))
    if mode == "binary":
        print(f"Mean base models used: {mm_framework.prediction_base_models_used.mean():.2f}")

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
        LocalClassifier(IndividualTDE(random_state=11), "IndividualTDE"),
        LocalClassifier(WEASEL(n_jobs=5, random_state=11), "WEASEL"),
        LocalClassifier(
            RandomIntervalSpectralForest(n_jobs=5, random_state=11),
            "RandomIntervalSpectralForest",
        ),
       NNDTW("DTW"),
        NNEuclideanDistance("EuclideanDistance"),
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
