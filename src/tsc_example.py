from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.classification.dictionary_based import (
    IndividualBOSS,
    BOSSEnsemble,
    MUSE,
    TemporalDictionaryEnsemble,
    IndividualTDE,
    WEASEL,
)

# from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.distance_based import ProximityForest, ProximityTree

# from sktime.classification.distance_based import ProximityStump
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier

from base_models import TSKNN_ED, TSKNN_DTW, LocalClassifierII
from meta_nn import NeuralNetworkMetaClassifier
from metamodel import MetaLearningModel
from naive_ensemble import NaiveEnsemble
import os
import sys


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
    y_test = [int(y) for y in y_test]

    # list of classifiers to be used
    bm_list = [
        LocalClassifierII(IndividualBOSS(), "IndividualBOSS"),
        LocalClassifierII(BOSSEnsemble(), "BOSSEnsemble"),
        LocalClassifierII(MUSE(), "MUSE"),
        LocalClassifierII(TemporalDictionaryEnsemble(), "TemporalDictionaryEnsemble"),
        LocalClassifierII(IndividualTDE(), "IndividualTDE"),
        LocalClassifierII(WEASEL(), "WEASEL"),
        LocalClassifierII(ProximityForest(), "ProximityForest"),
        LocalClassifierII(ProximityTree(), "ProximityTree"),
        LocalClassifierII(
            RandomIntervalSpectralForest(), "RandomIntervalSpectralForest"
        ),
        LocalClassifierII(TimeSeriesForest(), "TimeSeriesForest"),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # meta model
    input_shape = X_train.values[0][0].shape[0]

    mm = MetaLearningModel(
        NeuralNetworkMetaClassifier(input_shape, len(bm_list), 16),
        bm_list,
        "classification",
        "binary",
        multi_class=True,
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

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------

    # naive ensemble for comparison

    # reinitialize base models
    bm_list = [
        LocalClassifierII(IndividualBOSS(), "IndividualBOSS"),
        LocalClassifierII(BOSSEnsemble(), "BOSSEnsemble"),
        LocalClassifierII(MUSE(), "MUSE"),
        LocalClassifierII(TemporalDictionaryEnsemble(), "TemporalDictionaryEnsemble"),
        LocalClassifierII(IndividualTDE(), "IndividualTDE"),
        LocalClassifierII(WEASEL(), "WEASEL"),
        LocalClassifierII(ProximityForest(), "ProximityForest"),
        LocalClassifierII(ProximityTree(), "ProximityTree"),
        LocalClassifierII(
            RandomIntervalSpectralForest(), "RandomIntervalSpectralForest"
        ),
        LocalClassifierII(TimeSeriesForest(), "TimeSeriesForest"),
        TSKNN_DTW(),
        TSKNN_ED(),
    ]

    # naive ensemble
    ne = NaiveEnsemble(bm_list)

    # fit andpredcit methods
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
