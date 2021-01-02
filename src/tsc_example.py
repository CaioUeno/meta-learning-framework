from sktime.utils.data_io import load_from_tsfile_to_dataframe 
from sktime.classification.dictionary_based import IndividualBOSS, MUSE, IndividualTDE, WEASEL
from sktime.classification.distance_based import ProximityTree
from base_models import LocalClassifier
from meta_nn import NeuralNetworkMetaClassifier
from metamodel import MetaLearningModel
from naive_ensemble import NaiveEnsemble
import os

if __name__ == "__main__":

    # select dataset
    dataset_name = 'CBF'
    # if 'MetaModel_time_metrics.csv' in os.listdir('../Univariate_ts/'+dataset_name+'/'):
    #     print('This dataset is already done!')
    #     exit(0)

    # train and test sets
    X_train, y_train = load_from_tsfile_to_dataframe('../Univariate_ts/'+dataset_name+'/'+dataset_name+'_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe('../Univariate_ts/'+dataset_name+'/'+dataset_name+'_TRAIN.ts')

    y_test = [int(y) for y in y_test]

    # list of classifiers to be used
    bm_list = [LocalClassifier(IndividualBOSS(), 'InidividualBOSS'), LocalClassifier(IndividualTDE(), 'IndividualTDE'),
               LocalClassifier(WEASEL(), 'WAESEL')]

    # meta model 
    mm = MetaLearningModel(NeuralNetworkMetaClassifier(X_train.values[0][0].shape[0], len(bm_list)), bm_list, 'classification', 'binary', multi_class=True)
    mm.fit(X_train, y_train)
    meta_preds = mm.predict(X_test.values)
    mm.save_performance_metrics('../Univariate_ts/'+dataset_name+'/MetaModel_performance_metrics.csv', y_test, meta_preds)
    mm.save_time_metrics('../Univariate_ts/'+dataset_name+'/MetaModel_time_metrics.csv')

    # reinitialize these models
    bm_list = [LocalClassifier(IndividualBOSS(), 'InidividualBOSS'), LocalClassifier(IndividualTDE(), 'IndividualTDE'),
               LocalClassifier(WEASEL(), 'WAESEL')]

    # naive ensemble
    ne = NaiveEnsemble(bm_list)
    ne.fit(X_train, y_train)
    ne_preds = ne.predict(X_test)
    ne.save_performance_metrics('../Univariate_ts/'+dataset_name+'/NaiveEnsemble_performance_metrics.csv', y_test, ne_preds)
    ne.save_time_metrics('../Univariate_ts/'+dataset_name+'/NaiveEnsemble_time_metrics.csv')
    