from sktime.utils.data_io import load_from_tsfile_to_dataframe 
from sktime.classification.dictionary_based import IndividualBOSS, MUSE, IndividualTDE, WEASEL
from sktime.classification.distance_based import ProximityTree
from base_models import LocalClassifier
from meta_nn import NeuralNetworkMetaClassifier
from metamodel import MetaLearningModel
from naive_ensemble import NaiveEnsemble
import os

if __name__ == "__main__":

    dataset_name = 'CBF'
    # if 'MetaModel_time_metrics.csv' in os.listdir('../Univariate_ts/'+dataset_name+'/'):
    #     print('This dataset is already done!')
    #     exit(0)

    X_train, y_train = load_from_tsfile_to_dataframe('../Univariate_ts/'+dataset_name+'/'+dataset_name+'_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe('../Univariate_ts/'+dataset_name+'/'+dataset_name+'_TRAIN.ts')

    bm_list = [LocalClassifier(IndividualBOSS(), 'InidividualBOSS'), LocalClassifier(IndividualTDE(), 'IndividualTDE'),
               LocalClassifier(WEASEL(), 'WAESEL')]

    mm = MetaLearningModel(NeuralNetworkMetaClassifier(128, len(bm_list)), bm_list, 'classification', 'binary', multi_class=True)
    mm.fit(X_train, y_train)
    print(mm.predict(X_test.values))
    
    mm.save_time_metrics('../Univariate_ts/'+dataset_name+'/MetaModel_time_metrics.csv')

    bm_list = [LocalClassifier(IndividualBOSS(), 'InidividualBOSS'), LocalClassifier(IndividualTDE(), 'IndividualTDE'),
               LocalClassifier(WEASEL(), 'WAESEL')]

    ne = NaiveEnsemble(bm_list)
    ne.fit(X_train, y_train)
    print(ne.predict(X_test))
    ne.save_time_metrics('../Univariate_ts/'+dataset_name+'/NaiveEnsemble_time_metrics.csv')
    