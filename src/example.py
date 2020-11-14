from metamodel import MetaLearningModel

from sktime.classification.dictionary_based import BOSSEnsemble, BOSSIndividual, TemporalDictionaryEnsemble, IndividualTDE
# from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier # ElasticEnsemble with problems
# from sktime.classification.distance_based._shape_dtw import ShapeDTW
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
# ShapeletTransformClassifier too slow!
# from sktime.classification.shapelet_based import ShapeletTransformClassifier

from sklearn import svm
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from utils import *
from tqdm import tqdm
import sys

models = [BOSSEnsemble(), BOSSIndividual(), TimeSeriesForest()]

dataset = 'CBF'

X_train, y_train = load_from_tsfile_to_dataframe('../data/'+dataset+'/'+dataset+'_TRAIN.ts')
y_train = pd.Series(y_train).values

X_test, y_test = load_from_tsfile_to_dataframe('../data/'+dataset+'/'+dataset+'_TEST.ts')


model = MetaLearningModel(RandomIntervalSpectralForest(), models,
                          'classification', 'binary')

model.fit(X_train, y_train)
model.predict(X_test)
