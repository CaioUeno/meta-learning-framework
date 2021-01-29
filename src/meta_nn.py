import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.math as M
from sklearn.base import BaseEstimator
from abc import abstractmethod

def custom_loss(y_true, y_pred):

    abs_diff = K.abs(y_true - y_pred)
    mult = M.multiply(abs_diff, y_pred)
    custom_loss = abs_diff + mult
    return custom_loss

class NeuralNetworkMetaClassifier(object):
    
    def __init__(self, in_shape, out_shape, lstm_cells, batch_size=4, epochs=10):

        inputs = Input(shape=(1, in_shape,))
        lstm = LSTM(lstm_cells)(inputs)
        out = Dense(out_shape, activation='sigmoid')(lstm)

        self.meta_clf = Model(inputs=inputs, outputs=out)
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
        
class MetaClassifier(BaseEstimator):

    """
    Abstract class to define methods a meta classifier has to have to be used in this framework.

    Arguments:
        model: model.
    """

    def __init__(self, model, **kwargs):

        self.model = model

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_one(self, x):
        pass