import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model

class NeuralNetworkMetaClassifier(object):
    
    def __init__(self, in_shape, out_shape):

        inputs = Input(shape=(in_shape,))
        out = Dense(out_shape)(inputs)

        self.meta_clf = Model(inputs=inputs, outputs=out)
        self.meta_clf.compile(optimizer='rmsprop', loss='mae')
    
    def fit(self, X, y):
        
        X_array = np.array(X['dim_0'].apply(lambda x: x.values).tolist())
        self.meta_clf.fit(X_array, y)

    def predict(self, X):

        pred = self.meta_clf.predict(X[0].values.reshape(1, -1))[0].ravel()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred
        
