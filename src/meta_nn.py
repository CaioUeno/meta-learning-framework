import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model

class NeuralNetworkMetaClassifier(object):
    
    def __init__(self, in_shape, out_shape, lstm_cells, batch_size=4, epochs=5):

        inputs = Input(shape=(1, in_shape,))
        lstm = LSTM(lstm_cells)(inputs)
        out = Dense(out_shape)(lstm)

        self.meta_clf = Model(inputs=inputs, outputs=out)
        self.meta_clf.compile(optimizer='rmsprop', loss='mae')

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
        
