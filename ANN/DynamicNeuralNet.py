'''
Created on 22 Mar 2019

@author: Christian
'''

# Data structure [[Train data], [Train answer], [Test data], [Test answer]
from time import time
import logging
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Conv2D, BatchNormalization
from keras.callbacks import TensorBoard
import operator
import random
# import numpy

class DynamicNeuralNet(object):
    '''
    classdocs
    '''
# ------------------------------------------------- Creators -------------------------------------------------
    # Constructor
    # Different types of NN supported (CuDNNLSTM, LSTM, Conv2D, Dense)
    def __init__(self, typeNet, data, numberHiddenLayers, neurons, epochNumber=10, filterSize=(1,1), padding=0):
        # Command for opening TensorBoard: "tensorboard --logdir=logs/"
        self.tb = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, batch_size=32, write_graph=True,
                                    write_grads=True, write_images=True, embeddings_freq=0, update_freq='epoch')
        
        self.model = Sequential()
        if typeNet == "CuDNNLSTM" or "LSTM":
            self.model.add(exec(typeNet)(neurons, input_shape=(len(data[0][0])), return_sequences=True))
            self.model.add(BatchNormalization(), activation="relu")
            self.model.add(Dropout(0.2))
        elif typeNet == "Conv2D":
            paddingType = ["same", "valid"]
            self.model.add(Conv2D(neurons, filterSize, paddingType[padding], input_shape=(data[0].shape[1:])))
            self.model.add(BatchNormalization(), activation="relu")
        elif typeNet == "Dense":
            self.model.add(Dense(neurons, input_shape=(len(data[0])), return_sequences=True))
            self.model.add(BatchNormalization(), activation="relu")
            self.model.add(Dropout(0.2))
        else:
            logging.error("No valid NN type selected")
        
        for x in range(0, numberHiddenLayers):
            if x == numberHiddenLayers and typeNet == "CuDNNLSTM" or "LSTM":
                self.dynamicLayerCreator(typeNet, neurons[x], filterSize, padding, lastLstm=True)
            else:
                self.dynamicLayerCreator(typeNet, neurons[x], filterSize, padding)
        
        self.model.add(Dense(neurons[len(neurons-1)], BatchNormalization(), activation="relu"))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(len(data[1][0]), activation="softmax"))
        
        opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, matrics=["accuracy"])
        
        self.train(data, epochNumber)

    # Creates a new layer for the eural net
    def dynamicLayerCreator(self, typeNet, neurons, filterSize=(1,1), padding=0, lastLstm=False):
        if typeNet == "CuDNNLSTM" or "LSTM":
            if lastLstm == False:
                self.model.add(exec(typeNet)(neurons), return_sequences=True)
                self.model.add(BatchNormalization())
            else:
                self.model.add(exec(typeNet)(neurons))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.2))
        elif typeNet == "Conv2D":
            paddingType = ["same", "valid"]
            self.model.add(Conv2D(neurons, filterSize, paddingType[padding]))
            self.model.add(BatchNormalization(), activation="relu")
        elif typeNet == "Dense":
            self.model.add(Dense(32, BatchNormalization(), activation="relu"))
            self.model.add(Dropout(0.2))
        else:
            logging.error("No valid NN type selected")

# ------------------------------------------------- Training functions -------------------------------------------------
    # Trains neural net
    def train(self, data, epochNumber):
        normalizedData = self.normalize([data[0], data[2]])
        self.model.fit(normalizedData[0], data[1], epochs=epochNumber, validation_data=(normalizedData[1], data[3]), verbose=1, callbacks=[self.tb])

    # Normalizes training and testing data
    def normalize(self, data):
        data[0] = (data[0] - min(data[0]))/(max(data[0]) - min(data[0]))
        data[1] = (data[1] - min(data[1]))/(max(data[1]) - min(data[1]))
        return data

# ------------------------------------------------- Prediction functions -------------------------------------------------
    def predictLive(self, data, batchSize=32):
        return self.model.predict(data, batch_size=batchSize)
