"""
Created on 22 Mar 2019

@author: Christian Ovesen, KSVJ
"""

# Data structure [[Train data], [Train answer], [Test data], [Test answer]
from time import time
import timeit
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Conv2D, BatchNormalization, Flatten, Conv1D, \
    MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import numpy as np
from collections import Counter


class DynamicNeuralNet(object):
    """
    classdocs
    Creates and trains a neural net of the type and size of your choosing
    """
    def __init__(self):
        # Command for opening TensorBoard: "tensorboard --logdir=logs/"
        self.batchSize = 1
        self.tb = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, batch_size=self.batchSize,
                              write_graph=True, write_grads=True, write_images=True, embeddings_freq=0)
        self.model = None
        self.scoreList = []
        self.trainTime = None
        self.padding = 0
        self.neurons = None
        self.outs = 0
        self.fft = False
        self.wave = False
        self.filterSize = 0
        self.epochNumber = 0
        self.typeNet = None
        self.numberHiddenLayers = 0
        self.showNetSetup = False
        self.lossFunction = None
        if tf.test.is_gpu_available():
            self.gpu = True
        else:
            self.gpu = False

    # ------------------------------------------------- Creators -------------------------------------------------
    # Checks if gpu is available before running createModel()
    # @input typeNet: type of net to be created, string
    # @input data: data to train and test new model on
    # @input numberHiddenLayers: number of hidden layers, int
    # @input neurons: list of neuron values per layer, list of int
    # @input outs: number of outputs, int
    # @input fft: value to determine use of fft, boolean
    # @input wave: value to determine use of wavelet, boolean
    # @input epochNumber: number of epochs to be used for training, int default to 10
    # @input filterSize: size of filter to be used, int default to four
    # @input padding: value used to decide padding type from list, int default to zero
    # @input showNetSetup: value used to decide if setup should be logged, boolean default to False
    # @input lossFunction: loss function to use, string default to 'sparse_categorical_crossentropy'
    def create(self, typeNet, data, numberHiddenLayers, neurons, outs, fft, wave, epochNumber=10, filterSize=4,
               padding=0, showNetSetup=False, lossFunction="sparse_categorical_crossentropy"):
        self.padding = padding
        self.neurons = neurons
        self.outs = outs
        self.fft = fft
        self.wave = wave
        self.filterSize = filterSize
        self.epochNumber = epochNumber
        self.typeNet = typeNet
        self.numberHiddenLayers = numberHiddenLayers
        self.showNetSetup = showNetSetup
        self.lossFunction = lossFunction
        if self.gpu:
            with tf.device('/gpu:0'):
                self.createModel(data)
        else:
            self.createModel(data)

    # Constructor
    # Different types of NN supported (CuDNNLSTM, LSTM, Conv1D, Dense)
    # @input data: data to use for training and testing
    def createModel(self, data):
        self.model = Sequential()
        if self.showNetSetup:
            logging.error('\n TYPE: {} \n HIDDEN: {} \n NEURONS: {} '
                          '\n EPOCHS: {}'.format(self.typeNet, self.numberHiddenLayers, self.neurons, self.epochNumber))
        self.batchSize = self.batch(data)
        self.tb.batch_size = self.batchSize
        self.tb.update_freq = 'epoch'
        if self.fft:
            inputShape = data[0].shape[1:]
        elif self.wave:
            inputShape = (len(data[0]), 1)
        else:
            inputShape = (len(data[0]), 1)
        paddingType = ["same", "valid"]
        if self.typeNet == 'LSTM':
            if self.gpu:
                self.model.add(CuDNNLSTM(self.neurons[0], input_shape=inputShape, return_sequences=True,
                                         batch_size=self.batchSize, stateful=True))
            else:
                self.model.add(LSTM(self.neurons[0], input_shape=inputShape, return_sequences=True,
                                    batch_size=self.batchSize, stateful=True))
            self.model.add(Dropout(0.2))
        elif self.typeNet == "Conv1D":
            self.model.add(Conv1D(self.neurons[0], self.filterSize, padding=paddingType[self.padding],
                                  input_shape=inputShape, activation="relu", batch_size=self.batchSize))
            self.model.add(Dropout(0.2))
        elif self.typeNet == "Conv2D":
            self.model.add(Conv2D(self.neurons[0], (self.filterSize, 2), input_shape=inputShape))
            self.model.add(Activation('relu'))
            # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        elif self.typeNet == "Dense":
            self.model.add(Dense(self.neurons[0], input_shape=inputShape, activation='relu', batch_size=self.batchSize))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
        else:
            logging.error("No valid NN type selected")

        for x in range(0, self.numberHiddenLayers):
            if x == (self.numberHiddenLayers - 1) and self.typeNet == "LSTM":
                self.dynamicLayerCreator(lastLstm=True)
            else:
                self.dynamicLayerCreator()
        self.model.add(Flatten())
        if self.fft:
            self.model.add(Dense(self.neurons[len(self.neurons)-1], activation='sigmoid'))
            self.model.add(Dense(self.outs, activation="sigmoid"))
            self.model.compile(loss="sparse_categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])
        else:
            self.model.add(Dense(self.neurons[len(self.neurons)-1], activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.outs, activation="softmax"))
            opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
            # Loss Functions:
            # kullback_leibler_divergence
            # sparse_categorical_crossentropy
            self.model.compile(loss=self.lossFunction, optimizer=opt, metrics=['accuracy'])
        self.train(data)

    # Creates a new layer for the model
    # @input lastLstm: set true if it is the last lstm layer, boolean default to False
    def dynamicLayerCreator(self, lastLstm=False):
        if self.typeNet == "LSTM":
            if not lastLstm:
                if self.gpu:
                    self.model.add(CuDNNLSTM(self.neurons, return_sequences=True, batch_size=self.batchSize,
                                             stateful=True))
                else:
                    self.model.add(LSTM(self.neurons, return_sequences=True, batch_size=self.batchSize, stateful=True))
                self.model.add(BatchNormalization())
            else:
                if self.gpu:
                    self.model.add(CuDNNLSTM(self.neurons, batch_size=self.batchSize, stateful=True))
                else:
                    self.model.add(LSTM(self.neurons, batch_size=self.batchSize, stateful=True))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.2))
        elif self.typeNet == "Conv1D":
            paddingType = ["same", "valid"]
            self.model.add(Conv1D(self.neurons, self.filterSize, activation="relu", padding=paddingType[self.padding],
                                  batch_size=self.batchSize))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
        elif self.typeNet == "Dense":
            self.model.add(Dense(self.neurons, activation='relu', batch_size=self.batchSize))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
        else:
            logging.error("No valid NN type selected")

# ------------------------------------------------- Training functions -------------------------------------------------
    # Runs train function on model with timer
    # @input data: Data to use for training
    def train(self, data):
        # data = self.normalize([data[0], data[2]])
        if self.fft or self.wave:
            self.trainTime = timeit.Timer(lambda: self.fitModel(trainData=data[0], trainAnswer=data[1],
                                                                testData=data[2], testAnswer=data[3])).timeit(number=1)
        else:
            trainDataNP = np.array(data[0])
            trainAnswerNP = np.array(data[1])
            testDataNP = np.array(data[2])
            testAnswerNP = np.array(data[3])
            logging.debug(trainDataNP.shape)
            trainDataNP = trainDataNP.reshape((trainDataNP.shape[1], trainDataNP.shape[0], 1))
            testDataNP = testDataNP.reshape((testDataNP.shape[1], testDataNP.shape[0], 1))
            for sample in trainDataNP:
                sample = self.normalize(sample)
            for sample in testDataNP:
                sample = self.normalize(sample)
            self.trainTime = timeit.Timer(lambda: self.fitModel(trainData=trainDataNP, trainAnswer=trainAnswerNP,
                                                                testData=testDataNP,
                                                                testAnswer=testAnswerNP)).timeit(number=1)

    # Starts training of model
    # @input trainData: training data, list
    # @input trainAnswer: training answers, list
    # @input testData: test data, list
    # @input testAnswer: test answers, list
    def fitModel(self, trainData, trainAnswer, testData, testAnswer):
        self.model.fit(trainData, trainAnswer, batch_size=self.batchSize, epochs=self.epochNumber,
                       validation_data=(testData, testAnswer), verbose=2)
        # validation_split=0.2, callbacks=[self.tb])
        # validation_data=(testDataNP, testAnswerNP), callbacks=[self.tb])

    # Evaluate model
    # @input evalData: evaluation data, list with list of data and list of answers
    def evaluateScore(self, evalData):
        self.scoreList = self.model.evaluate(evalData[0], evalData[1], verbose=0, batch_size=self.batchSize)

    # Returns scoring parameters for model
    # @input evalData: evaluation data, list with list of data and list of answers
    # @return: score values
    def modelScore(self, evalData):
        evalTime = timeit.Timer(lambda: self.evaluateScore(evalData)).timeit(number=1)
        # score [loss, accuracy, train time(in sec), evaluation time]
        score = [self.scoreList[0], self.scoreList[1], self.trainTime, evalTime]
        return score

# ------------------------------------------------- Prediction functions ----------------------------------------------
    # Predicts answer from new data
    # @input data: data to predict on
    # @return: prediction
    def predictLive(self, data):
        if self.fft:
            npData = np.array(data)
            npData = npData.reshape((1, npData.shape[0], npData.shape[1], 1))
            return self.model.predict(npData, batch_size=self.batchSize)
        else:
            npData = np.array(data)
            npData = npData.reshape((npData.shape[1], npData.shape[0], 1))
            npData[0] = self.normalize(npData[0])
            return self.model.predict(npData, batch_size=self.batchSize)

# ------------------------------------------------- Data processing -------------------------------------------------
    # Normalizes training and testing data
    # @input data: data to normalize
    # @return: normalized data
    def normalize(self, data):
        divideBy = max(data)-min(data)
        if divideBy == 0:
            divideBy = 0.0001
        data = (data - min(data)) / divideBy
        return data

    # Find highest batch size
    # @input data: data to find batch size from
    # @input maxBatchSize: the highest possible batch size, needs to be set to keep GPU to run out of memory,
    # int default set to 1000
    # return: highest possible batch size
    def batch(self, data, maxBatchSize=1000):
        bList = []
        for d in range(0, 2):
            if d == 0:
                u = 0
            else:
                u = 2
            if self.fft or self.wave:
                for x in data[u].shape[:1]:
                    for y in range(1, x):
                        if x % y == 0:
                            if y < maxBatchSize:
                                bList.append(y)
            else:
                for x in data[u].shape:
                    for y in range(1, x):
                        if x % y == 0:
                            if y < maxBatchSize:
                                bList.append(y)
        logging.debug('batch list: {}'.format(bList))
        retSize = max([item for item, count in Counter(bList).items() if count >= 2])
        return retSize

    # Saves model
    # @input fileName: where to store model, string default set do 'dNN_model.h5'
    def saveModel(self, fileName='dNN_model.h5'):
        self.model.save(fileName)

    # Loads model
    # @input fileName: where to load model from, string default set do 'dNN_model.h5'
    def loadModel(self, fileName='dNN_model.h5'):
        self.model = load_model(fileName)
