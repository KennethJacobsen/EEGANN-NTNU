"""
dataProcessing, created for the eegann-ntnu bachelor project.
Module, singleton.

Created on 15 May 2019

@author: Christian Ovesen, KSVJ, Kenneth Olsen
"""

# Import
import numpy as np
import logging
import pywt
from numpy import array


# -----------------------------------------------Data processing------------------------------------------------------
# Normalizes data and takes fft
# @input y: data to normalize, list
# @input n: number of samples per fft, int
# @input cutoff: the cutoff used to remove unwanted sudden noise, int
# @return: one fft of signal
def normalizeAndFFT(y, n, cutoff):
    y = y - ((y[:1] + y[-1:]) / 2)
    y = y*np.hanning(n)
    y2 = abs(np.fft.fft(y)/n)
    y2 = y2[range(n//2)]
    for x in range(len(y2)):
        if y2[x] > cutoff:
            y2[x] = cutoff
    y2 = (y2/cutoff/2)**0.5
    y2 = y2/(cutoff/2)
    return y2


# Takes FFT of training data
# @input Signal: training data, list
# @input target: target command, int
# @input xtrain: list with train fft signals
# @input ytrain: list with train targets
# @input tw: windows size in s, float
# @input indent: number of samples window moves between each fft, int
# @input Hz: signal frequency, default int=128
# @input nodes: number of nodes on the headset default int=14
# @input cutoff: the cutoff used to remove unwanted sudden noise, default int=4
# return: fft of signal
def trainFFT(Signal, target, xtrain, ytrain, tw=1.5, indent=5, Hz=128, nodes=14, cutoff=4):
    n = int(Hz*tw)
    ffts = int((len(Signal[0])-n)/indent)
    for a in range(ffts):
        startSample = int(indent * a)
        endSample = int((indent * a)+n)
        for i in range(nodes):
            y = Signal[i][startSample: endSample]
            xtrain.append(normalizeAndFFT(y, n, cutoff))
            if i == 0:
                ytrain.append(target)
    logging.debug('xtrain shape: {}'.format(np.array(xtrain).shape))
    logging.debug('ytrain shape: {}'.format(np.array(ytrain).shape))
    return xtrain, ytrain


# Making fourier transforms of the signals and returning the transform as a list.
# @input n: n is used for calculating the fourier transform. It should be 96 when using 96 samples.
# @input nodes: number of nodes on headset, default int=14
# @input cutoff: the cutoff used to remove unwanted sudden noise, default int=4
# @return: output is a keras neural net model.
def evalFFT(tempChannelListLive, tw=1.5, indent=5, Hz=128, nodes=14, cutoff=4):
    n = int(Hz*tw)
    evalfft = []
    # Adjusting graph before using the hanning window function.
    # Then taking the fourier transform.
    for i in range(nodes):
        evalfft.append(normalizeAndFFT(tempChannelListLive, n, cutoff))
    return evalfft


# Shuffle order of samples/fft
# @input data: list of samples/fft to be shuffled
# @return: Shuffled lists
def shuffle(data, typeN, nodes=14):
    # Reshape
    reshapedTrain = np.reshape(data[0], (int(len(data[0]) / nodes), int(len(data[0][0]) * nodes)))
    reshapedTest = np.reshape(data[2], (int(len(data[2]) / nodes), int(len(data[2][0]) * nodes)))
    shuffledTraining = np.c_[reshapedTrain.reshape(len(reshapedTrain), -1), np.asarray(data[1]).reshape(len(data[1]))]
    shuffledTest = np.c_[reshapedTest.reshape(len(reshapedTest), -1), np.asarray(data[3]).reshape(len(data[3]))]
    # Shuffle lists
    np.random.shuffle(shuffledTraining)
    np.random.shuffle(shuffledTest)
    # Extract lists
    data[0] = shuffledTraining[:, :-1]
    data[1] = shuffledTraining[:, -1]
    data[2] = shuffledTest[:, :-1]
    data[3] = shuffledTest[:, -1]
    data[1] = [int(x) for x in data[1]]
    data[3] = [int(x) for x in data[3]]
    if typeN == 'Conv2D':
        data[0] = np.reshape(data[0], (int(len(data[0] / nodes)), nodes, int(len(data[0][0]) / nodes), 1))
        data[2] = np.reshape(data[2], (int(len(data[2] / nodes)), nodes, int(len(data[2][0]) / nodes), 1))
    else:
        data[0] = np.reshape(data[0], (int(len(data[0] / nodes)), nodes, int(len(data[0][0]) / nodes)))
        data[2] = np.reshape(data[2], (int(len(data[2] / nodes)), nodes, int(len(data[2][0]) / nodes)))
    return data


# Prepare data for FFT
# @input data: list of data to prep for FFT
def prepFFT(data, outputs, nodes, typeN):
    spo = int(len(data[0][0]) / (len(outputs) + 1))
    spo2 = int(len(data[2][0]) / (len(outputs) + 1))
    nodeSetTrain, nodeSetTest, trainSets, testSets = [], [], [], []
    for node in range(nodes):
        nodeSetTrain.append([])
        nodeSetTest.append([])
    for out in range(len(outputs) + 1):
        trainSets.append(nodeSetTrain)
        testSets.append(nodeSetTest)
    for x in range(len(outputs) + 1):
        for i in range(nodes):
            trainSets[x][i] = data[0][i][x * spo: (x * spo) + spo]
            testSets[x][i] = data[2][i][x * spo2: (x * spo2) + spo2]
        trainSets[x] = np.array(trainSets[x])
        testSets[x] = np.array(testSets[x])
    xTrain, yTrain, xTest, yTest = [], [], [], []
    for x in range(len(trainSets)):
        trainFFT(trainSets[x], x, xTrain, yTrain)
        trainFFT(testSets[x], x, xTest, yTest)
    data[0], data[1], data[2], data[3] = xTrain, yTrain, xTest, yTest
    logging.debug(np.array(data[0]).shape)
    data = shuffle(data, typeN)
    return data

# Split data into chunks for k-cross validation
def dataSplit(data, split=6, nodes=14, outs=5):
    dataFull = [[], []]
    # Input data structure [[[Node1], [Node2], [...]], [Answer]]
    for i in range(nodes):
        dataFull[0].append(np.append(data[0][i], data[2][i]))
    dataFull[1] = np.append(data[1], data[3])
    size = int(len(dataFull[0][0])/split)
    subSize = int(size/outs)
    data = dataFull
    train, test, trainAns, testAns = [], [], [], []
    for x in range(split):
        test.append([])
        train.append([])
        trainAns.append([])
        testAns.append([])
        for i in range(nodes):
            train[x].append([])
            test[x].append([])
            for o in range(outs):
                train[x][i].extend(data[0][i][max(o*size-(subSize*x), 0):((o+1)*size)-((x+1)*subSize)])
                test[x][i].extend(data[0][i][(o+1)*size-((x+1)*subSize):(o+1)*size-(subSize*x)])
                if o == (outs-1):
                    train[x][i].extend(data[0][i][(o+1)*size-(subSize*x):])
                if i == 0:
                    trainAns[x].extend(data[1][max(o*size-(subSize*x), 0):(o+1)*size-((x+1)*subSize)])
                    testAns[x].extend(data[1][(o+1)*size-((x+1)*subSize):(o+1)*size-(subSize*x)])
                    if o == (outs-1):
                        trainAns[x].extend(data[1][(o+1)*size-(subSize*x):])
            train[x][i], test[x][i] = array(train[x][i]), array(test[x][i])
        trainAns[x], testAns[x] = array(trainAns[x]), array(testAns[x])
        test[x], train[x] = array(test[x]), array(train[x])
    # Output data structure [[Train data], [Train answer], [Test data], [Test answer]]
    data = [train, trainAns, test, testAns]
    return data


def wavelet(data, outputs, nodes):
    dataPlace = [[], [], [], []]
    for z in range(len(data[0])):
        toWave, toWaveTest, wave, waveTest, ans, testAns, waveHolder, waveTestHolder = [], [], [], [], [], [], [], []
        waveLength = 0
        waveTestLength = 0
        lis = data[1][z].tolist()
        print('len lis: {}'.format(len(lis)))
        lis2 = data[3][z].tolist()
        print('len ans: {}'.format(len(ans)))
        for n in range(nodes):
            waveHolder.append([[], []])
            waveTestHolder.append([[], []])
            toWave.append([])
            wave.append([])
            waveTest.append([])
            for out in range(outputs):
                indices = [i for i, x in enumerate(lis) if x == out]
                indicesTest = [i for i, x in enumerate(lis2) if x == out]
                for elem in indices:
                    toWave[n].append(data[0][z][n][elem])
                toWaveTest.append([])
                for elem in indicesTest:
                    toWaveTest[n].append(data[2][z][n][elem])
                waveHolder[n][0] = pywt.dwt(toWave[n], 'sym9')[0]
                waveHolder[n][1] = pywt.dwt(toWave[n], 'sym9')[1]
                waveTestHolder[n][0].extend(pywt.dwt(toWaveTest[n], 'sym9')[0])
                waveTestHolder[n][1].extend(pywt.dwt(toWaveTest[n], 'sym9')[1])
                wave[n].extend([item for pair in zip(waveHolder[n][0], waveHolder[n][1]) for item in pair])
                waveTest[n].extend([item for pair in zip(waveTestHolder[n][0], waveTestHolder[n][1]) for item in pair])
                # print(array(wave).shape)
                if n == 0:
                    print('wavelength: {}'.format(waveLength))
                    print('len wave: {}'.format(len(wave[n])))
                    for r in range(waveLength, len(wave[n])):
                        ans.append(out)
                    for r in range(waveTestLength, len(waveTest[n])):
                        testAns.append(out)
                    print('len ans2: {}'.format(len(ans)))
                    print('z: {}'.format(z))
                    waveLength = len(ans)
                    waveTestLength = len(testAns)
            # wave[n] = np.reshape(wave[n], (len(wave[n])*len(wave[n][0]), 1))
        dataPlace[0].append(wave)
        dataPlace[1].append(ans)
        dataPlace[2].append(waveTest)
        dataPlace[3].append(testAns)
    for num in range(len(dataPlace)):
        for subNum in range(len(dataPlace[num])):
            dataPlace[num][subNum] = array(dataPlace[num][subNum])
    #    dataPlace[num] = array(dataPlace[num])
    dataPlace[0][0] = np.reshape(dataPlace[0][0], (dataPlace[0][0].shape[1], dataPlace[0][0].shape[0]))
    dataPlace[2][0] = np.reshape(dataPlace[2][0], (dataPlace[2][0].shape[1], dataPlace[2][0].shape[0]))
    return dataPlace
