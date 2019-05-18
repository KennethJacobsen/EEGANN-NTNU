"""
Logic, created for the eegann-ntnu bachelor project.
Module, singleton.

Created on 2 Apr 2019

@author: Christian Ovesen, KSVJ, Kenneth Olsen
"""

# Import
import queue
import numpy as np
from DataSetGenerator import CortexClient
import datetime
import time
from ANN.GA import GA
from ANN.dNN import DynamicNeuralNet
from RoboDK import roboDK
import threading
import multiprocessing
import logging
from GUI.GUI import guiThread
import pickle
import dataProcessing


# Cortex retry login
def cortexRelog():
    user = CortexClient.getUserLogin()[0]
    CortexClient.logout(user)
    CortexClient.defaultLogin()


# Cortex create session
# @return: create-return
def cortexSes():
    logging.debug(CortexClient.defaultAuthorize())
    logging.error(len(CortexClient.querySessions()))
    return CortexClient.createSession('active')


# Cortex subscribe
# @return: subscribe-return
def cortexSub():
    return CortexClient.subscribe(['eeg', 'dev'])


# Cortex setup
def cortex():
    logging.error('cortex')
    CortexClient.init()
    cortexSes()
    resp = cortexSub()
    # If subscribe returns an error msg try to log in again.
    if 'code' in resp:
        cortexRelog()
        sesResp = cortexSes()
        resp = cortexSub()
        if 'No headset connected.' in sesResp:
            logging.error('No headset connected.')
            return 'No headset connected.'
        elif 'code' in resp:
            logging.error('Cortex login failed')
            return 'Cortex login failed'
        else:
            return 'Successful'
    else:
        return 'Successful'


# Make datasets for the commands
# @input outputs: list of outputs, list of strings
# @input eegInnerQ: queue used to store eeg data, queue
# @input lockInner: lock used with queue, lock
# @input t: time per command, int default to 10
def makeCmds(outputs, eegInnerQ, lockInner, t=10):
    trainD, testD, trainA, testA = [], [], [], []
    logging.error('Idle in:')
    logging.error(3)
    time.sleep(1)
    logging.error(2)
    time.sleep(1)
    logging.error(1)
    time.sleep(1)
    gatherCmdData(0, trainD, testD, trainA, testA, eegInnerQ, lockInner, t)
    x = 1
    logging.error('InitCmds')
    for output in outputs:
        logging.error('{} in:'.format(output))
        logging.error(3)
        time.sleep(1)
        logging.error(2)
        time.sleep(1)
        logging.error(1)
        time.sleep(1)
        gatherCmdData(x, trainD, testD, trainA, testA, eegInnerQ, lockInner, t)
        x += 1
    logging.error('EndCmds')
    data = [trainD, trainA, testD, testA]
    return data


# collects last x seconds of data
# @input eegProQ: queue used to store eeg data, queue
# @input lockQ: lock used with queue, lock
# @input x: seconds of data, 15 default
# @input nodes: number of nodes, 14 default
# @return: x seconds of data in list of lists
def groupData(eegProQ, lockQ, x, nodes=14):
    command = copyQ(eegProQ, lockQ)
    for i in range(nodes):
        command[i] = command[i][-(128 * x):]
    logging.error(np.array(command).shape)
    return command


# Gather test and training data for one command.
# @input cmd: command to gather data for, int
# @input trainD: list to contain training data
# @input testD: list to contain test data
# @input trainA: list to contain training answers
# @input testA: list to contain test answers
# @input t: time to collect, time in seconds, int default to 10
def gatherCmdData(cmd, trainD, testD, trainA, testA, eegInnerQ, lockInner, t=10):
    time.sleep(t*2)
    logging.debug('gatherCMD trainD: {}'.format(np.array(trainD).shape))
    grp1 = groupData(eegInnerQ, lockInner, t * 2)
    if len(trainD) == 0:
        trainD.extend(grp1)
    else:
        for x in range(len(trainD)):
            trainD[x].extend(grp1[x])
    logging.debug('Answer: {}'.format(np.array(trainA).shape))
    for x in range(len(trainA), len(trainD[0])):
        trainA.append(cmd)
    time.sleep(t)
    logging.debug('gatherCMD testD: {}'.format(np.array(testD).shape))
    grp2 = groupData(eegInnerQ, lockInner, t)
    if len(testD) == 0:
        testD.extend(grp2)
    else:
        for x in range(len(testD)):
            testD[x].extend(grp2[x])
    logging.debug('Answer: {}'.format(np.array(testA).shape))
    for x in range(len(testA), len(testD[0])):
        testA.append(cmd)


# Gather test and training data for one command. Dummy data.
# @input cmd: command to gather data for, int
# @input trainD: list to contain training data
# @input testD: list to contain test data
# @input trainA: list to contain training answers
# @input testA: list to contain test answers
# @input t: time to collect, time in seconds, int default to 10
def dummyGatherCmdData(cmd, trainD, testD, trainA, testA, t=10):
    endT = datetime.datetime.now() + datetime.timedelta(seconds=t * 2)
    while datetime.datetime.now() < endT:
        dummyCollectData()
        logging.debug('collecting...')
    logging.debug('gatherCMD trainD: {}'.format(np.array(trainD).shape))
    grp1 = groupData(eegQ, lock, t * 2)
    if len(trainD) == 0:
        trainD.extend(grp1)
    else:
        for x in range(len(trainD)):
            trainD[x].extend(grp1[x])
    logging.debug('Answer: {}'.format(np.array(trainA).shape))
    for x in range(len(trainA), len(trainD[0])):
        trainA.append(cmd)
    endT = datetime.datetime.now() + datetime.timedelta(seconds=t)
    while datetime.datetime.now() < endT:
        dummyCollectData()
        logging.debug('collecting...')
    logging.debug('gatherCMD testD: {}'.format(np.array(testD).shape))
    grp2 = groupData(eegQ, lock, t)
    if len(testD) == 0:
        testD.extend(grp2)
    else:
        for x in range(len(testD)):
            testD[x].extend(grp2[x])
    logging.debug('Answer: {}'.format(np.array(testA).shape))
    for x in range(len(testA), len(testD[0])):
        testA.append(cmd)


# Add data to q
# @input q: queue
# @input data: data to add to queue
# input lockQ: lock
# @input data: data
def addQ(q, data, lockQ):
    lockQ.acquire()
    q.put(data)
    while q.qsize() > 1:
        q.get()
    lockQ.release()


# Get data from q
# @input q: queue
# input lockQ: lock
# @return: data
def getQ(q, lockQ):
    if q.qsize() > 0:
        lockQ.acquire()
        ret = q.get()
        lockQ.release()
    else:
        ret = 'ERROR'
        logging.warning('ERROR QUE IS EMPTY')
    return ret


# Copy data from q
# input q: queue
# input lockQ: lock
# return: copied data
def copyQ(q, lockQ):
    if q.qsize() > 0:
        lockQ.acquire()
        ret = q.get()
        q.put(ret)
        lockQ.release()
    else:
        ret = 'ERROR'
        logging.warning('ERROR QUE IS EMPTY')
    return ret


# Stores last 30seconds of data in list
# @input nodes: number of nodes, 14 default
# @input s: seconds of data to save, 30 default
def collectData(nodes=14, s=30):
    recent = CortexClient.receiveData()
    if 'eeg' in str(recent):
        eegData = recent['eeg']
        # Appends recent data to eegDataList, max s seconds
        for i in range(nodes):
            eegDataList[i].append(eegData[3 + i])
            eegDataList[i] = eegDataList[i][-(128 * s):]
        addQ(eegQ, eegDataList, lock)
    if 'dev' in str(recent):
        devData = recent['dev']
        addQ(devQ, devData, lock)


# Create dummy data
# @input nodes: number of nodes on headset in use, int default to 14
# @input s: number of seconds per command, int default to 30
def dummyCollectData(nodes=14, s=30):
    recentEEG = np.random.random((nodes, 128*s)).tolist()
    recentDEV = [4, 2, [np.random.random((nodes, 1)).tolist()]]
    logging.debug(np.array(recentEEG).shape)
    logging.debug(recentDEV)
    addQ(eegQ, recentEEG, lock)
    addQ(devQ, recentDEV, lock)


# Creates DynamicNeuralNet from the dNN.py
# @input data: data to use when creating the neural net, list of lists
# @input outs: number of outputs to have from the neural net, int
# @input typeNet: type of NN, string default to Dense
# @input hidden: number of hidden layers, int default to zero
# @input neurons: number of neurons, list of int, default to None
# @input fft: value to determine use of fft, boolean default to False
# @input wave: value to determine use of wavelet, boolean default to False
# @input **kwargs: possible dict of keyword arguments.
# @return: the neural net
def createNN(data, outs, typeNet='Dense', hidden=0, neurons=None, fft=False, wave=False, **kwargs):
    if 'epoch' in kwargs:
        dNN = DynamicNeuralNet()
        dNN.create(typeNet, data, hidden, neurons, outs, fft, wave, epochNumber=kwargs['epoch'])
    else:
        dNN = DynamicNeuralNet()
        dNN.create(typeNet, data, hidden, neurons, outs, fft, wave)
    return dNN


# Receive outputs from the NN
# @input data: data used to predict, list
# @input NN: neural net to use, DynamicNeuralNet from dNN.py
# @input fft: value to determine use of fft, boolean default to False
# @return: output from the NN, int value.
def receiveNNOut(data, NN, fft=False):
    output = NN.predictLive(data, fft)
    return output


# Record eeg data for one command at a time.
# @input eegInnerQ: queue for eeg data, queue
# @input lockInner: lock to use with queue, lock
# @input timeForEachCommand: the time in seconds, int default to 15.
# @return: output from the NN, int value.
# Not in use
def record(eegInnerQ, lockInner, timeForEachCommand=15):
    time.sleep(timeForEachCommand)
    return np.asanyarray(groupData(eegInnerQ, lockInner, timeForEachCommand))


# Change NNout to roboMove
# @input nn: list with output values
# @return: roboMove
def nnOutToMove(nn):
    move = nn.index(max(nn))
    return move


# Train a dNN
# @input commands: list of commands, formatting = [[trainingSet],[trainingReference],[testSet],[testReference]
# @input epochs: number of epochs to train for, int
# @input dNN: Neural net to train, DynamicNeuralNet from dNN.py
# Not in use
def trainDNN(commands, epochs, dNN):
    dNN.train(commands, epochs)


# Start roboDKprocess and move robot using eeg and loading a dNN
# @input roboEEGQ: queue used to store eeg data, queue
# @input roboProQ: queue used to store start and stop commands for the robo process, queue
# @input proLock: lock to be used with the queues, lock
# @input dNN: neural net to use, DynamicNeuralNet from dNN.py
# Not in use
def roboProLive(roboEEGQ, roboProQ, proLock, dNN):
    # dNN = DynamicNeuralNet()
    dNN.loadModel()
    roboDK.__init__()
    live = True
    while live:
        if roboProQ.qsize() > 0:
            live = getQ(roboProQ, proLock)
        else:
            logging.debug(roboEEGQ)
            if roboEEGQ.qsize() > 0:
                data = copyQ(roboEEGQ, proLock)
                for x in range(14):
                    data[x] = data[x][-32:]
                nn = receiveNNOut(data, dNN).tolist()
                move = nnOutToMove(nn[0])
                if move == 0:
                    pass
                else:
                    logging.debug(move)
                    roboDK.moveBot(move)
            else:
                pass


# Start roboDK and move robot using dummydata and dNN
# @input dNN: neural net to use, DynamicNeuralNet from dNN.py
# Not in use
def dummyRoboLive(dNN):
    roboDK.__init__()
    while True:
        dummyCollectData()
        data = copyQ(eegQ, lock)
        for x in range(14):
            data[x] = data[x][-1:]
        nn = receiveNNOut(data, dNN)
        move = nnOutToMove(nn[0])
        logging.error(move)
        if move == 0:
            logging.error('Idle')
        else:
            roboDK.moveBot(move)


# Call evalFFT command from dataProcessing.py
# @input n: number of samples used per fft, int default to 96
# @input nodes: number of nodes on headset, int default to 14
# @input cutoff: value to cutoff data at, int default to 4
# @return fft data
def evalFFT(n=96, nodes=14, cutoff=4):
    tempChannelListLive = copyQ(eegQ, lock)
    for i in range(nodes):
        tempChannelListLive[i] = tempChannelListLive[i][-n:]
    ret = dataProcessing.evalFFT(tempChannelListLive, n, nodes, cutoff)
    return ret


# Call shuffle command from dataProcessing.py
# @input data: list of data to prepare for fft, list of lists
# @input nodes: number of nodes on headset, int default to 14
# @input typeN: type of DynamicNeuralNet
# @return shuffled data
def shuffle(data, typeN, nodes=14):
    ret = dataProcessing.shuffle(data, typeN, nodes)
    return ret


# Call prepFFT command from dataProcessing.py
# @input data: list of data to prepare for fft, list of lists
# @input outputs: list of outputs, list of strings
# @input nodes: number of nodes on headset, int
# @input typeN: type of DynamicNeuralNet
# @return fft data
def prepFFT(data, outputs, nodes, typeN):
    ret = dataProcessing.prepFFT(data, outputs, nodes, typeN)
    return ret


# -------------------------------------------RUN AND TEST METHODS------------------------------------------------------
# Setup and run guiThread
def testGui():
    cortex()
    collectData()
    guiThread(lock, eegQ, devQ, guiQ, returnQ)
    while True:
        collectData()


# Test wavelet transform
# @input typeN: type of neural net, string
# @input hidden: number of hidden layres, int
# @input neurons: list with number of neurons per layer, list of int
# @input outputs: number of outputs, int default to five
# @input nodes: number of nodes on headset in use, int default to 14
def testWavelet(typeN, hidden, neurons, outputs=5, nodes=14):
    with open('dataset3.pickle', 'rb') as handle:
        data = pickle.load(handle)
    data = dataProcessing.dataSplit(data)
    data = dataProcessing.wavelet(data, outputs, nodes)
    wavelet = [data[0][0], data[1][0], data[2][0], data[3][0]]
    createNN(wavelet, outputs, typeN, hidden, neurons, wave=True)


# Setup and run mainThread.
# @input outputs: list of outputs, list of strings
# @input nnVars: dictionary with variables for the neural net
# @input eegInnerQ: queue for keeping the eeg signals, queue
# @input lockInner: lock to be used with queue, lock
# @input epochs: number of epochs to run, int
# @input t: time to record per output, int
# @input fft: value to determine use of fft, boolean default to False
def mainRun(outputs, nnVars, eegInnerQ, lockInner, epochs, t, fft):
    # if cortex() != 'Successful':
    #    return 'Cortex failed'
    mainThread(outputs, nnVars, eegInnerQ, lockInner, epochs, t, fft=fft)
    # while True:
    #    collectData(s=t*2)


# Setup and run GA.
# @input outputs: list of outputs, list of strings
# @input GAVars: dictionary with variables for the genetic algorithm
# @input fft: value to determine use of fft, boolean default to False
# @input maxMinVal: max and min values for genetic algorithm, list
def GARun(outputs, GAVars, fft, maxMinVal):
    with open('dataset3.pickle', 'rb') as handle:
        data = pickle.load(handle)
    for x in range(len(data)):
        data[x] = np.array(data[x])
    GAOptimized = [GA(data, outs=outputs, popSize=GAVars["popSize"], genSize=GAVars["genSize"],
                      mutationChance=GAVars["mutationChance"], fft=fft, maxMinVal=maxMinVal)]
    GAOptimized[0].run()


# --------------------------------------------------THREAD CLASSES-----------------------------------------------------
class roboThread(threading.Thread):
    """
    Class for threading roboDK commands.
    """
    def __init__(self, outputs, eegProQ, roboProQ, lockPro, NN):
        threading.Thread.__init__(self)
        self.outputs = outputs
        self.eegQ = eegProQ
        self.roboQ = roboProQ
        self.lock = lockPro
        self.dNN = NN
        roboDK.__init__()
        self.start()

    def run(self):
        print('Threading')
        live = True
        while live:
            logging.error('live')
            if self.roboQ.qsize() > 0:
                live = getQ(self.roboQ, self.lock)
            else:
                logging.error('roboLive')
                roboProLive(self.eegQ, self.roboQ, self.lock, self.dNN)


class mainThread(threading.Thread):
    """
    Class for threading old system, main
    """
    def __init__(self, outputs, nnVars, eegInnerQ, lockInner, epochs, t, nodes=14, fft=False):
        threading.Thread.__init__(self)
        self.outputs = outputs
        self.eegQ = eegInnerQ
        self.lock = lockInner
        self.t = t
        self.vars = nnVars
        self.epoch = epochs
        self.nodes = nodes
        self.fft = fft
        self.start()

    def run(self):
        # data = makeCmds(self.outputs, self.eegQ, self.lock, self.t)
        # Save dataset
        # with open('dataset3.pickle', 'wb') as handle:
        #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Load dataset
        with open('dataset3.pickle', 'rb') as handle:
            data = pickle.load(handle)
        if self.fft:
            data = prepFFT(data, self.outputs, self.nodes, self.vars['Type'])
        else:
            for x in range(len(data)):
                data[x] = np.array(data[x])
        dNN = createNN(data, (len(self.outputs)+1), self.vars['Type'], self.vars['Layers'], self.vars['Neurons'],
                       fft=self.fft, epoch=self.epoch)
        dNN.saveModel()
        evalD = [np.reshape(data[2], (len(data[2][0]), len(data[2]), 1)), data[3]]
        print(dNN.modelScore(evalD))
        roboDK.__init__()
        while True:
            if self.fft:
                predictData = evalFFT()
            else:
                predictData = copyQ(self.eegQ, self.lock)
                for n in range(self.nodes):
                    predictData[n] = predictData[n][-1:]
            move = receiveNNOut(predictData, dNN, fft=self.fft)
            move = move.tolist()
            move = nnOutToMove(move)
            logging.error(move)
            roboDK.moveBot(move)
            time.sleep(0.05)


# ----------------------------------------------------MAIN-------------------------------------------------------------
# Starts gui and runs everything.
# Not implemented.
def main():
    global numCmds
    readGui = True
    guiThread(lock, eegQ, devQ, guiQ, returnQ)
    while readGui:
        if guiQ.qsize() > 0:
            toDo = getQ(guiQ, lock)
            if toDo['Command'] == 'Kill':
                readGui = False
            elif toDo['Command'] == 'StartCortex':
                addQ(returnQ, cortex(), lock)
            elif toDo['Command'] == 'Create':
                testCreateFullSizeNN(toDo['Outputs'], toDo['Variables'])
            elif toDo['Command'] == 'Train':
                outputs = toDo['Outputs']
                if len(outputs) > numCmds:
                    numCmds = len(outputs)
                epoch = toDo['Epoch']
                trainD, testD, trainA, testA = [], [], [], []
                gatherCmdData(0, trainD, testD, trainA, testA)
                x = 1
                addQ(returnQ, 'InitCmds', lock)
                for output in outputs:
                    addQ(returnQ, output, lock)
                    gatherCmdData(x, trainD, testD, trainA, testA)
                    x += 1
                addQ(returnQ, 'EndCmds', lock)
                data = [trainD, trainA, testD, testA]
                trainDNN(data, epoch=epoch)
                addQ(returnQ, 'Successful', lock)
            elif toDo['Command'] == 'TrainGA':
                outputs = toDo['Outputs']
                # TODO: write command to GA
                pass
            elif toDo['Command'] == 'LiveRobo':
                roboLive(dNN)
            # TODO: add profile commands. If enough time.
            else:
                addQ(returnQ, 'Unknown command', lock)


if __name__ == '__main__':
    logging.basicConfig(filename='debug.log', level='ERROR')

    # Variables needed to run functions:
    returnQ, guiQ, devQ, eegQ, collectQ = queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()
    # returnQ, guiQ, devQ, eegQ, roboQ = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue(), \
    #                                   multiprocessing.Queue(), multiprocessing.Queue()
    eegDataList = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    lock = multiprocessing.Lock()
    globalOutputs = ['Left', 'Right', 'Up', 'Down']
    globalNNVars = {'Type': 'Conv1D', 'Layers': 1, 'Neurons': [8, 4]}
    globalGAVars = {"popSize": 4, "genSize": 2, "mutationChance": 10}

    # Gene setup [type net, number of hidden layers, [neurons layer one, neurons layer two, neurons layer three...],
    # epochs, loss function]
    # Type positions: ["Dense", "Conv1D", "Conv2D", "LSTM"]
    # Loss positions: ["kullback_leibler_divergence", "sparse_categorical_crossentropy", "categorical_crossentropy"]
    globalMin = [1, 0, [14, 12, 10, 8, 6, 2], 0, 1]
    globalMax = [1, 0, [400, 300, 200, 150, 100, 5], 2, 1]
    globalMaxMinVal = [globalMin, globalMax]

    # Write command to run below

else:
    pass
