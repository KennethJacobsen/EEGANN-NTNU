"""
Created on 21 Apr 2019

@author: Christian Ovesen, KSVJ
"""

import logging
from numpy import array
from random import randint
from ANN.dNN import DynamicNeuralNet
import sys
sys.path.append("../")
import dataProcessing


# Run dataSplit command from dataProcessing.py
# @input data: data to split up
# @input split: number of splits, int default to 6
# @input nodes: number of nodes for headset in use, int default to 14
# @return: Data post split
def dataSplit(data, split=6, nodes=14):
    ret = dataProcessing.dataSplit(data, split, nodes)
    return ret


# -------------- GA functions --------------
# Gene setup [type net, number of hidden layers, [neurons layer one, neurons layer two, neurons layer three...],
# epochs, fft true/false, loss function]
# Gene min values [0, 0, [10,10,10,10,10,10], 5, 0, 0]
# Gene max values [3, 5, [100,100,100,100,100,100], 25, 1, 1]
# Generates random genes and returns them
# @input maxMinVal: max and min values for gene setup. list of lists
# @return list of genes
def randomGenes(maxMinVal):
    typeList = ["Dense", "Conv1D", "Conv2D", "LSTM"]
    typeNet = None
    lossFunction = ["kullback_leibler_divergence", "sparse_categorical_crossentropy", "categorical_crossentropy"]
    if maxMinVal is not None:
        genes = [typeList[randint(maxMinVal[0][0], maxMinVal[1][0])], randint(maxMinVal[0][1], maxMinVal[1][1]),
                 [randint(maxMinVal[0][2][0], maxMinVal[1][2][0]), randint(maxMinVal[0][2][1], maxMinVal[1][2][1]),
                  randint(maxMinVal[0][2][2], maxMinVal[1][2][2]), randint(maxMinVal[0][2][3], maxMinVal[1][2][3]),
                  randint(maxMinVal[0][2][4], maxMinVal[1][2][4]), randint(maxMinVal[0][2][5], maxMinVal[1][2][5])],
                 randint(maxMinVal[0][3], maxMinVal[1][3]), lossFunction[randint(maxMinVal[0][4], maxMinVal[1][4])]]
    else:
        if typeNet is None:
            genes = [typeList[randint(0, 3)], randint(0, 5), [randint(10, 100), randint(10, 100), randint(10, 100),
                                                              randint(10, 100), randint(10, 100), randint(10, 100)],
                     randint(5, 25), randint(0, 1), randint(0, 1)]
        else:
            genes = [typeNet, randint(0, 5), [randint(10, 100), randint(10, 100), randint(10, 100), randint(10, 100),
                                              randint(10, 100), randint(10, 100)], randint(5, 25), randint(0, 1),
                     randint(0, 1)]
    return genes


# Calculates fitness of a NN based on speed and accuracy of the predictions from the test data set
# @input score: list of score values, format: [loss, acc, train time, test time]
# @return: Fitness value
def fitness(score):
    fitnessI = (10/(score[0])) + (100*score[1]) + (10/score[2]) + (0.1/score[3])
    return fitnessI


# Sort list of neural nets based on fitness
# @input dNNPop: Population of neural nets
# @return sorted dNNPop
def sortFitness(dNNPop):
    # sort list with key
    dNNPop.sort(key=takeThirdElement, reverse=True)
    logging.error("\n Sorted population:")
    for x in dNNPop:
        logging.error('Fitness:{}'.format(x[2]))
        logging.error('TestLoss:{} TestAcc:{} TrainTime:{} EvalTime:{}'.format(x[3][0], x[3][1], x[3][2], x[3][3]))
        logging.error('Network setup: {}\n'.format(x[1]))
    return dNNPop


# Select element three in list
# @input elem: List to select from
def takeThirdElement(elem):
    return elem[2]


# Train neural net using k-crossed data.
# @input genes: genes to be used
# @input data: data to train with, k-crossed
# @input outs: number of output values, int
# @input nodes: number of nodes, int
# @input fft: value to determine use of fft, boolean
# @input wave: value to determine use of wavelet, boolean
# @return list with format:
#   [DynamicNeuralNet, list of values for DynamicNeuralNet, fitness value, list with fitness factors]
def kCrossValidate(genes, data, outs, nodes, fft, wave):
    fitnessScore = []
    for i in range(len(data[0])):
        fitnessScore.append([])
    newNN = None
    score = []
    showNetSetup = True
    for i in range(len(data[0])):
        print("K-cross round: {}".format(i))
        kCrossData = [data[0][i], data[1][i], data[2][i], data[3][i]]
        if fft:
            kCrossData = dataProcessing.prepFFT(kCrossData, outs, nodes, genes[0])
        newNN = DynamicNeuralNet()
        newNN.create(typeNet=genes[0], data=kCrossData, numberHiddenLayers=genes[1], neurons=genes[2],
                     outs=(len(outs)+1), fft=fft, wave=wave, epochNumber=genes[3], showNetSetup=showNetSetup,
                     lossFunction=genes[4])
        evalTestData = array(kCrossData[2])
        if not fft:
            evalTestData = evalTestData.reshape((evalTestData.shape[1], evalTestData.shape[0], 1))
        evalAns = kCrossData[3]
        evalData = [evalTestData, evalAns]
        # Score [loss, accuracy, train time(in sec), evaluation time]
        score = newNN.modelScore(evalData)
        fitnessScore[i] = fitness(score)
        showNetSetup = False
    dNNinfo = [newNN, genes, (sum(fitnessScore)/len(fitnessScore)), score]
    logging.error("Network Done: \n Fitness score:{} \n Score last epoch: [TestLoss:{} TestAcc:{} TrainTime:{} "
                  "EvalTime:{}]  \n".format(dNNinfo[2], dNNinfo[3][0], dNNinfo[3][1], dNNinfo[3][2], dNNinfo[3][3]))
    print("Network Done: \n Fitness score:{} \n Score last epoch: [TestLoss:{} TestAcc:{} TrainTime:{} "
          "EvalTime:{}]  \n".format(dNNinfo[2], dNNinfo[3][0], dNNinfo[3][1], dNNinfo[3][2], dNNinfo[3][3]))
    return dNNinfo


# -------------- Genetic Algorithm --------------
class GA(object):
    """
    classdocs
    Creates a genetic algorithm for optimizing hyperparameters in dNN
    """
    def __init__(self, data, outs, popSize, genSize, mutationChance, nodes=14, fft=False, wave=False, maxMinVal=None):
        self.data = data
        self.dataSplit = None
        self.bestDynamicNeuralNetwork = None
        self.outs = outs
        self.genSize = genSize
        self.popSize = popSize
        self.mutationChance = mutationChance
        self.fft = fft
        self.wave = wave
        genes = randomGenes(maxMinVal)
        self.typeNN = genes[0]
        self.nodes = nodes
        self.maxMinVal = maxMinVal

    # dNNinfo [NN, genes, fitness score, score numbers]
    def run(self):
        self.dataSplit = dataSplit(self.data)
        dNNPop = []
        dNNChildren = self.createRandNN()
        for gen in range(1, self.genSize):
            dNNPop.extend(dNNChildren)
            dNNPop = sortFitness(dNNPop)
            logging.error("Gen: {}".format(gen))
            dNNPop = dNNPop[:max((int(len(dNNPop) / 5)), 2)]
            numberOfChildren = self.popSize - len(dNNPop)

            dNNPop[0][0].saveModel('topGen_model.h5')
            dNNPop[1][0].saveModel('second_model.h5')
            dNNChildren = []
            while len(dNNChildren) <= numberOfChildren:
                male = randint(0, len(dNNPop)-1)
                female = randint(0, len(dNNPop)-1)

                if male != female:
                    male = dNNPop[male]
                    female = dNNPop[female]
                    parents = [male, female]

                    dNNChildren.append(self.newdNNChild(parents))

        dNNPop.extend(dNNChildren)
        dNNPop = sortFitness(dNNPop)
        self.bestDynamicNeuralNetwork = dNNPop[0]
        self.bestDynamicNeuralNetwork[0].saveModel()

    # Returns the best neural network found by the genetic algorithm
    def getBestNN(self):
        return self.bestDynamicNeuralNetwork

    # Create new dNN based on two existing once
    # @input parents: list of parents to create new child from
    # @return new DynamicNeuralNet created from parent values and mutation
    def newdNNChild(self, parents):
        genes = [parents[randint(0, 1)][1][0], parents[randint(0, 1)][1][1],
                 [parents[randint(0, 1)][1][2][0], parents[randint(0, 1)][1][2][1], parents[randint(0, 1)][1][2][2],
                  parents[randint(0, 1)][1][2][3], parents[randint(0, 1)][1][2][4], parents[randint(0, 1)][1][2][5]],
                 parents[randint(0, 1)][1][3], parents[randint(0, 1)][1][4]]
        if randint(0, 100) > self.mutationChance:
            mutatedGene = randint(1, 4)
            mutationGenes = randomGenes(self.maxMinVal)
            if mutatedGene == 2:
                for x in range(0, randint(1, 5)):
                    genes[2][x] = mutationGenes[2][x]
            else:
                genes[mutatedGene] = mutationGenes[mutatedGene]
        dNNChild = kCrossValidate(genes, self.dataSplit, self.outs, self.nodes, self.fft, self.wave)
        return dNNChild

    # Creates neural nets with random genes
    # @return population of neural nets
    def createRandNN(self):
        dNNPop = []
        for x in range(self.popSize):
            genes = randomGenes(maxMinVal=self.maxMinVal)
            dNNPop.append(kCrossValidate(genes, self.dataSplit, self.outs, self.nodes, self.fft, self.wave))
            print('Network number {} of {} done.'.format((x+1), self.popSize))
        return dNNPop
