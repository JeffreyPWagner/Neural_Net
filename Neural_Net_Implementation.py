import numpy as np
import math


def sigmoid(value):
    return 1 / (1 + math.exp(value))


class Neuron:
    def __init__(self, previousLayerLen):
        self.bias = np.random.random_sample()
        self.weights = np.random.random_sample(previousLayerLen)
        self.value = 0.0

    def updateValue(self, previousLayerValues):
        sumProduct = 0.0
        for i, value in enumerate(previousLayerValues):
            sumProduct += value * self.weights[i]
        self.value = sigmoid(sumProduct + self.bias)


class NeuralNet:
    def __init__(self, numInputs, numOutputs):
        # TODO updates layer and node count
        numLayers = numInputs
        neuronsPerLayer = numOutputs
        self.neurons = [[Neuron() for j in range(neuronsPerLayer)] for i in range(numLayers)]



