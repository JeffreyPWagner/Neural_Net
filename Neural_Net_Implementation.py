import random
import math
import numpy as np

random.seed(1)


def relu(value):
    return np.maximum(value, 0)


def bipolarSigmoid(value):
    return 1 / (1 + math.exp(-value))


class Neuron:
    def __init__(self, previousLayerLen):
        self.bias = random.uniform(-0.05, 0.05)
        self.weights = []
        self.value = 0.0
        self.error = 0.0

        for i in range(0, previousLayerLen):
            randomFloat = random.uniform(-0.05, 0.05)
            self.weights.append(randomFloat)

    def updateValue(self, previousLayerValues):
        sumProduct = 0.0
        for i, value in enumerate(previousLayerValues):
            sumProduct += value * self.weights[i]
        self.value = bipolarSigmoid(sumProduct + self.bias)

    def updateError(self, nextLayerNeurons, neuronPosition):
        sumProduct = 0.0
        for neuron in nextLayerNeurons:
            sumProduct += neuron.error * neuron.weights[neuronPosition]
        self.error =self.value * (1 - self.value) * sumProduct

    def updateOutputError(self, targetValue):
        self.error = self.value * (1 - self.value) * (targetValue - self.value)

    def updateWeightsAndBias(self, learningFactor, previousLayerValues):
        for i, value in enumerate(previousLayerValues):
            self.weights[i] = self.weights[i] + learningFactor * self.error * value
        self.bias = self.bias + learningFactor * self.error * self.bias


class NeuralNet:
    def __init__(self, numOutputs, numLayers, neuronsPerLayer, learningFactor):
        self.neurons = [[Neuron(neuronsPerLayer) for j in range(neuronsPerLayer)] for i in range(numLayers)]
        self.outputs = [Neuron(neuronsPerLayer) for i in range(numOutputs)]
        self.learningFactor = learningFactor

    def feedForward(self, inputs):
        for neuron in self.neurons[0]:
            neuron.updateValue(inputs)
        for i in range(1, len(self.neurons)):
            previousLayerValues = []
            for neuron in self.neurons[i-1]:
                previousLayerValues.append(neuron.value)
            for neuron in self.neurons[i]:
                neuron.updateValue(previousLayerValues)
        lastHiddenLayerValues = []
        for neuron in self.neurons[len(self.neurons) - 1]:
            lastHiddenLayerValues.append(neuron.value)
        for neuron in self.outputs:
            neuron.updateValue(lastHiddenLayerValues)

    def backpropogateErrors(self, correctOutputs):
        for i, neuron in enumerate(self.outputs):
            neuron.updateOutputError(correctOutputs[i])
        for i, neuron in enumerate(self.neurons[len(self.neurons) - 1]):
            neuron.updateError(self.outputs, i)
        for i in range(len(self.neurons) - 2, -1, -1):
            for j, neuron in enumerate(self.neurons[i]):
                neuron.updateError(self.neurons[i + 1], j)

    def learnFromExample(self, inputs, correctOutputs):
        self.feedForward(inputs)
        self.backpropogateErrors(correctOutputs)
        for neuron in self.neurons[0]:
            neuron.updateWeightsAndBias(self.learningFactor, inputs)
        for i in range(1, len(self.neurons)):
            previousLayerValues = []
            for neuron in self.neurons[i-1]:
                previousLayerValues.append(neuron.value)
            for neuron in self.neurons[i]:
                neuron.updateWeightsAndBias(self.learningFactor, previousLayerValues)
        lastHiddenLayerValues = []
        for neuron in self.neurons[len(self.neurons) - 1]:
            lastHiddenLayerValues.append(neuron.value)
        for neuron in self.outputs:
            neuron.updateWeightsAndBias(self.learningFactor, lastHiddenLayerValues)

    def classify(self, inputs):
        self.feedForward(inputs)
        maxActivation = 0.0
        indexOfMax = -1
        for i, neuron in enumerate(self.outputs):
            if neuron.value > maxActivation:
                maxActivation = neuron.value
                indexOfMax = i
        return indexOfMax


targetClasses = set()
targetClassList = list()
trainingExamples = list()
trainingExampleAnswers = list()
testExamples = list()
testExampleAnswers = list()


def normalize(value, min, max):
    return (value - min) / (max - min)


trainingFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\Neural_Net\digits-training.data", "r")
for row in trainingFile:
    stringExample = row.split()
    example = [float(num) for num in stringExample]
    targetClass = example[len(example) - 1]
    trainingExampleAnswers.append(targetClass)
    targetClasses.add(targetClass)
    del example[len(example) - 1]
    newExample = [normalize(value, 0, 9) for value in example]
    trainingExamples.append(newExample)
trainingFile.close()

for target in targetClasses:
    targetClassList.append(target)
targetClassList.sort()

testFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\Neural_Net\digits-test.data", "r")
for row in testFile:
    stringExample = row.split()
    example = [float(num) for num in stringExample]
    targetClass = example[len(example) - 1]
    testExampleAnswers.append(targetClass)
    del example[len(example) - 1]
    newExample = [normalize(value, 0, 9) for value in example]
    testExamples.append(newExample)
testFile.close()

myNeuralNet = NeuralNet(len(targetClassList), 2, len(trainingExamples[0]), 0.1)

for n in range(0, 30):

    for k, example in enumerate(trainingExamples):
        answers = [0.0] * len(targetClassList)
        answers[targetClassList.index(trainingExampleAnswers[k])] = 1.0
        myNeuralNet.learnFromExample(example, answers)

    correctCount = 0

    for m, example in enumerate(testExamples):
        guess = myNeuralNet.classify(example)
        if guess == targetClassList.index(testExampleAnswers[m]):
            correctCount += 1

    print(correctCount / len(testExamples))






























































