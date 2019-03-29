import random
import math

# todo remove seed setter
random.seed(0)

targetClasses = set()
targetClassList = list()
trainingExamples = list()
trainingExampleAnswers = list()
testExamples = list()
testExampleAnswers = list()


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def normalize(value, minValue, maxValue):
    return (value - minValue) / (maxValue - minValue)


class Neuron:
    def __init__(self, previousLayerLen):
        self.bias = random.uniform(-0.1, 0.1)
        self.weights = []
        self.value = 0.0
        self.error = 0.0

        for i in range(0, previousLayerLen):
            randomFloat = random.uniform(-0.1, 0.1)
            self.weights.append(randomFloat)

    def updateValue(self, previousLayerValues):
        sumProduct = 0.0
        for i, value in enumerate(previousLayerValues):
            sumProduct += value * self.weights[i]
        self.value = sigmoid(sumProduct + self.bias)

    def updateError(self, nextLayerNeurons, neuronPosition):
        sumProduct = 0.0
        for neuron in nextLayerNeurons:
            sumProduct += neuron.error * neuron.weights[neuronPosition]
        self.error = self.value * (1 - self.value) * sumProduct

    def updateOutputError(self, targetValue):
        self.error = self.value * (1 - self.value) * (targetValue - self.value)

    def updateWeightsAndBias(self, learningFactor, previousLayerValues):
        for i, value in enumerate(previousLayerValues):
            self.weights[i] = self.weights[i] + learningFactor * self.error * value
        self.bias = self.bias + learningFactor * self.error * self.bias


class NeuralNet:
    def __init__(self, numInputs, numOutputs, numLayers, neuronsPerLayer, learningFactor, neuronStepDown):
        self.neurons = []
        self.outputs = []
        self.learningFactor = learningFactor

        for i in range(0, numLayers):
            self.neurons.append([])
            for j in range(0, neuronsPerLayer - (i * neuronStepDown)):
                if i == 0:
                    self.neurons[i].append(Neuron(numInputs))
                else:
                    self.neurons[i].append(Neuron(neuronsPerLayer - ((i - 1) * neuronStepDown)))

        for i in range(0, numOutputs):
            self.outputs.append(Neuron(len(self.neurons[len(self.neurons) - 1])))

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

myNetNeurons = int(2 / 3 * (len(trainingExamples[0]) + len(targetClassList)))
myNeuralNet = NeuralNet(len(trainingExamples[0]), len(targetClassList), 1, myNetNeurons, 1, 0)

for n in range(0, 1000):
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






























































