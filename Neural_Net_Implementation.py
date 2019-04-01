import random
import math
from sklearn.metrics import confusion_matrix
import pandas as pd

random.seed(0)

# set of possible target classes
targetClasses = set()

# list of possible target classes
targetClassList = list()

# list of unprocessed training examples
rawTrainingExamples = list()

# list of processed training examples
trainingExamples = list()

# list of correct outputs for training examples
trainingExampleAnswers = list()

# list of unprocessed test examples
rawTestExamples = list()

# list of processed test examples
testExamples = list()

# list of correct outputs for test examples
testExampleAnswers = list()

# set of possible input values
inputValuesSet = set()

# list of possible inputs values
inputValuesList = list()

# dictionary with possible input values as keys, assigned to new numeric values
inputValuesDict = dict()

# list of classifications for test examples
guessList = list()

# count of correctly classified test examples
correctCount = 0

# average mean squared error for the epoch
error = 0.0


# checks if an input is a number
# this method copied from https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def checkNumber(inputString):
    try:
        int(inputString)
        return True
    except ValueError:
        return False


# sigmoid activation function for neurons
def sigmoid(val):
    return 1 / (1 + math.exp(-val))


# normalizes input values
def normalize(val, minValue, maxValue):
    return (val - minValue) / (maxValue - minValue)


# a single neuron in the hidden or output layers of a neural network
class Neuron:

    # instantiates new neuron with random bias and weights
    # param previousLayerLen: how many inputs or neurons are in the preceding layer
    def __init__(self, previousLayerLen):
        self.bias = random.uniform(-0.1, 0.1)
        self.weights = []
        self.value = 0.0
        self.error = 0.0

        for i in range(0, previousLayerLen):
            randomFloat = random.uniform(-0.1, 0.1)
            self.weights.append(randomFloat)

    # uses activation function to update this neuron's stored value
    # param previousLayerValues: the values of the inputs or neurons in the preceding layer
    def updateValue(self, previousLayerValues):
        sumProduct = 0.0
        for z, neuronVal in enumerate(previousLayerValues):
            sumProduct += neuronVal * self.weights[z]
        self.value = sigmoid(sumProduct + self.bias)

    # uses the derivative of the sigmoid activation function to update this neuron's stored error value
    # param nextLayerNeurons: the neurons in the following layer
    # param neuronPosition: the position of this neuron in its layer
    def updateError(self, nextLayerNeurons, neuronPosition):
        sumProduct = 0.0
        for neuron in nextLayerNeurons:
            sumProduct += neuron.error * neuron.weights[neuronPosition]
        self.error = self.value * (1 - self.value) * sumProduct

    # uses the derivative of the sigmoid activation function to update this output neuron's stored error value
    # param targetValue: the target output value for this neuron
    def updateOutputError(self, targetValue):
        self.error = self.value * (1 - self.value) * (targetValue - self.value)

    # updates the weights and bias of this neuron
    # param learningFactor: the learning factor of the neural net
    # param previousLayerValues: the values of the inputs or neurons in the preceding layer
    def updateWeightsAndBias(self, learningFactor, previousLayerValues):
        for i, val in enumerate(previousLayerValues):
            self.weights[i] = self.weights[i] + learningFactor * self.error * val
        self.bias = self.bias + learningFactor * self.error * self.bias


# a neural network made up of an input layer, output layer, and at least 1 hidden layer
class NeuralNet:

    # instantiates a new neural network
    # param numInputs: the number of input values
    # param numOutputs: the number of output values
    # param numLayers: the number of hidden layers
    # param neuronsPerLayer: the number of neurons in the hidden layers
    # param learningFactor: the learning factor of the network
    # param neuronStepDown: the number of neurons to subtract from each hidden layer following the first
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

    # feeds an example forward through the network and updates the neurons' values
    # param inputs: a list of inputs representing an example
    def feedForward(self, inputs):

        # update values of first hidden layer using input values
        for neuron in self.neurons[0]:
            neuron.updateValue(inputs)

        # update values of each subsequent hidden layer using previous layer values
        for i in range(1, len(self.neurons)):
            previousLayerValues = []
            for neuron in self.neurons[i-1]:
                previousLayerValues.append(neuron.value)
            for neuron in self.neurons[i]:
                neuron.updateValue(previousLayerValues)

        # update values of output neurons using values of the last hidden layer
        lastHiddenLayerValues = []
        for neuron in self.neurons[len(self.neurons) - 1]:
            lastHiddenLayerValues.append(neuron.value)
        for neuron in self.outputs:
            neuron.updateValue(lastHiddenLayerValues)

    # backpropogates errors through the network, updating each neurons' error value
    # param correctOutputs: the target outputs for the output layer
    def backpropogateErrors(self, correctOutputs):

        # update output layer error values
        for i, neuron in enumerate(self.outputs):
            neuron.updateOutputError(correctOutputs[i])

        # update final hidden layer error values
        for i, neuron in enumerate(self.neurons[len(self.neurons) - 1]):
            neuron.updateError(self.outputs, i)

        # update remaining hidden layer error values
        for i in range(len(self.neurons) - 2, -1, -1):
            for j, neuron in enumerate(self.neurons[i]):
                neuron.updateError(self.neurons[i + 1], j)

    # calculates total error for the network using sum of squared errors
    # param correctOutputs: the target outputs for the output layer
    def calculateError(self, correctOutputs):
        sumOfSquaredError = 0
        for i, target in enumerate(correctOutputs):
            sumOfSquaredError += (target - self.outputs[i].value) ** 2
        return sumOfSquaredError

    # updates the weights and biases of all neurons according to a training example
    # param inputs: a list of inputs representing an example
    # param correctOutputs: the target outputs for the output layer
    def learnFromExample(self, inputs, correctOutputs):
        self.feedForward(inputs)
        self.backpropogateErrors(correctOutputs)

        # update first hidden layer weights and biases according to input values
        for neuron in self.neurons[0]:
            neuron.updateWeightsAndBias(self.learningFactor, inputs)

        # update subsequent hidden layers' weights and biases according to previous layer values
        for i in range(1, len(self.neurons)):
            previousLayerValues = []
            for neuron in self.neurons[i-1]:
                previousLayerValues.append(neuron.value)
            for neuron in self.neurons[i]:
                neuron.updateWeightsAndBias(self.learningFactor, previousLayerValues)

        # update output layer weights and biases according to final hidden layer values
        lastHiddenLayerValues = []
        for neuron in self.neurons[len(self.neurons) - 1]:
            lastHiddenLayerValues.append(neuron.value)
        for neuron in self.outputs:
            neuron.updateWeightsAndBias(self.learningFactor, lastHiddenLayerValues)

    # classifies an example by returning the location of the highest activation output neuron
    # param inputs: a list of inputs representing an example
    def classify(self, inputs):
        self.feedForward(inputs)
        maxActivation = 0.0
        indexOfMax = -1
        for i, neuron in enumerate(self.outputs):
            if neuron.value > maxActivation:
                maxActivation = neuron.value
                indexOfMax = i
        return indexOfMax


# read the training data from file and close, make data numeric if possible
trainingFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\Neural_Net\digits-training.data", "r")
for row in trainingFile:
    example = row.split()
    targetClass = example[len(example) - 1]
    trainingExampleAnswers.append(targetClass)
    targetClasses.add(targetClass)
    del example[len(example) - 1]
    if checkNumber(example[0]):
        example = [float(num) for num in example]
    for value in example:
        inputValuesSet.add(value)
    rawTrainingExamples.append(example)
trainingFile.close()

# read the test data from file and close, make data numeric if possible
testFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\Neural_Net\digits-test.data", "r")
for row in testFile:
    example = row.split()
    targetClass = example[len(example) - 1]
    testExampleAnswers.append(targetClass)
    del example[len(example) - 1]
    if checkNumber(example[0]):
        example = [float(num) for num in example]
    for value in example:
        inputValuesSet.add(value)
    rawTestExamples.append(example)
testFile.close()

# sort target class list so they correspond to output layer neuron locations
for target in targetClasses:
    targetClassList.append(target)
targetClassList.sort()

# move input values from set to list and sort them
for value in inputValuesSet:
    inputValuesList.append(value)
inputValuesList.sort()

# assign input values to a corresponding integer, this allows both numeric and non-numeric inputs
for p, value in enumerate(inputValuesList):
    inputValuesDict[value] = p

# normalize training example inputs
for example in rawTrainingExamples:
    numericExample = [inputValuesDict[value] for value in example]
    normalizedExample = [normalize(value, 0, 9) for value in numericExample]
    trainingExamples.append(normalizedExample)

# set the index of the furthest validation example to be taken from the training set
validationIndex = int(len(trainingExamples) * 0.10)

# create a list of validation examples and remove them from the training set
validationExamples = trainingExamples[0:validationIndex]
del trainingExamples[0:validationIndex]
validationExampleAnswers = trainingExampleAnswers[0:validationIndex]
del trainingExampleAnswers[0:validationIndex]

# normalize test example inputs
for example in rawTestExamples:
    numericExample = [inputValuesDict[value] for value in example]
    normalizedExample = [normalize(value, 0, 9) for value in numericExample]
    testExamples.append(normalizedExample)

# calculate number of hidden layer neurons to use based on data
myNetNeurons = int(2 / 3 * (len(trainingExamples[0]) + len(targetClassList)))

# create a neural network according to data and chosen hyperparameters
myNeuralNet = NeuralNet(len(trainingExamples[0]), len(targetClassList), 1, myNetNeurons, 0.3, 0)

# train the neural network using training examples, set loop to desired number of epochs
previousValAcc = 0.90
for n in range(0, 30):
    error = 0.0
    for k, example in enumerate(trainingExamples):
        answers = [0.0] * len(targetClassList)
        answers[targetClassList.index(trainingExampleAnswers[k])] = 1.0
        myNeuralNet.learnFromExample(example, answers)
        error += myNeuralNet.calculateError(answers)
    error /= len(trainingExamples)
    print('MSE ', error)

    validationCorrectCount = 0
    for i, example in enumerate(validationExamples):
        guess = myNeuralNet.classify(example)
        if validationExampleAnswers[i] in targetClassList:
            if guess == targetClassList.index(validationExampleAnswers[i]):
                validationCorrectCount += 1
    validationAccuracy = validationCorrectCount / len(validationExamples)
    print('Validation Accuracy: ', validationAccuracy)
    if validationAccuracy > 0.965:
        break
    if validationAccuracy >= previousValAcc:
        myNeuralNet.learningFactor += 0.05
    else:
        myNeuralNet.learningFactor *= 0.7
    previousValAcc = validationAccuracy

# test the trained neural network using the test data
for m, example in enumerate(testExamples):
    guess = myNeuralNet.classify(example)
    guessList.append(targetClassList[guess])
    if testExampleAnswers[m] in targetClassList:
        if guess == targetClassList.index(testExampleAnswers[m]):
            correctCount += 1

# create a confusion matrix, print it, and write it to file
cm = confusion_matrix(testExampleAnswers, guessList, labels=targetClassList)
cmDataFrame = pd.DataFrame(cm, index=targetClassList, columns=targetClassList)
print(cmDataFrame)
cmDataFrame.to_csv('confusionMatrix.csv')

# print accuracy and write it to file along with MSE and validation set accuracy
print(correctCount / len(testExampleAnswers))
with open("results.txt", 'w') as f:
    f.write('Accuracy: %r' % "{:.2%}".format(correctCount / len(testExampleAnswers)) + '\n\n')
    f.write('Final Validation Accuracy: %r' % "{:.2%}".format(previousValAcc) + '\n\n')
    f.write('Final Average Mean Squared Error: %r' % "{:.2%}".format(error) + '\n\n')
