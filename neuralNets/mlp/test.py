import numpy as np
import numpy.matlib
import MultiLayerPerceptron as mlp
from activations import relu, sigmoid

np.random.seed(1)

testMLP = mlp.MultiLayerPerceptron(inputSize = 3,
                                   numHiddenLayers = 0,
                                   numNeuronByHiddenLayer = [2],
                                   activationFunction = sigmoid,
                                   outputSize = 1,
                                   trainingIterations = 1)



X = np.array([[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]])

Y = np.array([[0,1,1,0]]).T


print(testMLP.weightsByLayer)
testMLP.weightsByLayer[0] = 2 * testMLP.weightsByLayer[0] - 1
testMLP.biasByLayer[0] = 0 


Z = testMLP.feedforward(X)
print(Z)
