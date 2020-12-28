import numpy as np
import MultiLayerPerceptron as mlp
from activations import relu, sigmoid

np.random.seed(1)

testMLP = mlp.MultiLayerPerceptron(inputSize = 3,
                                   numHiddenLayers = 0,
                                   activationFunction = sigmoid,
                                   outputSize = 1,
                                   trainingIterations = 1)



X = np.array([[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]])

Y = np.array([[0,1,1,0]]).T


print(testMLP.weightsByLayer)
testMLP.weightsByLayer = 2 * testMLP.weightsByLayer[0] - 1 
print(testMLP.weightsByLayer)


Z = testMLP.train(X)
print(Z)
