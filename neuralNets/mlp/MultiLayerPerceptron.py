'''
  Author:  Jordan Leiker
  Last Date Modified: 12/28/2020
 
  Description:
  This file contains the class MultiLayerPerceptrton and any additional support class/functions. 
  The MultiLayerPerceptron class implments MLP neural nets.
'''

import numpy as np
from activations import relu, sigmoid

class MultiLayerPerceptron(object):
    def __init__(self, inputSize, numHiddenLayers, numNeuronByHiddenLayer=[], activationFunction=relu, outputSize=1, trainingIterations=1):
        # High level parameters 
        self.inputSize = int(inputSize)
        self.numHiddenLayers = numHiddenLayers
        if self.numHiddenLayers > 0:
            if (len(self.neuronsByLayer) != self.numHiddenLayers):
                print("ERROR - input parameter numHiddenLayers must be the length of list numNeuronsByHiddenLayer")
                exit
            else:
                self.neuronsByLayer = numNeuronByHiddenLayer
        self.activation = activationFunction
        self.outputSize = int(outputSize)
        self.trainingIterations = trainingIterations
        
        # Neurel Net Setup
        ## NN has a list member for each layer, and each layer has all neurons stored in numpy array
        ## Weights / Neurons stored as:
        #### rows are weights per neuron
        #### columns are neurons per layer
        #### | w1_n1   w1_n2  ...  w1_nN | 
        #### | w2_n1   w2_n2  ...  w2_nN |
        #### | w3_n1   w3_n2  ...  w3_nN |
        if self.numHiddenLayers > 0: 
            ### layer with input connections
            self.weightsByLayer = [np.random.random((self.inputSize, self.numNeuronByHiddenLayer[0]))]
            ### all hidden layers
            for ii in range(numHiddenLayers-1):
                self.weightsByLayer.append(np.random.random((self.numNeuronByHiddenLayer[ii], self.numNeuronByHiddenLayer[ii+1]))) # neurons are initialized at random for now
            ### output layer
            self.weightsByLayer.append(np.random.random((self.outputSize, self.numNeuronByHiddenLayer[-1])))
        else:
            ### single layer
            self.weightsByLayer = [np.random.random((self.inputSize, self.outputSize))]

    def train(self, data):
        # Error check input data size
        if self.numHiddenLayers > 0: 
            if data.shape[1] != self.weightsByLayer[0].shape[0]:
                print("ERROR - data size must be equal to inputSize")
                exit()

            # Training iterations
            for iter in range(self.trainingIterations):
                # first layer
                weights = self.activation(data @ self.weightsByLayer[0]) 
                # the rest of the layers (if they exist)
                for ii in range(len(self.weightsByLayer) - 1):
                    weights = self.activation(weights @ self.weightsByLayer[ii+1])

        else:
            if data.shape[1] != self.weightsByLayer.shape[0]:
                print("ERROR - data size must be equal to inputSize")
                exit()

            # Training iterations
            for iter in range(self.trainingIterations):
                # only layer
                print(data)
                print(self.weightsByLayer)
                weights = self.activation(data @ self.weightsByLayer) 


        return weights


    def predict(self, data):
        pass


