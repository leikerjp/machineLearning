'''
  Author:  Jordan Leiker
  Last Date Modified: 12/28/2020
 
  Description:
  This file contains activation functions for use in custom neural net library neurons.
'''

import numpy as np

def relu(x):
    return np.max(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
