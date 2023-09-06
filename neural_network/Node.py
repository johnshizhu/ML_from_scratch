'''
Implementation of the Node object
A node can take in a matrix/list of activations from a previous layer and output a single activaiton. 
'''
import numpy as np

class NN_Node():

    def __init__(self, activation):
        self.activation = activation
        self.weight = np.random.randn()
        

    