'''
Implementation of the Node object
A node can take in a matrix/list of activations from a previous layer and output a single activaiton. 
'''
import numpy as np

class NN_Node():

    def __init__(self, input, activation, m, n):
        self.activation_name = activation
        self.input = input
        self.m = m
        self.n = n
        self.weight = np.random.randn(m, n) * 0.01 # initialize weight matrix in correct shape
        self.bias = np.zeros((m, 1))
        self.activation_output = None
    
    def calc_linear(self):
        '''
        Helper function that calculates the current linear output of the node. 
        '''
        mult = np.dot(self.input, self.weight)
        linear = mult + self.bias
        return linear

    def calc_activation(self):
        '''
        Calculates the activation based on the selected activation name
        '''
        if self.activation_name == 'sigmoid':
            return 1/(1 + np.exp(-self.calc_linear()))

        if self.activation_name ==  'linear':
            return self.calc_linear()
        
        if self.activation_name == 'ReLU':
            if self.calc_linear() > 0:
                return self.calc_linear()
            else:
                return 0
    
        

    