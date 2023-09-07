'''
Implementation of the Layer object
A layer is a collection of Node objects that can take in input from a previous layer and output a list of activations

'''
from Node import NN_node
class Layer():

    def __init__(self, node_count, activation):
        # list of nodes in the layer
        self.nodes = []
        
        # create all nodes and put into list
        