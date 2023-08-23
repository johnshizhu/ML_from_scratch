import numpy as np

class Logistic_Regression():

    def sigmoid(self, z):
        '''
        Calculates the sigmoid activation based on z (linear output)
        Input:
        z - Linear output
        Out:
        sig_out - sigmoid activation
        '''
        denom = 1 + np.exp(-z)
        sig_out = 1 / denom
        return sig_out
    
    def linear(self, X, W, b):
        '''
        Calculates the linear output based on weights and bias
        Input:
        X - Data values
        W - Weights
        b - bias
        Output:
        linear_out: Linear output
        '''
        linear_out = np.dot(X, W) + b
        return linear_out
    
    def calculate_cost(self, X, Y_hat, W, b):
        '''
        Calculate the cost based on inputs, truth and weights/bias
        Input:
        X - Data values
        Y_hat - Truth values
        W - Weights
        b - bias
        Output:
        cost - cross-entropy loss
        '''
        m = X.shape[0]
        z = self.linear(X, W, b)
        Y = self.sigmoid(z)
        
        cost_left = np.dot(-Y_hat, np.log(Y))
        cost_right = np.dot((1-Y_hat), np.log(1 - Y))
        cost = cost_left - cost_right
        cost = cost / m
        return cost
    
    def gradient_descent():

        return
    



class linear_test():

    def testfunction():
        print("Test Success")