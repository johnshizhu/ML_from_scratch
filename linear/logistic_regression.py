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
    
    def gradient_descent(self, X, Y_hat, W, b, learning_rate):
        '''
        Computes One step of gradient descent
        Inputs:
        X - Data
        Y_hat - truth labels
        W - weights
        b - bias
        learning_rate - learning rate alpha
        Outputs:
        W - Updated weights
        b - updated bias
        '''
        z = self.linear(X, W, b)
        Y = self.sigmoid(z)
        m = X.shape[0]
        
        # Calculate dW and db
        dW = (np.sum(np.dot(X, (Y - Y_hat)))) / m
        db = (np.sum(Y - Y_hat)) / m

        W = W - learning_rate * dW
        b = b - learning_rate * db

        return W, b
    
    def logistic_fit(self, X, Y_hat, learning_rate, iterations):
        '''
        Run logistic regression "iterations" number of times
        Inputs:
        X - Data
        Y_hat - True labels
        learning_rate - learning rate alpha
        iterations - number of iterations of gradient descent
        '''
        # Randomly initialize weights
        W = np.random.randn(X.shape[1])
        b = 0.5

        cost_list = []
        
        for i in range(iterations):
            cost = self.calculate_cost(X, Y_hat, W, b)
            cost_list.append(cost)

            W, b = self.gradient_descent(X, Y_hat, W, b, learning_rate)
        return W, b, cost_list
    



class linear_test():

    def testfunction():
        print("Test Success")