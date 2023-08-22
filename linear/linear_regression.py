import numpy as np

class Linear_Regression():
    
    def calculate_cost(X, Y_hat, W, b):
        '''
        Calculates the means squared error (MSE)
        Inputs
         - X:     Input Data
         - Y_hat: True labels
         - W:     Weights
         - b:     Bias
        '''
        Y = np.dot(X, W) + b
        return np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])

    def linear_regression():
        return
    
    def predict():
        return
    
    


class linear_test():

    def testfunction():
        print("Test Success")