import numpy as np

class Linear_Regression():

    def shape_validation(self, X, Y_hat, W, b):
        '''
        Validates the matrix shapes (helper)
        '''
        if X.shape[1] != W.shape[0]:
            raise ValueError("X and W shape Mismatch")
        if Y_hat.shape[0] != X.shape[0]:
            raise ValueError("X and Y_hat shape Mismatch")
        return

    def calculate_cost(self, X, Y_hat, W, b,):
        '''
        Calculates the means squared error (MSE)
        Inputs
         - X:     Input Data
         - Y_hat: True labels
         - W:     Weights
         - b:     Bias
        '''
        self.shape_validation(X, Y_hat, W, b)

        Y = np.dot(X, W) + b
        cost = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])
        return cost

    def gradient_descent(self, X, Y_hat, W, b, learning_rate):
        '''
        Performs Gradient Descent on Weight and bias
        Inputs
         - X:     Input Data
         - Y_hat: True labels
         - W:     Weights
         - b:     Bias
         
        '''

        self.shape_validation(X, Y_hat, W, b)

        m = X.shape[0]
        Y = np.dot(X, W) + b

        errors = Y - Y_hat
        dW = np.sum(np.dot(errors, X.T)) / m
        db = np.sum(errors) / m 
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return W, b

    def predict(self, X, W, b):
        return np.dot(X, W) + b

    def fit_model(self, X, Y_hat, learning_rate, iterations):
        '''
        Performs Linear regression by gradient descent
        Input:
        - X: Input values
        - Y_hat: truth values
        - learning_rate: alpha
        '''
        m = X.shape[0]
        # Randomly Initialize weights
        W = np.random.randn(X.shape[1])
        b = 0.5
        
        # blank list to store cost
        cost_list = []

        for i in range(iterations):
            W, b = self.gradient_descent(X, Y_hat, W, b, learning_rate)
            cost = self.calculate_cost(X, Y_hat, W, b)
            cost_list.append(cost)

        return W, b, cost_list
    

    


class linear_test():

    def testfunction():
        print("Test Success")