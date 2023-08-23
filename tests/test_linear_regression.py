import unittest
import sys
import numpy as np
sys.path.append('../ML_from_scratch')

from linear.linear_regression import linear_test
from linear.linear_regression import Linear_Regression

class TestLinearRegression(unittest.TestCase):
    
    # Testing basic calculation of cost. 
    def test_calculate_cost(self):

        X = np.array([[1, 2], [3, 4]])
        Y_hat = np.array([1, 2]) 
        W = np.array([[1], [2]])
        b = np.array([1])

        lr = Linear_Regression()
        cost = lr.calculate_cost(X, Y_hat, W, b)
        expected_cost = 65.5
        self.assertEqual(cost, expected_cost)

    # Test case 2
    def test_predict(self):
        X = np.array([[1,2,3], [4,5,6], [7,8,9]])
        Y = np.array([[1,2,1], [2,4,6], [7,2,5]])

        lr = Linear_Regression()

        result = lr.predict(X, Y, 0)
        expected = np.array([[26, 16, 28], [56, 40, 64], [86, 64, 100]])

        np.testing.assert_array_equal(result, expected)

    # Test case 3


if __name__ == '__main__':
    unittest.main()

linear_test.testfunction()