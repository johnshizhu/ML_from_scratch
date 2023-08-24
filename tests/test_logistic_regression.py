import unittest
import sys
import numpy as np
sys.path.append('../ML_from_scratch')

from linear.logistic_regression import linear_test
from linear.logistic_regression import Logistic_Regression

class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        test1_x = 15
        test2_x = 0
        test3_x = -15
        test4_x = 1
        test5_x = -1
        test6_x = 3
        test7_x = -3
        lr = Logistic_Regression()
        test1_res = lr.sigmoid(test1_x)
        test2_res = lr.sigmoid(test2_x)
        test3_res = lr.sigmoid(test3_x)
        test4_res = lr.sigmoid(test4_x)
        test5_res = lr.sigmoid(test5_x)
        test6_res = lr.sigmoid(test6_x)
        test7_res = lr.sigmoid(test7_x)

        self.assertAlmostEqual(test1_res, 0.9999996940977730743753)
        self.assertAlmostEqual(test2_res, 0.5)
        self.assertAlmostEqual(test3_res, 3.059022269256247251468E-7)
        self.assertAlmostEqual(test4_res, 0.7310585786300048792512)
        self.assertAlmostEqual(test5_res, 0.2689414213699951207488)
        self.assertAlmostEqual(test6_res, 0.9525741268224332191212)
        self.assertAlmostEqual(test7_res, 0.04742587317756678087885)

        return
    
    def test_linear(self):


        return
    
    def test_calculate_cost(self):


        return
    
    def test_gradient_descent(self):


        return


if __name__ == '__main__':
    unittest.main()


linear_test.testfunction()