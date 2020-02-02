from logistic_regression import LogisticRegression
import numpy as np
import pytest


def test_loss():
    test_x = [np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])]
    test_y = [np.array([1, 0, 1])]
    test_beta = [np.array([1.2, 3.4, 2.5])]
    expected = [-22.599999]
    lr_model = LogisticRegression()
    for idx in range(len(test_x)):
        actual = lr_model._loss(test_x[idx],
                                test_y[idx],
                                test_beta[idx])
        assert pytest.approx(expected[idx], 1e-06) == actual


def test_gradient():
    test_x = [np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])]
    test_y = [np.array([1, 0, 1])]
    test_beta = [np.array([1.2, 3.4, 2.5])]
    expected = [[1.99999981, 2.99999963, 3.99999944]]
    lr_model = LogisticRegression()
    for idx in range(len(test_x)):
        actual = lr_model._gradient(test_x[idx],
                                    test_y[idx],
                                    test_beta[idx])
        assert pytest.approx(expected[idx], 1e-06) == actual


def test_integ_fit():
    test_x = [np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])]
    test_y = [np.array([1, 0, 1])]
    expected = [np.array([0.01328192, 0.06222676, 0.1111716])]
    lr_model = LogisticRegression()
    for idx in range(len(test_x)):
        lr_model.fit(test_x[idx], test_y[idx])
        assert pytest.approx(expected[idx], 1e-06) == lr_model.parameters
