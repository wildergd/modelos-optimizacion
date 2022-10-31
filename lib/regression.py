from typing import Callable
import numpy as np

def linear_regression(x: np.ndarray, y: np.ndarray):
    tmp_x = np.column_stack((np.ones(x.shape[0]), x))
    calc_theta = lambda x, y: np.linalg.inv(x.T @ x) @ x.T @ y
    return calc_theta(tmp_x, y)

def linear_regression_gd_quad(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable,
    grad_fn: Callable,
    error:float = 0.001,
    max_iters=10000
):
    tmp_x = np.column_stack((np.ones(x.shape[0]), x))
    return

def logistic_regression():
    return