from typing import Callable
import numpy as np

def mse(theta:np.ndarray, x:np.ndarray, y: np.ndarray) -> float:
    y1 = np.dot(x, theta)
    return np.sum((y1 - y) ** 2) / y.shape[0]

def mse_gradient(theta:np.ndarray, x:np.ndarray, y: np.ndarray) -> np.ndarray:
    y1 = np.dot(x, theta)
    return np.dot(np.transpose(x), (y1 - y)) * (2 / y.shape[0])

def mse2(h: Callable, theta:np.ndarray, x:np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(np.dot(np.transpose(x), (h(x, theta) - y)) ** 2) / (2 * y.shape[0])
