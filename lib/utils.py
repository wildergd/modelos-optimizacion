from typing import Callable
from time import time
import numpy as np
import matplotlib.pyplot as plt

def plot_method(fn: Callable, method: Callable, interval, **kwargs):
    start = time()
    r, iter = method(fn, **kwargs)
    end = time()

    elapsed = (end - start) * 1000
    
    x = np.arange(interval.start, interval.stop, interval.step) if type(interval) == 'range' else np.linspace(interval[0], interval[1], 100)
    y = fn(x)
    yr = fn(r)
    plt.plot(x, y)
    plt.scatter(r, yr, color = "black")
    plt.text(r, yr + 1, f'Found f({r:.2f}) = 0 in {elapsed:.2f}ms with {iter} iters')
    plt.show()

def print_result(method_name: str, result: dict = {}) -> None:
    if not result: 
        return None
    theta_v = result['theta']
    iters = result['iters']
    error = result['current_error']
    print(f'Method: {method_name} [iters: {iters}, thetas: {np.array(theta_v).reshape(theta_v.shape[0])}, error: {error:.6f}]')