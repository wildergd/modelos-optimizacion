
from math import ceil, log, sqrt
from time import time
from typing import Callable
from random import randint
import numpy as np

import contextlib
with contextlib.redirect_stdout(None):
    np.seterr(divide = 'ignore')

# global helpers
switch_values = lambda a, b: (b, a)

# sección dorada
def seccion_dorada(fn: Callable, xl: float, xu: float, error: float = 0.000001, max_iters: int = 100, iter:int = 1) -> tuple:
    if xu < xl:
        xu, xl = switch_values(xu, xl)

    R = (sqrt(5) - 1.0) / 2.0

    b = R * (xu - xl)
    x1 = xu - b
    x2 = xl + b

    if fn(x2) > fn(x1):
        xl = x1
    else:
        xu = x2

    if abs(fn(x1) - fn(x2)) > error and max_iters > iter:
        return seccion_dorada(fn, xu, xl, error, max_iters, iter + 1)

    return x1, iter

# método de la secante
def secante(fn: Callable, p1, p2, error: float = 0.000001, max_iters: int = 100, iter: int = 1) -> tuple:
    x = p2 - fn(p2)*(p2 - p1)/(fn(p2) - fn(p1))

    if abs(fn(x)) <= error or max_iters == iter:
        return x, iter

    return secante(fn, p1, x, error, max_iters, iter + 1)

# método de la bisección
def bisec(fn: Callable, interval: tuple, error: float = 0.000000001, r: float = 0, max_iters:int = 100, iter:int = 1) -> tuple:  # type: ignore
    a, b = interval
    m = (a + b) / 2

    if max_iters == iter:
        return m, iter

    fa = fn(a)
    fb = fn(b)
    fm = fn(m)

    if fa == 0: 
        return a
    if fb == 0:
        return b

    if fa * fb < 0:
        if abs(fm) <= error:
            return m, iter

        if fm * fa < 0:
            return bisec(fn, (a, m), error, m, max_iters, iter + 1)

        return bisec(fn, (m, b), error, m, max_iters, iter + 1)

    raise Exception('Bisection methond cannot be applied')

# método de Newton-Raphson
def newton_raphson(fn: Callable, x0: float, error: float = 0.000001, max_iters:int = 100, iter: int = 1) -> tuple:
    if fn(x0) == 0: 
        return x0, iter

    df = lambda x, delta: (fn(x + delta) - fn(x - delta)) / (2 * delta)

    df_x0 = df(x0, error)

    if df_x0 == 0:
        raise Exception(f'Root cannot be found. Null derivate at x0 = {x0}')
    
    delta = fn(x0) / df_x0

    x = x0 - delta

    if abs(fn(x)) <= error or abs(delta) < error or max_iters == iter:
        return x, iter

    return newton_raphson(fn, x, error, max_iters, iter + 1)

# método polinomial cúbico
def polinomial_cubico(fn: Callable, error: float = 0.000001, max_iters:int = 100, iter: int = 1) -> tuple:
    return ()