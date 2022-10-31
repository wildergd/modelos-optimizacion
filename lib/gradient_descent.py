import numpy as np
from typing import Callable
from .loss_functions import mse, mse_gradient, mse2

grad_fn = lambda theta, x, y: np.dot(x.T, (fn(theta, x) - y)) / x.shape[0]

def descenso_gradiente(
    fn: Callable,
    x0: float,
    alpha: float = 0.05,
    error: float = 0.0001,
    max_iter: int = 10000
):
    grad = lambda x, delta: (fn(x + delta) - fn(x - delta)) / (2 * delta)
    delta = 0.00001
    x = x0
    i = 1
    while i < max_iter:
        if grad(x, delta) < error:
            return x, i
        x = x - alpha * grad(x, delta)
        i += 1

    return x, max_iter

def descenso_gradiente_multi(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable = mse,
    grad_fn: Callable = grad_fn,
    cost_fn: Callable = mse_gradient,
    alpha: float = 0.005,
    error: float = 0.001,
    max_iter: int = 10000,
):
    rows, cols = x.shape
    theta = np.ones((cols + 1, 1))
    x0 = np.ones(rows)
    tmp_x = np.column_stack((x0, x))
    current_error = cost_fn(theta, tmp_x, y)
    errors = []
    thetas = [theta]

    for iter in range(max_iter):
        grad = grad_fn(theta, tmp_x, y)
        theta = theta - alpha * grad  # type: ignore
        thetas.append(theta)
        iter_error = cost_fn(theta, tmp_x, y)
        errors.append(abs(iter_error - current_error))
        current_error = iter_error
        if errors[-1] <= error:
            return { 'theta': theta, 'iters': iter + 1, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }
        
    return { 'theta': theta, 'iters': max_iter, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }


def descenso_gradiente_estocastico(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable = mse,
    grad_fn: Callable = mse_gradient,
    alpha: float = 0.005,
    error: float = 0.001,
    max_iter: int = 10000,
    sample_size: int = 8
):
    rows, cols = x.shape
    theta = np.ones((cols + 1, 1))
    x0 = np.ones(rows)
    tmp_x = np.column_stack((x0, x))
    current_error = fn(theta, tmp_x, y)
    errors = []
    thetas = [theta]

    for iter in range(max_iter):
        np.random.seed(iter)
        indexes = np.random.permutation(min(max(sample_size, 1), rows))
        smp_x = tmp_x[indexes]
        smp_y = y[indexes]
        grad = grad_fn(theta, smp_x, smp_y)
        theta = theta - alpha * grad
        thetas.append(theta)
        # current_error = abs(np.sum(grad_fn(theta, smp_x, smp_y)))
        iter_error = fn(theta, tmp_x, y)
        errors.append(abs(iter_error - current_error))
        if errors[-1] <= error:
            return { 'theta': theta, 'iters': iter + 1, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }
        current_error = iter_error
        
    return { 'theta': theta, 'iters': max_iter, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }

def descenso_gradiente_con_momentum(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable = mse,
    grad_fn: Callable = mse_gradient,
    alpha: float = 0.005,
    betha: float = 1,
    error: float = 0.001,
    max_iter: int = 10000,
    sample_size: int = 8
):
    rows, cols = x.shape
    theta = np.ones((cols + 1, 1))
    x0 = np.ones(rows)
    tmp_x = np.column_stack((x0, x))
    p = np.zeros(theta.shape)
    errors = []
    thetas = [theta]
    current_error = fn(theta, tmp_x, y)

    for iter in range(max_iter):
        np.random.seed(iter)
        indexes = np.random.permutation(min(max(sample_size, 1), rows))
        smp_x = tmp_x[indexes]
        smp_y = y[indexes]
        grad = grad_fn(theta, smp_x, smp_y)
        p = betha * p + (1 - betha)*grad
        theta = theta - alpha * p
        thetas.append(theta)
        iter_error = fn(theta, tmp_x, y)
        errors.append(abs(iter_error - current_error))
        if errors[-1] <= error:
            return { 'theta': theta, 'iters': iter + 1, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }
        current_error = iter_error
        
    return { 'theta': theta, 'iters': max_iter, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }

def descenso_gradiente_con_momentum_adam(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable = mse,
    grad_fn: Callable = mse_gradient,
    alpha: float = 0.005,
    eta1: float = 1,
    eta2: float = 1,
    error: float = 0.001,
    max_iter: int = 10000,
    sample_size: int = 8
):
    rows, cols = x.shape
    theta = np.ones((cols + 1, 1))
    x0 = np.ones(rows)
    tmp_x = np.column_stack((x0, x))
    p = np.zeros(theta.shape)
    speed = np.zeros(theta.shape)

    eta1_t = eta1
    eta2_t = eta2
    epsilon = np.finfo(float).eps

    current_error = fn(theta, tmp_x, y)
    errors = []
    thetas = [theta]

    for iter in range(max_iter):
        np.random.seed(iter)
        indexes = np.random.permutation(min(max(sample_size, 1), rows))
        smp_x = tmp_x[indexes]
        smp_y = y[indexes]
        grad = grad_fn(theta, smp_x, smp_y)
        p  = eta1_t * p + (1 - eta1_t) * grad
        speed = eta2_t * speed + (1 - eta2_t) * (grad**2)
        alpha_iter = alpha * np.sqrt(1 - eta2_t) / (1 - eta1_t)
        theta = theta - alpha_iter * p / (np.sqrt(speed) + epsilon)
        eta1_t *= eta1
        eta2_t *= eta2
        thetas.append(theta)
        iter_error = fn(theta, tmp_x, y)
        errors.append(abs(iter_error - current_error))
        if errors[-1] <= error:
            return { 'theta': theta, 'iters': iter + 1, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }
        current_error = iter_error
        
    return { 'theta': theta, 'iters': max_iter, 'current_error': errors[-1], 'errors': errors, 'thetas': thetas }

def descenso_gradiente_rmsprop(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable = mse,
    grad_fn: Callable = mse_gradient,
    step_size = 0.01,
    rho: float = 0.005,
    error: float = 0.001,
    max_iter: int = 10000,
):
    rows, cols = x.shape
    theta = np.ones((cols + 1, 1))
    x0 = np.ones(rows)
    tmp_x = np.column_stack((x0, x))
    current_error = np.Inf
    epsilon = np.finfo(float).eps

    errors = []
    thetas = []
    squared_grad_avg = np.zeros(theta.shape)

    for iter in range(max_iter):
        grad = grad_fn(theta, tmp_x, y)
        squared_grad = grad**2
        squared_grad_avg = (squared_grad_avg * rho) + (squared_grad * (1 - rho))
        alphas = step_size / (epsilon + np.sqrt(squared_grad_avg))
        theta = theta - alphas * grad
        current_error = abs(np.sum(grad_fn(theta, tmp_x, y)))
        errors.append(current_error)
        thetas.append(theta)
        if current_error <= error:
            return { 'theta': theta, 'iters': iter + 1, 'current_error': current_error, 'errors': errors, 'thetas': thetas }
        
    return { 'theta': theta, 'iters': max_iter, 'current_error': current_error, 'errors': errors, 'thetas': thetas }
