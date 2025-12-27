from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class History:
    xs: np.ndarray
    fvals: np.ndarray
    n_iter: int

def nesterov_strongly_convex(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha: float,
    beta: float,
    max_iter: int,
    stop: Callable[[int, np.ndarray, float], bool],
) -> History:
    x = np.asarray(x0, float).copy()
    x_prev = x.copy()

    xs = [x.copy()]
    fvals = [float(np.asarray(f(x)))]

    k = 0
    while k < max_iter and not stop(k, x, fvals[-1]):
        y = x + beta * (x - x_prev)
        x_next = y - alpha * grad(y)

        x_prev = x
        x = x_next

        xs.append(x.copy())
        fvals.append(float(np.asarray(f(x))))
        k += 1

    return History(xs=np.asarray(xs), fvals=np.asarray(fvals), n_iter=k)

def nesterov_convex(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    stop: Callable[[int, np.ndarray, float], bool],
) -> History:
    x = np.asarray(x0, float).copy()
    x_prev = x.copy()

    xs = [x.copy()]
    fvals = [float(np.asarray(f(x)))]

    k = 0
    while k < max_iter and not stop(k, x, fvals[-1]):
        beta_k = (k - 1.0) / (k + 2.0) if k >= 1 else 0.0
        y = x + beta_k * (x - x_prev)
        x_next = y - alpha * grad(y)

        x_prev = x
        x = x_next

        xs.append(x.copy())
        fvals.append(float(np.asarray(f(x))))
        k += 1

    return History(xs=np.asarray(xs), fvals=np.asarray(fvals), n_iter=k)
