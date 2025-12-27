from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PiecewiseStronglyConvex1D:
    """
    f(x) = 25 x^2                   if x < 1
           x^2 + 48x - 24           if 1 <= x <= 2
           25x^2 - 48x + 72         if x > 2

    Vectorized implementation.
    """
    def f(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return np.where(
            x < 1.0,
            25.0 * x**2,
            np.where(
                x <= 2.0,
                x**2 + 48.0 * x - 24.0,
                25.0 * x**2 - 48.0 * x + 72.0,
            ),
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return np.where(
            x < 1.0,
            50.0 * x,
            np.where(
                x <= 2.0,
                2.0 * x + 48.0,
                50.0 * x - 48.0,
            ),
        )
