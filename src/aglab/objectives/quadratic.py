from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Quadratic:
    """
    f(x) = 0.5 x^T A x + b^T x
    grad(x) = A x + b

    A should be symmetric PSD/PD for standard convex analysis.
    """
    A: np.ndarray
    b: np.ndarray

    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        return float(0.5 * x.T @ self.A @ x + self.b.T @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return self.A @ x + self.b

    def minimizer(self) -> np.ndarray:
        """
        If A is SPD, returns the unique minimizer x* = -A^{-1} b.
        If A is singular, returns a least-squares stationary point (not necessarily a minimizer).
        """
        x, *_ = np.linalg.lstsq(self.A, -self.b, rcond=None)
        return x


def make_symmetric_psd_with_spectrum(n: int, mu: float, L: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric A with eigenvalues in [mu, L]. If mu>0, A is SPD.

    Returns:
      A: (n,n) symmetric
      eigs: (n,) eigenvalues used
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    U, _, _ = np.linalg.svd(M, full_matrices=False)

    d = rng.random((n,))
    dsrt = np.sort(d)[::-1]
    D = 10 ** dsrt
    Dnorm = (D - D.min()) / (D.max() - D.min() + 1e-15)

    eigs = mu + Dnorm * (L - mu)
    A = U @ np.diag(eigs) @ U.T
    return A, eigs
