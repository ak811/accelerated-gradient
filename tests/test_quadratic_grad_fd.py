from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.objectives.quadratic import Quadratic

def test_quadratic_grad_matches_fd() -> None:
    rng = np.random.default_rng(0)
    n = 10
    M = rng.standard_normal((n, n))
    A = M.T @ M + 1e-2 * np.eye(n)
    b = rng.standard_normal((n,))
    obj = Quadratic(A=A, b=b)

    x = rng.standard_normal((n,))
    g = obj.grad(x)

    eps = 1e-6
    gfd = np.zeros_like(g)
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        gfd[i] = (obj.f(x + eps * e) - obj.f(x - eps * e)) / (2 * eps)

    assert np.allclose(g, gfd, atol=1e-5, rtol=1e-5)
