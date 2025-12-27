from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.objectives.quadratic import Quadratic
from aglab.optim.gd import gradient_descent_fixed

def test_gd_decreases_for_alpha_1_over_L() -> None:
    rng = np.random.default_rng(1)
    n = 20
    M = rng.standard_normal((n, n))
    A = M.T @ M + 1e-1 * np.eye(n)
    b = rng.standard_normal((n,))
    obj = Quadratic(A=A, b=b)

    L = float(np.linalg.eigvalsh(A).max())
    alpha = 1.0 / L

    x0 = rng.standard_normal((n,))
    stop = lambda k, x, fx: k >= 50
    hist = gradient_descent_fixed(obj.f, obj.grad, x0, alpha=alpha, max_iter=50, stop=stop)
    assert hist.fvals[-1] < hist.fvals[0]
