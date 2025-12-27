from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.objectives.quadratic import Quadratic
from aglab.optim.nesterov import nesterov_strongly_convex

def test_nesterov_runs_and_improves() -> None:
    rng = np.random.default_rng(2)
    n = 15
    M = rng.standard_normal((n, n))
    A = M.T @ M + 0.5 * np.eye(n)
    b = rng.standard_normal((n,))
    obj = Quadratic(A=A, b=b)

    w = np.linalg.eigvalsh(A)
    mu = float(w.min())
    L = float(w.max())

    alpha = 1.0 / L
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    x0 = rng.standard_normal((n,))
    stop = lambda k, x, fx: k >= 50
    hist = nesterov_strongly_convex(obj.f, obj.grad, x0, alpha=alpha, beta=beta, max_iter=50, stop=stop)
    assert hist.fvals[-1] < hist.fvals[0]
