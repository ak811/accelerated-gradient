from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.objectives.piecewise1d import PiecewiseStronglyConvex1D

def test_piecewise_grad_matches_fd_away_from_kinks() -> None:
    obj = PiecewiseStronglyConvex1D()
    xs = [np.array([0.3]), np.array([1.5]), np.array([2.7])]

    eps = 1e-6
    for x in xs:
        g = obj.grad(x)
        gfd = (obj.f(x + eps) - obj.f(x - eps)) / (2 * eps)
        assert np.allclose(g, gfd, atol=1e-4, rtol=1e-4)
