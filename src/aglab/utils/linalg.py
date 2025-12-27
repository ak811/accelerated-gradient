from __future__ import annotations
import numpy as np

def sym_eig_minmax(A: np.ndarray) -> tuple[float, float]:
    w = np.linalg.eigvalsh(A)
    return float(w.min()), float(w.max())
