from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.config import ensure_figures_dir
from aglab.objectives.quadratic import Quadratic, make_symmetric_psd_with_spectrum
from aglab.optim.gd import gradient_descent_fixed
from aglab.optim.heavy_ball import heavy_ball
from aglab.optim.nesterov import nesterov_strongly_convex, nesterov_convex
from aglab.plotting.lines import semilog_lines, line_plot
from aglab.utils.seeds import set_global_seed


def _stop_on_gap(eps: float, f_star: float):
    def stop(k: int, x: np.ndarray, fx: float) -> bool:
        return (fx - f_star) <= eps
    return stop


def _stop_on_value(target: float):
    def stop(k: int, x: np.ndarray, fx: float) -> bool:
        return fx <= target
    return stop


def main() -> None:
    figs = ensure_figures_dir()
    set_global_seed(4)

    # Benchmark settings
    n = 100
    epsilon = 1e-6
    num_mc = 10

    # -------------------------
    # Case A: strongly convex quadratic (mu > 0)
    # -------------------------
    mu = 0.01
    L = 1.0
    kappa = L / mu

    A, eigs = make_symmetric_psd_with_spectrum(n=n, mu=mu, L=L, seed=4)
    rng = np.random.default_rng(4)
    b = rng.standard_normal((n,))
    obj = Quadratic(A=A, b=b)

    x_star = obj.minimizer()  # unique since mu>0 -> SPD in our construction
    f_star = obj.f(x_star)

    alpha_gd_opt = 2.0 / (L + mu)
    alpha_gd_L = 1.0 / L

    alpha_hb = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta_hb = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    alpha_nag = 1.0 / L
    beta_nag = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    methods = {
        "GD 2/(L+mu)": ("gd", dict(alpha=alpha_gd_opt)),
        "GD 1/L": ("gd", dict(alpha=alpha_gd_L)),
        "Heavy-Ball (tuned)": ("hb", dict(alpha=alpha_hb, beta=beta_hb)),
        "Nesterov (tuned)": ("nag_sc", dict(alpha=alpha_nag, beta=beta_nag)),
    }

    iters = {name: [] for name in methods.keys()}
    typical_hist = {}

    max_iter = 200000
    for trial in range(num_mc):
        x0 = rng.standard_normal((n,))

        for name, (kind, params) in methods.items():
            stop = _stop_on_gap(epsilon, f_star)

            if kind == "gd":
                hist = gradient_descent_fixed(obj.f, obj.grad, x0, max_iter=max_iter, stop=stop, **params)
            elif kind == "hb":
                hist = heavy_ball(obj.f, obj.grad, x0, max_iter=max_iter, stop=stop, **params)
            elif kind == "nag_sc":
                hist = nesterov_strongly_convex(obj.f, obj.grad, x0, max_iter=max_iter, stop=stop, **params)
            else:
                raise ValueError(kind)

            iters[name].append(hist.n_iter)
            if trial == 0:
                typical_hist[name] = hist

    print("=== Quadratic benchmark: strongly convex (mu>0) ===")
    print(f"n={n}, mu={mu}, L={L}, kappa={kappa:.2f}, epsilon={epsilon:g}")
    print(f"eig(A) approx in [{eigs.min():.4g}, {eigs.max():.4g}]")
    for name in methods.keys():
        arr = np.asarray(iters[name], float)
        print(f"{name:22s} mean iters={arr.mean():.2f}  std={arr.std():.2f}")

    gaps = {name: (typical_hist[name].fvals - f_star) for name in typical_hist.keys()}
    semilog_lines(gaps, figs / "quadratic_strongly_convex_gaps.png", ylabel="Optimality gap f(x_k)-f*")

    # -------------------------
    # Case B: weakly convex PSD quadratic (mu = 0), b != 0 (often unbounded below)
    # -------------------------
    mu0 = 0.0
    A0, eigs0 = make_symmetric_psd_with_spectrum(n=n, mu=mu0, L=L, seed=4)
    b0 = rng.standard_normal((n,))
    obj0 = Quadratic(A=A0, b=b0)

    alpha_gd_aggressive = 2.0 / (L + mu0)  # = 2/L
    alpha_gd_safe = 1.0 / L
    alpha_nag0 = 1.0 / L
    beta_nag0 = 1.0
    alpha_hb0 = 4.0 / (np.sqrt(L) + 0.0) ** 2
    beta_hb0 = 1.0

    target_f = -2000.0
    methods0 = {
        "GD 2/L": ("gd", dict(alpha=alpha_gd_aggressive)),
        "GD 1/L": ("gd", dict(alpha=alpha_gd_safe)),
        "HB (beta=1)": ("hb", dict(alpha=alpha_hb0, beta=beta_hb0)),
        "NAG (beta=1)": ("nag_sc", dict(alpha=alpha_nag0, beta=beta_nag0)),
    }

    typical0 = {}
    for name, (kind, params) in methods0.items():
        x0 = rng.standard_normal((n,))
        stop = _stop_on_value(target_f)
        if kind == "gd":
            hist = gradient_descent_fixed(obj0.f, obj0.grad, x0, max_iter=5000, stop=stop, **params)
        elif kind == "hb":
            hist = heavy_ball(obj0.f, obj0.grad, x0, max_iter=5000, stop=stop, **params)
        elif kind == "nag_sc":
            hist = nesterov_strongly_convex(obj0.f, obj0.grad, x0, max_iter=5000, stop=stop, **params)
        else:
            raise ValueError(kind)
        typical0[name] = hist

    print("\n=== Quadratic demo: PSD (mu=0) with linear term (may be unbounded below) ===")
    print("Stopping once f(x_k) <= -2000.")
    for name, hist in typical0.items():
        print(f"{name:14s} iters={hist.n_iter:4d}  f_last={hist.fvals[-1]:.3f}")

    series_vals = {name: typical0[name].fvals for name in typical0.keys()}
    line_plot(series_vals, figs / "quadratic_mu0_unbounded_values.png", ylabel="Function value f(x_k)")

    # -------------------------
    # Case C: mu=0, b=0 (convex quadratic, minimizer at 0)
    # -------------------------
    obj0b0 = Quadratic(A=A0, b=np.zeros_like(b0))
    f_star0 = 0.0

    x0 = rng.standard_normal((n,))
    stop_gap0 = _stop_on_gap(epsilon, f_star0)

    hist_gd = gradient_descent_fixed(obj0b0.f, obj0b0.grad, x0, alpha=1.0 / L, max_iter=200000, stop=stop_gap0)
    hist_nag_bad = nesterov_strongly_convex(obj0b0.f, obj0b0.grad, x0, alpha=1.0 / L, beta=1.0, max_iter=200000, stop=stop_gap0)
    hist_nag_cvx = nesterov_convex(obj0b0.f, obj0b0.grad, x0, alpha=1.0 / L, max_iter=200000, stop=stop_gap0)

    gaps0 = {
        "GD 1/L": hist_gd.fvals - f_star0,
        "NAG beta=1": hist_nag_bad.fvals - f_star0,
        "NAG beta_k": hist_nag_cvx.fvals - f_star0,
    }
    semilog_lines(gaps0, figs / "quadratic_mu0_b0_gaps.png", ylabel="Optimality gap f(x_k)-f*")

    # Rate reference curves (for visual comparison)
    T = min(len(hist_gd.fvals), len(hist_nag_cvx.fvals), 2000)
    one_over_k = 1.0 / np.arange(1, T + 1, dtype=float)
    one_over_k2 = 1.0 / (np.arange(1, T + 1, dtype=float) ** 2)
    compare = {
        "GD 1/L": (hist_gd.fvals[:T] - f_star0),
        "NAG beta=1": (hist_nag_bad.fvals[:T] - f_star0),
        "NAG beta_k": (hist_nag_cvx.fvals[:T] - f_star0),
        "1/k": one_over_k,
        "1/k^2": one_over_k2,
    }
    semilog_lines(compare, figs / "quadratic_mu0_b0_rate_compare.png", ylabel="Scale comparison")

    print(f"\nSaved figures to: {figs}")


if __name__ == "__main__":
    main()
