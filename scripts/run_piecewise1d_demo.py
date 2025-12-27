from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aglab.config import ensure_figures_dir
from aglab.objectives.piecewise1d import PiecewiseStronglyConvex1D
from aglab.optim.gd import gradient_descent_fixed
from aglab.optim.heavy_ball import heavy_ball
from aglab.optim.nesterov import nesterov_strongly_convex
from aglab.plotting.lines import line_plot


def main() -> None:
    figs = ensure_figures_dir()
    obj = PiecewiseStronglyConvex1D()

    # Parameters used for the standard comparison
    L = 50.0
    mu = 2.0
    kappa = L / mu

    x0 = np.array([3.0])
    x_star = np.array([0.0])
    f_star = float(obj.f(x_star))
    f0 = float(obj.f(x0))

    # Method parameters
    alpha_gd = 1.0 / 50.0
    alpha_nag = 1.0 / 50.0
    beta_nag = 2.0 / 3.0
    alpha_hb = 1.0 / 18.0
    beta_hb = 4.0 / 9.0

    num_iters = 40
    stop = lambda k, x, fx: k >= num_iters

    hist_gd = gradient_descent_fixed(obj.f, obj.grad, x0, alpha=alpha_gd, max_iter=num_iters, stop=stop)
    hist_nag = nesterov_strongly_convex(obj.f, obj.grad, x0, alpha=alpha_nag, beta=beta_nag, max_iter=num_iters, stop=stop)
    hist_hb = heavy_ball(obj.f, obj.grad, x0, alpha=alpha_hb, beta=beta_hb, max_iter=num_iters, stop=stop)

    k = np.arange(num_iters, dtype=float)

    # Worst-case bounds for a quadratic with same (mu, L) (comparison baseline)
    gd_bound = (f0 - f_star) * (1.0 - mu / L) ** k + f_star
    nag_bound = (f0 - f_star + (mu / 2.0) * float((x0 - x_star) @ (x0 - x_star))) * (1.0 - np.sqrt(mu / L)) ** k + f_star
    hb_bound = (f0 - f_star) * (1.0 - 2.0 / (np.sqrt(kappa) + 1.0)) ** k + f_star

    series = {
        "GD": hist_gd.fvals[:num_iters],
        "GD bound (quadratic)": gd_bound,
        "Nesterov": hist_nag.fvals[:num_iters],
        "Nesterov bound (quadratic)": nag_bound,
        "Heavy-Ball": hist_hb.fvals[:num_iters],
        "Heavy-Ball bound (quadratic)": hb_bound,
    }
    line_plot(series, figs / "piecewise1d_values_vs_bounds_40.png", ylabel="Function value f(x_k)")

    t10 = 10
    line_plot({k: np.asarray(v)[:t10] for k, v in series.items()},
              figs / "piecewise1d_values_vs_bounds_10.png",
              ylabel="Function value f(x_k)")

    print("=== Piecewise 1D demo ===")
    print(f"Saved figures to: {figs}")
    print(f"Final values: GD={hist_gd.fvals[-1]:.6g}, Nesterov={hist_nag.fvals[-1]:.6g}, HB={hist_hb.fvals[-1]:.6g}")


if __name__ == "__main__":
    main()
