# Accelerated Gradient Lab

A small, self-contained codebase for experimenting with **first-order optimization methods** and comparing their empirical convergence behavior on controlled objectives.

This repo focuses on clarity, reproducibility, and extensibility: you can add new objectives and methods without turning the project into a framework nobody asked for.

## What’s included

### Methods
- **Gradient Descent (GD)** with fixed step size
- **Heavy-Ball (HB)** momentum method
- **Nesterov Accelerated Gradient (NAG)** with:
  - fixed momentum parameter **β** (typical tuned form for strongly convex settings)
  - adaptive **βₖ = (k−1)/(k+2)** (standard convex-setting schedule)

### Objectives
- **Quadratic**
  - \( f(x) = \tfrac{1}{2} x^\top A x + b^\top x \)
  - controlled spectrum construction with eigenvalues in \([\mu, L]\)
  - supports both strongly convex (\(\mu>0\)) and PSD/weakly convex (\(\mu=0\)) cases
- **Piecewise 1D strongly convex** example
  - smooth but not twice differentiable at kink points (useful for “theory vs practice” behavior)

### Outputs
The scripts generate plots in `figures/` (created automatically if missing).

---

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want an editable install (recommended while developing):

```bash
pip install -e .
```

---

## Run the demos / benchmarks

Run everything:

```bash
python scripts/make_all.py
```

Or run individual scripts:

### Quadratic benchmark + demos

```bash
python scripts/run_quadratic_benchmark.py
```

This script runs three cases:

1. **Strongly convex quadratic** (\(\mu>0\))  
   - compares GD (two step sizes), HB tuned parameters, and tuned NAG  
   - reports mean/std iteration counts over multiple random initializations  
   - produces a semilog optimality-gap plot

2. **PSD quadratic with linear term** (\(\mu=0\), typically unbounded below)  
   - demonstrates practical divergence behavior  
   - stops when the function value reaches a target threshold (e.g., `-2000`)

3. **PSD quadratic with no linear term** (\(\mu=0,\ b=0\))  
   - compares GD, “bad” NAG with fixed β=1, and convex-schedule NAG  
   - includes reference curves (1/k and 1/k²) for visual comparison

Expected figure outputs (filenames):
- `figures/quadratic_strongly_convex_gaps.png`
- `figures/quadratic_mu0_unbounded_values.png`
- `figures/quadratic_mu0_b0_gaps.png`
- `figures/quadratic_mu0_b0_rate_compare.png`

### Piecewise 1D demo + quadratic-style worst-case bounds

```bash
python scripts/run_piecewise1d_demo.py
```

This script:
- runs GD, NAG, HB from the same initial point
- plots their function values
- overlays **quadratic worst-case style bounds** computed using specified \((\mu, L)\) as a baseline comparison

Expected figure outputs:
- `figures/piecewise1d_values_vs_bounds_40.png`
- `figures/piecewise1d_values_vs_bounds_10.png`

---

## Run tests

```bash
pytest -q
```

The tests cover:
- finite-difference gradient checks for both objectives (away from kink points for the piecewise function)
- basic “GD decreases with safe step” sanity check
- “Nesterov runs and improves” sanity check

---

## Project structure

```
accelerated-gradient-lab/
  scripts/          # reproducible runs that generate plots into figures/
  src/aglab/        # library code (objectives, optimizers, plotting helpers)
  tests/            # pytest test suite
  figures/          # generated outputs (created automatically)
```

---

## Extending the project

### Add a new objective
Create `src/aglab/objectives/<name>.py` and implement at least:
- `f(x)` returning a scalar (float or 0-d array)
- `grad(x)` returning a vector

Export it in `src/aglab/objectives/__init__.py`.

### Add a new optimizer
Create `src/aglab/optim/<name>.py` and follow the style of existing methods:
- accept `f`, `grad`, `x0`, hyperparameters, `max_iter`, and a `stop` callback
- return a `History` object with `xs`, `fvals`, and `n_iter`

Export it in `src/aglab/optim/__init__.py`.

### Add a new experiment script
Create `scripts/<name>.py` that:
- constructs an objective and one or more optimizers
- saves plots into `figures/` using helpers in `src/aglab/plotting/`

---

## Reproducibility

- Deterministic randomness is set via `numpy` seeds in scripts.
- Quadratic matrices are constructed with a controlled spectrum in \([\mu, L]\) using an orthonormal basis + diagonal eigenvalue assignment.
- Scripts are designed to stop using explicit criteria (gap threshold, function-value threshold, or fixed iteration count for plots).

---

## License and citation

- License: MIT (add a `LICENSE` file if you want it explicit)
- Citation metadata: see `CITATION.cff`

