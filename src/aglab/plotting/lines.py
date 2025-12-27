from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def semilog_lines(series: dict[str, np.ndarray], outpath: Path, ylabel: str) -> None:
    fig = plt.figure()
    for label, y in series.items():
        yy = np.asarray(y, float).ravel()
        yy = np.maximum(yy, 1e-300)  # avoid log(0)
        plt.semilogy(yy, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.legend()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def line_plot(series: dict[str, np.ndarray], outpath: Path, ylabel: str) -> None:
    fig = plt.figure()
    for label, y in series.items():
        plt.plot(np.asarray(y, float).ravel(), label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.legend()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
