from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from scripts.run_quadratic_benchmark import main as quad_main
from scripts.run_piecewise1d_demo import main as pw_main


def main() -> None:
    quad_main()
    pw_main()


if __name__ == "__main__":
    main()
