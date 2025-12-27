import numpy as np

def set_global_seed(seed: int = 0) -> None:
    np.random.seed(seed)
