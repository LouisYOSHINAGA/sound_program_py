import numpy as np


def hanning_window(n: int) -> np.ndarray:
    d: float = 0.0 if n % 2 == 0 else 0.5
    window: np.ndarray = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + d) / n)
    return window