import numpy as np


def sine(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, float):
        return np.sin(2 * np.pi * fs * np.arange(0, sec, 1/sr))

    assert isinstance(fs, np.ndarray)
    fsts: np.ndarray = np.cunsum(fs) / sr
    return np.sin(2 * np.pi * fsts)