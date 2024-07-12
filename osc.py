import numpy as np


def sine(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, float):
        assert sec is not None
        return np.sin(2 * np.pi * fs * np.arange(0, sec, 1/sr))

    assert isinstance(fs, np.ndarray)
    fsts: np.ndarray = np.cunsum(fs) / sr
    return np.sin(2 * np.pi * fsts)

def sawtooth(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, int):
        fs = float(fs)
    if isinstance(fs, float):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    fsts: np.ndarray = np.cumsum(fs) / sr % 1
    print(fsts)
    ys: np.ndarray = -2 * fsts + 1
    # TODO: impl Poly BLEP
    return ys