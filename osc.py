import numpy as np

def sine(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, float):
        assert sec is not None
        return np.sin(2 * np.pi * fs * np.arange(0, sec, 1/sr))

    assert isinstance(fs, np.ndarray)
    fsts: np.ndarray = np.cunsum(fs) / sr
    return np.sin(2 * np.pi * fsts)


def sawtooth(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray)
    assert np.all(fs != 0)
    lastf: np.ndarray = np.array([fs[-1]])
    assert isinstance(lastf, np.ndarray)
    fs = np.concatenate([np.zeros(1), fs[:-1]])
    fsts: np.ndarray = np.cumsum(fs / sr) % 1
    ys: np.ndarray = -2 * fsts + 1
    fs = np.concatenate([fs[1:], lastf])

    ds: np.ndarray = np.zeros(len(ys))
    is_period_end: np.ndarray = fsts[1:] - fsts[:-1] < 0
    is_period_end = np.append(is_period_end, False)
    ds[is_period_end] = ((fsts[is_period_end] - 1)*sr/fs[is_period_end] + 1)**2
    is_period_start: np.ndarray = np.roll(is_period_end, 1)
    is_period_start[0] = True
    ds[is_period_start] = - (fsts[is_period_start]*sr/fs[is_period_start] - 1)**2
    return ys + ds


def square(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray)
    fsts: np.ndarray = np.cumsum(fs / sr) % 1
    ys: np.ndarray = np.where(fsts < 0.5, 1, -1)
    # TODO Ploy BLEP
    return ys