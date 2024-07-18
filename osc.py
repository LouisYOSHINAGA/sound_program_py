import numpy as np
from typing import Any


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
    assert isinstance(fs, np.ndarray) and np.all(fs != 0)
    phases: np.ndarray = np.cumsum(np.concatenate([[0], fs[:-1]/sr])) % 1
    ys: np.ndarray = -2 * phases + 1

    ds: np.ndarray = np.zeros(len(ys))
    is_period_end: np.ndarray = phases[1:] < phases[:-1]
    is_period_end = np.append(is_period_end, False)
    ds[is_period_end] = ((phases[is_period_end] - 1)*sr/fs[is_period_end] + 1)**2
    is_period_start: np.ndarray = np.roll(is_period_end, 1)
    is_period_start[0] = True
    ds[is_period_start] = - (phases[is_period_start]*sr/fs[is_period_start] - 1)**2
    return ys + ds


def square(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray) and np.all(fs != 0)
    phases: np.ndarray = np.cumsum(np.concatenate([[0], fs[:-1]/sr])) % 1
    ys: np.ndarray = np.where(phases < 0.5, 1, -1)

    ds: np.ndarray = np.zeros(len(ys))
    is_period_end: np.ndarray = phases[1:] < phases[:-1]
    is_period_end = np.append(is_period_end, False)
    ds[is_period_end] = ((phases[is_period_end] - 1)*sr/fs[is_period_end] + 1)**2
    is_period_start: np.ndarray = np.roll(is_period_end, 1)
    is_period_start[0] = True
    ds[is_period_start] = - (phases[is_period_start]*sr/fs[is_period_start] - 1)**2

    is_before_jump: np.ndarray = ys[1:] < ys[:-1]
    is_before_jump = np.append(is_before_jump, False)
    ds[is_before_jump] = - ((phases[is_before_jump] - 0.5)*sr/fs[is_before_jump] + 1)**2
    is_after_jump: np.ndarray = np.roll(is_before_jump, 1)
    ds[is_after_jump] = ((phases[is_after_jump] - 0.5)*sr/fs[is_before_jump] - 1)**2
    return ys + ds


def white_noise(sr: int, sec: float, fs: Any) -> np.ndarray:
    ys: np.ndarray = 2 * np.random.rand(int(sec*sr)) - 1
    assert np.all(-1 <= ys) and np.all(ys <= 1)
    return ys