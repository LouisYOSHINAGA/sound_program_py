import numpy as np


def sine(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray)
    dim: int|tuple[int, int] = 1 if len(fs.shape) == 1 else (fs.shape[0], 1)
    phases: np.ndarray = np.cumsum(np.concatenate([np.zeros(dim), fs[..., :-1]/sr], axis=-1), axis=-1) % 1
    return np.sin(2 * np.pi * phases)


def sawtooth(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray) and np.all(fs != 0)
    dim: int|tuple[int, int] = 1 if len(fs.shape) == 1 else (fs.shape[0], 1)
    phases: np.ndarray = np.cumsum(np.concatenate([np.zeros(dim), fs[..., :-1]/sr], axis=-1), axis=-1) % 1
    ys: np.ndarray = -2 * phases + 1

    ds: np.ndarray = np.zeros(ys.shape)
    is_period_end: np.ndarray = phases[..., 1:] < phases[..., :-1]
    is_period_end = np.concatenate([is_period_end, np.full(dim, False)], axis=-1)
    ds[is_period_end] = ((phases[is_period_end] - 1)*sr/fs[is_period_end] + 1)**2
    is_period_start: np.ndarray = np.roll(is_period_end, 1, axis=-1)
    is_period_start[..., 0] = True
    ds[is_period_start] = - (phases[is_period_start]*sr/fs[is_period_start] - 1)**2
    return ys + ds


def square(fs: float|np.ndarray, sr: int, sec: float|None =None) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray) and np.all(fs != 0)
    dim: int|tuple[int, int] = 1 if len(fs.shape) == 1 else (fs.shape[0], 1)
    phases: np.ndarray = np.cumsum(np.concatenate([np.zeros(dim), fs[..., :-1]/sr], axis=-1), axis=-1) % 1
    ys: np.ndarray = np.where(phases < 0.5, 1, -1)

    ds: np.ndarray = np.zeros(ys.shape)
    is_period_end: np.ndarray = phases[..., 1:] < phases[..., :-1]
    is_period_end = np.concatenate([is_period_end, np.full(dim, False)], axis=-1)
    ds[is_period_end] = ((phases[is_period_end] - 1)*sr/fs[is_period_end] + 1)**2
    is_period_start: np.ndarray = np.roll(is_period_end, 1, axis=-1)
    is_period_start[..., 0] = True
    ds[is_period_start] = - (phases[is_period_start]*sr/fs[is_period_start] - 1)**2

    is_before_jump: np.ndarray = ys[..., 1:] < ys[..., :-1]
    is_before_jump= np.concatenate([is_before_jump, np.full(dim, False)], axis=-1)
    ds[is_before_jump] = - ((phases[is_before_jump] - 0.5)*sr/fs[is_before_jump] + 1)**2
    is_after_jump: np.ndarray = np.roll(is_before_jump, 1, axis=-1)
    ds[is_after_jump] = ((phases[is_after_jump] - 0.5)*sr/fs[is_before_jump] - 1)**2
    return ys + ds


def white_noise(fs: float|np.ndarray, sr: int, sec: float) -> np.ndarray:
    if isinstance(fs, (int, float)):
        assert sec is not None
        fs = np.full(int(sec*sr), fs)
    assert isinstance(fs, np.ndarray)
    ys: np.ndarray = 2 * np.random.rand(*fs.shape) - 1
    assert np.all(-1 <= ys) and np.all(ys <= 1)
    return ys