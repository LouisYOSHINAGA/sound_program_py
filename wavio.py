import numpy as np
from scipy.io import wavfile
from typing import Tuple  # TODO upgrade python (>= 3.9)


def read_wave_16bit(filename: str, is_mono: bool =False, seed: int =0) -> Tuple[np.ndarray, int]:
    sr, data = wavfile.read(filename)
    data = data.astype(np.float)  # (sample, channel)
    assert (len(data.shape) == 1 and is_mono) or (len(data.shape) == 2 and not is_mono)
    assert np.all(-32768 <= data) and np.all(data <= 32767)

    data = 2 * (data + 32768) / 65535 - 1
    assert np.all(-1 <= data) and np.all(data <= 1)
    return data, sr