import numpy as np
from scipy.io import wavfile


def read_wave_16bit(filename: str, is_mono: bool =False) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(filename)
    data = data.astype(np.float64)  # (sample, channel)
    assert (len(data.shape) == 1 and is_mono) or (len(data.shape) == 2 and not is_mono)
    assert np.all(-32768 <= data) and np.all(data <= 32767)

    data = 2 * (data + 32768) / 65535 - 1
    assert np.all(-1 <= data) and np.all(data <= 1)
    return data, sr

def write_wave_16bit(data: np.ndarray, sr: int, filename: str, is_mono: bool =False) -> None:
    assert (len(data.shape) == 1 and is_mono) or (len(data.shape) == 2 and not is_mono)
    assert np.all(-1 <= data) and np.all(data <= 1)
    data = 65536 * (data + 1) / 2 - 32768

    assert np.all(-32768 <= data) and np.all(data <= 32767)
    wavfile.write(filename, sr, data.astype(np.int16))