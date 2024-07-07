import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from biquad import lpf_coef


if __name__ == "__main__":
    data, sr = read_wave_16bit("p0503_input.wav", is_mono=True)

    fc: float = 1000
    Q: float = 1 / np.sqrt(2)
    a, b = lpf_coef(fc, Q, sr)

    z1: np.ndarray = np.concatenate([np.array([0]), data[:-1]])
    z2: np.ndarray = np.concatenate([np.array([0, 0]), data[:-2]])
    assert z1.shape == data.shape and z2.shape == data.shape
    mid_data: np.ndarray = b[0] * data + b[1] * z1 + b[2] * z2
    assert mid_data.shape == data.shape

    lpf_data: np.ndarray = np.concatenate([np.array([mid_data[0],
                                                     mid_data[1] - a[1] * mid_data[0]]),
                                           np.zeros(len(mid_data)-2)])
    for i in range(2, len(lpf_data)):
        lpf_data[i] = mid_data[i] - a[1] * lpf_data[i-1] - a[2] * lpf_data[i-2]
    assert lpf_data.shape == data.shape

    write_wave_16bit(lpf_data, sr, "p0503_output.wav", is_mono=True)