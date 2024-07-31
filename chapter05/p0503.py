import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from biquad import biquad_filter


if __name__ == "__main__":
    data, sr = read_wave_16bit("p0503_input.wav", is_mono=True)

    fc: float = 1000
    Q: float = 1 / np.sqrt(2)
    lpf_data: np.ndarray = biquad_filter(data, filter_type="lowpass", fc=fc, Q=Q, sr=sr)

    write_wave_16bit(lpf_data, sr, "p0503_output.wav", is_mono=True)