import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from biquad import biquad_filter

if __name__ == "__main__":
    x, sr = read_wave_16bit("p0704_input.wav", is_mono=True)

    fc: float = 500.0
    Q: float = 1 / np.sqrt(2)
    g: float = -1.0
    x = biquad_filter(x, filter_type="lowshelf", fc=fc, Q=Q, g=g, sr=sr)

    fc = 1000.0
    Q = 1 / np.sqrt(2)
    g = 1.0
    x = biquad_filter(x, filter_type="peaking", fc=fc, Q=Q, g=g, sr=sr)

    fc = 2000.0
    Q = 1 / np.sqrt(2)
    g = -1.0
    x = biquad_filter(x, filter_type="highshelf", fc=fc, Q=Q, g=g, sr=sr)

    volume: float = 1.0
    y: np.ndarray = volume * x / np.max(np.abs(x))

    write_wave_16bit(y, sr, "p0704_output.wav", is_mono=True)