import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from effect import compressor

if __name__ == "__main__":
    x, sr = read_wave_16bit("p0703_input.wav", is_mono=True)
    threshold: float = 0.2
    width: float = 0.1
    ratio: float = 8.0
    y: np.ndarray = compressor(x, threshold, width, ratio)
    write_wave_16bit(y, sr, "p0703_output.wav", is_mono=True)