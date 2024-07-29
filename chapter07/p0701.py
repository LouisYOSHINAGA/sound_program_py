import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from effect import reverb


if __name__ == "__main__":
    reverb_time: float = 2.0
    level: float = 0.1
    x, sr = read_wave_16bit("p0701_input.wav", is_mono=True)
    y = reverb(x, reverb_time, level, sr)
    write_wave_16bit(y, sr, "p0701_output.wav", is_mono=True)