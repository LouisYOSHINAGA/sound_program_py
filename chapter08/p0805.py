import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit

if __name__ == "__main__":
    ys, sr = read_wave_16bit("p0805_input.wav")
    z: np.ndarray = ys[:, 0] - ys[:, 1]
    write_wave_16bit(z, filename="p0805_output.wav", sr=sr, is_mono=True)