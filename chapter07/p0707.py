import sys
sys.path.append("..")
import numpy as np
from osc import sine, sawtooth
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    sec: float = 4.0
    depth: float = 50.0
    rate: float = 1.0

    vco: np.ndarray = 440.0 + depth * sine(fs=rate, sr=sr, sec=sec)
    vca: np.ndarray = np.ones(int(sec*sr))
    y: np.ndarray = vca * sawtooth(fs=vco, sr=sr, sec=sec)

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, "p0707_output.wav", is_mono=True)