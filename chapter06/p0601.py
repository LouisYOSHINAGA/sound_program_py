import sys
sys.path.append("..")
import numpy as np
from osc import sine
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    sec: float = 1.0

    f: float = 440.0
    vco: np.ndarray = sine(fs=f, sr=sr, sec=sec)
    vca: np.ndarray = np.ones(int(sec*sr))
    ys: np.ndarray = vco * vca

    blank: float = 1.0
    vol: float = 0.5
    zs: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([zs, ys, zs])

    write_wave_16bit(out, sr, "p0601_output.wav", is_mono=True)