import sys
sys.path.append("..")
import numpy as np
from osc import sawtooth
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    sec: float = 4.0
    fs: float = 440.0
    depth: float = 0.8
    rate: float = 1.0

    x: np.ndarray = sawtooth(fs=fs, sr=sr, sec=sec)
    a: np.ndarray = 1 + depth * np.sin(2 * np.pi * rate * np.arange(0, sec, 1/sr))
    y: np.ndarray = a * x

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0706_output.wav", is_mono=True)