import sys
sys.path.append("..")
import numpy as np
from osc import sawtooth
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    sec: float = 4.0

    params: list[dict[str, float]] = [{'vco': 440.0, 'vca': 1.0}, {'vco': 440.5, 'vca': 1.0}]

    ys: np.ndarray = np.empty((len(params), int(sec*sr)))
    for i, param in enumerate(params):
        ys[i] = param['vca'] * sawtooth(fs=param['vco'], sr=sr, sec=sec)
    y: np.ndarray = np.sum(ys, axis=0)

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, "p0708_output.wav", is_mono=True)