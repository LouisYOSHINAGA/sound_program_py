import sys
sys.path.append("..")
import numpy as np
from wavio import write_wave_16bit


def gen_sine(amp: float, freq: float, duration: float, sr: int =44100,
             fade_sec: float =0.01, blank: float =1.0) -> np.ndarray:
    ts: np.ndarray = np.arange(0, duration, 1/sr)
    x: np.ndarray = amp * np.sin(2 * np.pi * freq * ts)
    c: np.ndarray = np.arange(0, 1, 1/(fade_sec*sr))
    x[:int(fade_sec*sr)] *= c
    x[-int(fade_sec*sr):] *= c[: : -1]
    b: np.ndarray = np.zeros(int(blank * sr))
    return np.concatenate([b, x, b])


if __name__ == "__main__":
    sr: int = 44100
    sine: np.ndarray = gen_sine(amp=0.5, freq=880, duration=1.0, sr=sr)
    write_wave_16bit(sine, sr=sr, filename="p0201_output.wav", is_mono=True)