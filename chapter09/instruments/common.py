import numpy as np

A4_FREQ: float = 440.0
A4_NOTE: int = 69

def calc_freq(note_no: int) -> float:
    return A4_FREQ * 2 ** ((note_no - A4_NOTE) / 12)

def gen_noise(n_sample: int, seed: int =0) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    return 2 * (rng.random(size=n_sample) - 0.5)