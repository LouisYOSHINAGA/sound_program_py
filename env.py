import numpy as np


def adsr(sec_A: float, sec_D: float, sec_gate: float, amp_S: float, sec_R: float,
         sec: float, sr: int =44100) -> np.ndarray:
    sample_A: int = int(sec_A * sr)
    sample_D: int = int(sec_D * sr)
    sample_gate: int = int(sec_gate * sr)
    sample_R: int = int(sec_R * sr)
    sample: int = int(sec * sr)

    env: np.ndarray = np.empty(sample)
    env[0:sample_A] = (1 - np.exp(-5 * np.arange(sample_A) / sample_A)) / (1 - np.exp(-5))
    env[sample_A:sample_gate] = 1 + (amp_S - 1) * (1 - np.exp(-5 * (np.arange(sample_A, sample_gate) - sample_A) / sample_D)) \
                                if sec_D > 0 else np.full((sample_gate-sample_A), amp_S)
    env[sample_gate:sample] = env[sample_gate-1] - env[sample_gate-1] * (1 - np.exp(-5 * (np.arange(sample_gate, sample) - sample_gate + 1) / sample_R))
    return env