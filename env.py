import numpy as np


def adsr(A: float, D: float, S: float, R: float, gate: float, dur: float, alpha: float =-5.0, sr: int =44100) -> np.ndarray:
    sample_A: int = int(A * sr)
    sample_D: int = int(D * sr)
    sample_gate: int = int(gate * sr)
    sample_R: int = int(R * sr)
    sample: int = int(dur* sr)

    env: np.ndarray = np.empty(sample)
    env[0:sample_A] = (1 - np.exp(alpha * np.arange(sample_A) / sample_A)) / (1 - np.exp(alpha))
    env[sample_A:sample_gate] = 1 + (S - 1) * (1 - np.exp(alpha * (np.arange(sample_A, sample_gate) - sample_A) / sample_D)) \
                                if D > 0 else np.full((sample_gate-sample_A), S)
    env[sample_gate:sample] = env[sample_gate-1] * np.exp(alpha * (np.arange(sample_gate, sample) - sample_gate + 1) / sample_R)
    return env