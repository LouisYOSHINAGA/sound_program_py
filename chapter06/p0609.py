import sys
sys.path.append("..")
import numpy as np
import inspect
from env import adsr
from biquad import biquad_filter
from wavio import write_wave_16bit


def generate_normalized_noise_pad(n_sample: int, length: int, seed: int =0) -> np.ndarray:
    np.random.seed(seed)
    noise: np.ndarray = 2 * np.random.rand(n_sample) - 1
    noise -= np.mean(noise)
    noise = np.concatenate([noise, np.zeros(length-n_sample)])
    return noise


if __name__ == "__main__":
    sr: int = 44100
    f0: int = 440
    gate: float = 3.0
    sec: float = 4.0
    sample: int = int(sec * sr)

    decay_sec: float = 8.0
    decay_rate: float = 0.5
    period: float = 1 / f0

    c: float = np.clip(
        np.power(10, -3*period/decay_sec) \
        / np.sqrt((1 - decay_rate)**2 + 2 * decay_rate * (1 - decay_rate) * np.cos(2 *np.pi * f0 / sr) + decay_rate**2),
        None, 1.0
    )
    delay_int: int = int(period * sr - decay_rate)
    delay_frac: float = period * sr - decay_rate - delay_int
    apf_coef: float = (1 - delay_frac) / (1 + delay_frac)

    y: np.ndarray = generate_normalized_noise_pad(n_sample=delay_int+1, length=sample)
    apf_out: float = 0.0
    apf_out_z: float = 0.0
    for n in range(delay_int+1, sample):
        apf_out = - apf_coef * apf_out_z + apf_coef * y[n-delay_int] + y[n-delay_int-1]
        y[n] += c * ((1 - decay_rate) * apf_out + decay_rate * apf_out_z)
        apf_out_z = apf_out

    fc: float = 5.0
    Q: float = 1 / np.sqrt(2)
    vca_param: dict[str, float] = {
        'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.1, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0
    }
    vca: np.ndarray = vca_param['offset']  \
                    + vca_param['depth'] * adsr(**{k: v for k, v in vca_param.items()
                                                   if k in inspect.signature(adsr).parameters.keys()})
    y = vca * biquad_filter(y, filter_type="highpass", fc=fc, Q=Q, sr=sr)

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0609_output.wav", is_mono=True)