import sys
sys.path.append("..")
import numpy as np
from osc import noise, sine
from env import adsr, cos_env
from biquad import biquad_filter
from instruments.utils import calc_freq, calc_delayar


def trumpet(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    freq: float = calc_freq(note_no)
    n_partial: int = int(np.clip(6000 / freq, 5, 30))

    vco_as: np.ndarray = np.repeat(0, n_partial)
    vco_ds: np.ndarray = np.repeat(0.15, n_partial)
    vco_ss: np.ndarray = np.repeat(0, n_partial)
    vco_rs: np.ndarray = np.repeat(0.15, n_partial)
    vco_gates: np.ndarray = np.repeat(duration, n_partial)
    vco_durs: np.ndarray = np.repeat(duration, n_partial)
    vco_offsets: np.ndarray = freq * np.arange(1, 1+n_partial)
    vco_depths: np.ndarray = -0.15 * freq * np.arange(1, 1+n_partial)

    vca0_as: np.ndarray = np.repeat(0.01, n_partial)
    vca0_ds: np.ndarray = np.repeat(0.02, n_partial)
    vca0_ss: np.ndarray = np.repeat(0, n_partial)
    vca0_rs: np.ndarray = np.repeat(0.02, n_partial)
    vca0_gates: np.ndarray = np.repeat(gate, n_partial)
    vca0_durs: np.ndarray = np.repeat(duration, n_partial)
    vca0_offsets: np.ndarray = np.repeat(0, n_partial)
    vca0_depths: np.ndarray = np.repeat(1, n_partial)

    vca1_delays: np.ndarray =  calc_delayar(vco_offsets+vco_depths, p1=3000, p2=0, p3=3000/12, p4=0.03, is_delay=True)
    vca1_as: np.ndarray =  calc_delayar(vco_offsets, p1=3000, p2=0.02, p3=3000/12, p4=0.02, is_sub=True)
    vca1_ss: np.ndarray = np.repeat(1, n_partial)
    vca1_rs: np.ndarray =  calc_delayar(vco_offsets, p1=3000, p2=0.2, p3=3000/12, p4=0.1, is_sub=True)
    vca1_gates: np.ndarray = np.repeat(gate, n_partial)
    vca1_durs: np.ndarray = np.repeat(duration, n_partial)
    vca1_offsets: np.ndarray = np.repeat(0, n_partial)
    vca1_depths: np.ndarray = np.repeat(1, n_partial)

    z0: np.ndarray = np.zeros(int(duration * sr))
    z1: np.ndarray = np.zeros(int(duration * sr))
    for i in range(n_partial):
        jitter: np.ndaray = biquad_filter(data=noise(sec=duration, sr=sr),
                                          filter_type="lowpass", fc=40, Q=1/np.sqrt(2), sr=sr)
        jitter = calc_delayar(note_no, p1=108, p2=1, p3=150/12, p4=20) * jitter / np.max(np.abs(jitter))
        vco: np.ndarray = jitter + vco_offsets[i] \
                        + vco_depths[i] * adsr(vco_as[i], vco_ds[i], vco_ss[i], vco_rs[i], vco_gates[i], vco_durs[i], sr=sr)

        vca0: np.ndarray = vca0_offsets[i] \
                         + vca0_depths[i] * adsr(vca0_as[i], vca0_ds[i], vca0_ss[i], vca0_rs[i], vca0_gates[i], vca0_durs[i], sr=sr)
        z0 += vca0 * sine(fs=vco, sr=sr, sec=duration)

        shimmer: np.ndarray = biquad_filter(data=noise(sec=duration, sr=sr),
                                            filter_type="lowpass", fc=40, Q=1/np.sqrt(2), sr=sr)
        shimmer = calc_delayar(i+1, p1=1, p2=-0.3, p3=10/12, p4=0.8) * shimmer / np.max(np.abs(shimmer))
        vca1: np.ndarray = vca1_offsets[i] \
                        + vca1_depths[i] * cos_env(vca1_delays[i], vca1_as[i], vca1_ss[i], vca1_rs[i], vca1_gates[i], vca1_durs[i], sr=sr)
        vca1 = np.maximum((1 + shimmer) * vca1, 0)
        z1 += vca1 * sine(fs=vco, sr=sr, sec=duration)
    z: np.ndarray = 0.3 * z0 + 0.7 * z1

    y: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=1500, Q=1/np.sqrt(2), sr=sr)
    y = biquad_filter(data=y, filter_type="lowpass", fc=3000, Q=1/np.sqrt(2), sr=sr)
    return (velocity / 127) / np.max(np.abs(y)) * y