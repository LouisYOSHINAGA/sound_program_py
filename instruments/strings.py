import sys
sys.path.append("..")
import numpy as np
from osc import noise
from env import adsr
from biquad import biquad_filter
from instruments.utils import calc_freq, calc_delayar


def abstract_string(note_no: int, velocity: int, gate: float, duration: float, f_lpf: float, f_hpf: float, sr: int) -> np.ndarray:
    freq: float = calc_freq(note_no)
    n_partial: int = int(np.clip(6000/freq, 5, 30))

    ratio: float = calc_delayar(offsets=note_no, p1=69, p2=0.02, p3=70/12, p4=0.18, is_sub=True)
    vib_depth: float = calc_freq(note_no + ratio) - freq
    vib: np.ndarray = vib_depth * np.sin(2 * np.pi * 8 * np.arange(int(duration*sr)) / sr)
    fluc: np.ndarray = biquad_filter(noise(sec=duration, sr=sr), filter_type="lowpass", fc=8, Q=1/np.sqrt(2), sr=sr)
    fluc = 0.5 * vib_depth / np.max(np.abs(fluc)) * fluc
    vcf: np.ndarray = (freq + vib + fluc) * np.arange(1, n_partial+1).reshape(-1, 1)

    v: np.ndarray = biquad_filter(noise(sec=duration, sr=sr), filter_type="lowpass", fc=f_lpf, Q=1/np.sqrt(2), sr=sr)
    v = biquad_filter(v, filter_type="lowpass", fc=f_lpf, Q=1/np.sqrt(2), sr=sr)
    v = biquad_filter(v, filter_type="highpass", fc=f_hpf, Q=1/np.sqrt(2), sr=sr)
    v = biquad_filter(v, filter_type="highpass", fc=f_hpf, Q=1/np.sqrt(2), sr=sr)

    y: np.ndarray = np.zeros(int(duration * sr))
    for i in range(n_partial):
        x: np.ndarray = biquad_filter(v, filter_type="bandpass", fc=vcf[i], Q=200, sr=sr)
        y += biquad_filter(x, filter_type="bandpass", fc=vcf[i], Q=200, sr=sr)
    y *= adsr(A=0.2, D=0, S=1, R=0.4, gate=gate, dur=duration, sr=sr)
    return (velocity / 127) / np.max(np.abs(y)) * y


def violin(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    return abstract_string(
        note_no=note_no, velocity=velocity, gate=gate, duration=duration, f_lpf=2000, f_hpf=250, sr=sr
    )