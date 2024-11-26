import sys
sys.path.append("..")
import numpy as np
from osc import noise, sine
from env import adsr, cos_env
from biquad import biquad_filter
from instruments.utils import calc_freq, calc_delayar


def abstruct_brass(note_no: int, velocity: int, gate: float, duration: float, f: float, sr: int =44100) -> np.ndarray:
    freq: float = calc_freq(note_no)
    n_partial: int = int(np.clip(2*f/freq, 5, 30))

    jitter: np.ndaray = biquad_filter(data=noise(sec=duration, sr=sr), filter_type="lowpass", fc=40, Q=1/np.sqrt(2), sr=sr)
    jitter = calc_delayar(note_no, p1=108, p2=1, p3=150/12, p4=20) * jitter / np.max(np.abs(jitter))
    vco_offsets: np.ndarray = freq * np.arange(1, 1+n_partial)
    vco_depths: np.ndarray = -0.15 * vco_offsets
    vco: np.ndarray = np.tile(jitter, (n_partial, 1)) + vco_offsets[:, np.newaxis] \
                    + vco_depths[:, np.newaxis] * np.tile(adsr(A=0, D=0.15, S=0, R=0.15, gate=gate, dur=duration, sr=sr), (n_partial, 1))
    vco_sine: np.ndarray = sine(fs=vco, sr=sr, sec=duration)

    vca0: np.ndarray = adsr(A=0.01, D=0.02, S=0, R=0.02, gate=gate, dur=duration, sr=sr)
    z0: np.ndarray = np.sum(vca0 * vco_sine, axis=0)

    vca1_delays: np.ndarray =  calc_delayar(vco_offsets+vco_depths, p1=f, p2=0, p3=f/12, p4=0.03, is_delay=True)
    vca1_as: np.ndarray =  calc_delayar(vco_offsets, p1=f, p2=0.02, p3=f/12, p4=0.02, is_sub=True)
    vca1_rs: np.ndarray =  calc_delayar(vco_offsets, p1=f, p2=0.2, p3=f/12, p4=0.1, is_sub=True)
    z1: np.ndarray = np.zeros(int(duration * sr))
    for i in range(n_partial):
        shimmer: np.ndarray = biquad_filter(data=noise(sec=duration, sr=sr), filter_type="lowpass", fc=40, Q=1/np.sqrt(2), sr=sr)
        shimmer = calc_delayar(i+1, p1=1, p2=-0.3, p3=100/12, p4=0.8) * shimmer / np.max(np.abs(shimmer))
        vca1: np.ndarray = cos_env(delay=vca1_delays[i], A=vca1_as[i], S=1, R=vca1_rs[i], gate=gate, dur=duration, sr=sr)
        vca1 = np.maximum((1 + shimmer) * vca1, 0)
        z1 += vca1 * vco_sine[i]

    y: np.ndarray = biquad_filter(0.3*z0 + 0.7*z1, filter_type="bandpass", fc=f/2, Q=1/np.sqrt(2), sr=sr)
    y = biquad_filter(data=y, filter_type="lowpass", fc=f, Q=1/np.sqrt(2), sr=sr)
    return (velocity / 127) / np.max(np.abs(y)) * y


def trumpet(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    return abstruct_brass(note_no=note_no, velocity=velocity, gate=gate, duration=duration, f=3000, sr=sr)


def trombone(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    return abstruct_brass(note_no=note_no, velocity=velocity, gate=gate, duration=duration, f=1800, sr=sr)


def horn(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    return abstruct_brass(note_no=note_no, velocity=velocity, gate=gate, duration=duration, f=1200, sr=sr)


def tuba(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    return abstruct_brass(note_no=note_no, velocity=velocity, gate=gate, duration=duration, f=800, sr=sr)