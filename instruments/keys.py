import sys
sys.path.append("..")
import numpy as np
from osc import sine, sawtooth, noise
from env import adsr, cos_env
from biquad import biquad_filter
from effect import reverb
from instruments.utils import calc_freq, calc_delayar


def pipe_organ(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100,
               enable_reverb: bool =True) -> np.ndarray:
    N_PARTIAL: int = 15
    freq: float = calc_freq(note_no)
    vcos: np.ndarray = np.tile(
        freq*np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])[:np.newaxis], int(duration * sr)
    )
    vca_delays: np.ndarray = calc_delayar(vcos, p1=0, p2=-0.05, p3=8000/12, p4=0.1, is_delay=True)
    vca_as: np.ndarray = calc_delayar(vcos, p1=0, p2=0.05, p3=8000/12, p4=0.1, is_sub=True)
    vca_rs: np.ndarray = calc_delayar(vcos, p1=0, p2=0.2, p3=8000/12, p4=0.4, is_sub=True)
    vca_depths: np.ndarray = np.array([0.30, 0.60, 0.30, 0.85, 0.25, 0.50, 0.95, 0.25, 0.50, 0.15,
                                       0.95, 0.00, 0.25, 0.00, 0.50])

    z: np.ndarray = np.zeros(int(duration*sr))
    for i in range(N_PARTIAL):
        jitter: np.ndarray = biquad_filter(noise(sec=duration, sr=sr), filter_type="lowpass",
                                           fc=40, Q=1/np.sqrt(2), sr=sr)
        jitter = calc_delayar(note_no, p1=108, p2=0.2, p3=120/12, p4=4) / np.max(np.abs(jitter)) * jitter
        shimmer: np.ndarray = biquad_filter(noise(sec=duration, sr=sr), filter_type="lowpass",
                                            fc=40, Q=1/np.sqrt(2), sr=sr)
        shimmer = calc_delayar(vcos[i], p1=1, p2=-0.05, p3=100/12, p4=0.2) / np.max(np.abs(shimmer)) * shimmer
        vca: np.ndarray = vca_depths[i] \
                        * cos_env(delay=vca_delays[i], A=vca_as[i], S=1, R=vca_rs[i], gate=gate, dur=duration, sr=sr)
        vca = np.maximum((1 + shimmer) * vca, 0)
        z += vca * sine(vcos[i], sec=duration, sr=sr)
    z = (velocity / 127) / np.max(np.abs(z)) * z
    return reverb(z, reverb_time=2, level=0.1, sr=sr) if enable_reverb else z


def read_organ(note_no: int, velocity: int, gate: float, duration: float, sr: int =44100) -> np.ndarray:
    freq: float = calc_freq(note_no)
    vco: np.ndarray = freq * np.ones(int(duration*sr))
    vca: np.ndarray = adsr(A=0.5, D=0, S=1, R=0.2, gate=gate, dur=duration, sr=sr)
    z: np.ndarray = sawtooth(vco, sec=duration, sr=sr)
    z = vca * biquad_filter(z, filter_type="lowpass", fc=500, Q=1/np.sqrt(2), sr=sr)
    return (velocity / 127) / np.max(np.abs(z)) * z