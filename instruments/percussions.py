import sys
sys.path.append("..")
import numpy as np
from env import adsr
from biquad import biquad_filter
import instruments.utils as utils


def marimba(note_no: int, velocity: int, gate: float, duration: float =1.8, sr: int =44100) -> np.ndarray:
    params: dict[str, dict[str, float]] = {
        'vcf': {'A': 0.0, 'D': 0.2, 'S': 0.0, 'R': 0.2, 'gate': gate, 'dur': duration, 'offset': 500, 'depth': 2000},
        'vca': {'A': 0.0, 'D': 0.8, 'S': 0.0, 'R': 0.8, 'gate': gate, 'dur': duration, 'offset':   0, 'depth':    1},
    }
    adsr_args: list[str] = utils.get_func_kwargs(adsr)
    vco: np.ndarray = utils.noise(int(duration * sr))
    vcf: np.ndarray = params['vcf']['offset'] \
                    + params['vcf']['depth'] * adsr(**{k: v for k, v in params['vcf'].items() if k in adsr_args})
    vca: np.ndarray = params['vca']['offset'] \
                    + params['vca']['depth'] * adsr(**{k: v for k, v in params['vca'].items() if k in adsr_args})

    freq: float = utils.calc_freq(note_no)
    x: np.ndarray = biquad_filter(data=vco, filter_type="lowpass", fc=freq, Q=1/np.sqrt(2), sr=sr)
    z: np.ndarray = biquad_filter(data=x, filter_type="lowpass", fc=vcf, Q=1/np.sqrt(2), sr=sr)
    z0: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=freq, Q=200, sr=sr)
    z1: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=4*freq, Q=200, sr=sr)
    z2: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=10*freq, Q=200, sr=sr)
    y: np.ndarray = vca * (z0 + z1 + z2)
    return (velocity / 127) / np.max(np.abs(y)) * y


def xylophone(note_no: int, velocity: int, gate: float, duration: float =1.8, sr: int =44100) -> np.ndarray:
    params: dict[str, dict[str, float]] = {
        'vcf': {'A': 0.0, 'D': 0.2, 'S': 0.0, 'R': 0.2, 'gate': gate, 'dur': duration, 'offset': 500, 'depth': 2000},
        'vca': {'A': 0.0, 'D': 0.8, 'S': 0.0, 'R': 0.8, 'gate': gate, 'dur': duration, 'offset':   0, 'depth':    1},
    }
    adsr_args: list[str] = utils.get_func_kwargs(adsr)
    vco: np.ndarray = utils.noise(int(duration * sr))
    vcf: np.ndarray = params['vcf']['offset'] \
                    + params['vcf']['depth'] * adsr(**{k: v for k, v in params['vcf'].items() if k in adsr_args})
    vca: np.ndarray = params['vca']['offset'] \
                    + params['vca']['depth'] * adsr(**{k: v for k, v in params['vca'].items() if k in adsr_args})

    freq: float = utils.calc_freq(note_no)
    z: np.ndarray = biquad_filter(data=vco, filter_type="lowpass", fc=vcf, Q=1/np.sqrt(2), sr=sr)
    z0: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=freq, Q=200, sr=sr)
    z1: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=3*freq, Q=200, sr=sr)
    z2: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=6.5*freq, Q=200, sr=sr)
    z3: np.ndarray = biquad_filter(data=z, filter_type="bandpass", fc=10*freq, Q=200, sr=sr)
    y: np.ndarray = vca * (z0 + z1 + z2 + z3)
    return (velocity / 127) / np.max(np.abs(y)) * y