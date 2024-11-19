import sys
sys.path.append("..")
import numpy as np
from env import adsr
from osc import sine
from biquad import biquad_filter
import instruments.utils as utils


def tubular_bells(note_no: int, velocity: int, gate: float, duration: float =5.0, sr: int =44100) -> np.ndarray:
    freq: float = utils.calc_freq(note_no)
    params_mod: dict[str, dict[str, float]] = {
        'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset': 3.5*freq, 'depth': 0.0},
        'vca': {'A': 0.0, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate,     'dur': duration, 'offset':        0, 'depth': 1.0},
    }
    params_car: dict[str, dict[str, float]] = {
        'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset': 1.0*freq, 'depth': 0.0},
        'vca': {'A': 0.0, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate,     'dur': duration, 'offset':        0, 'depth': 1.0},
    }

    adsr_args: list[str] = utils.get_func_kwargs(adsr)
    vco_mod: np.ndarray = params_mod['vco']['offset'] \
                        + params_mod['vco']['depth'] * adsr(**{k: v for k, v in params_mod['vco'].items() if k in adsr_args})
    vca_mod: np.ndarray = params_mod['vca']['offset'] \
                        + params_mod['vca']['depth'] * adsr(**{k: v for k, v in params_mod['vca'].items() if k in adsr_args})
    vco_car: np.ndarray = params_car['vco']['offset'] \
                        + params_car['vco']['depth'] * adsr(**{k: v for k, v in params_car['vco'].items() if k in adsr_args})
    vca_car: np.ndarray = params_car['vca']['offset'] \
                        + params_car['vca']['depth'] * adsr(**{k: v for k, v in params_car['vca'].items() if k in adsr_args})

    mod: np.ndarray = vca_mod * sine(fs=vco_mod, sr=sr, sec=duration)
    phases: np.ndarray = np.cumsum(np.concatenate([np.zeros(1), vco_car[:-1]/sr], axis=-1), axis=-1) % 1
    y: np.ndarray = vca_car * np.sin(2 * np.pi * phases + mod)
    return (velocity / 127) / np.max(np.abs(y)) * y


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