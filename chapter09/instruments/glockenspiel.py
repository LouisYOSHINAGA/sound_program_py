import sys
sys.path.append("....")
import numpy as np
import inspect
from env import adsr
from osc import sine
from .common import calc_freq


def glockenspiel(note_no: int, velocity: int, gate: float, duration: float =5.0, sr: int =44100) -> np.ndarray:
    freq: float = calc_freq(note_no)
    vco_params: list[dict[str, float]] = [
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset':  1.0*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset':  2.8*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset':  5.4*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset':  8.9*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset': 13.3*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset': 18.6*freq, 'depth': 0.0},
        {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': duration, 'dur': duration, 'offset': 24.8*freq, 'depth': 0.0},
    ]
    vca_params: list[dict[str, float]] = [
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 0.8, 'S': 0.0, 'R': 0.8, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 0.6, 'S': 0.0, 'R': 0.6, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 0.4, 'S': 0.0, 'R': 0.4, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
        {'A': 0.01, 'D': 0.3, 'S': 0.0, 'R': 0.3, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.0},
    ]
    assert len(vco_params) == len(vca_params)

    ys: np.ndarray = np.zeros((len(vco_params), int(duration*sr)))
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()
    for i, (vco_param, vca_param) in enumerate(zip(vco_params, vca_params)):
        vco: np.ndarray = vco_param['offset'] \
                        + vco_param['depth'] * adsr(**{k: v for k, v in vco_param.items() if k in adsr_args})
        vca: np.ndarray = vca_param['offset'] \
                        + vca_param['depth'] * adsr(**{k: v for k, v in vca_param.items() if k in adsr_args})
        if np.max(vco) < sr / 2:
            ys[i] = vca * sine(fs=vco, sr=sr)

    y: np.ndarray = np.sum(ys, axis=0)
    return (velocity / 127) / np.max(np.abs(y)) * y