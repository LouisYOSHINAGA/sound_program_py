import sys
sys.path.append("..")
import numpy as np
from env import adsr
from osc import sine
from biquad import biquad_filter
import instruments.utils as utils


def triangle_in(velocity: int, gate: float, duration: float =9.0, sr: int =44100) -> np.ndarray:
    vco_offsets: list[float] = [
          172.4,   297.8,  1066.0,  1639.0,  1827.0,  3035.0,  3428.0,  4208.0,  5072.0,  6856.0,
         7001.0,  8577.0,  9474.0, 11442.0, 11616.0, 12753.0, 13836.0, 15367.0, 16085.0, 16383.0,
        16961.0, 18217.0, 19221.0, 20309.0, 21818.0
    ]
    vca_params: list[dict[str, float]] = [
        {'A': 0.01, 'D': 8.0, 'S': 0.0, 'R': 8.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 8.0, 'S': 0.0, 'R': 8.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 6.0, 'S': 0.0, 'R': 6.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.07},
        {'A': 0.01, 'D': 6.0, 'S': 0.0, 'R': 6.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.15},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.29},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.08},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.56},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.37},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.04},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.41},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.00},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.05},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.05},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.04},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.03},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.03},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
    ]
    assert len(vco_offsets) == len(vca_params)

    adsr_args: list[str] = utils.get_func_kwargs(adsr)
    ys: np.ndarray = np.zeros((len(vco_offsets), int(duration*sr)))
    for i, (vco_offset, vca_param) in enumerate(zip(vco_offsets, vca_params)):
        vco: np.ndarray = np.repeat(vco_offset, int(duration*sr))
        vca: np.ndarray = vca_param['offset'] \
                        + vca_param['depth'] * adsr(**{k: v for k, v in vca_param.items() if k in adsr_args})
        if np.max(vco) < sr / 2:
            ys[i] = vca * sine(fs=vco, sr=sr)
    y: np.ndarray = np.sum(ys, axis=0)
    return (velocity / 127) / np.max(np.abs(y)) * y


def triangle_out(velocity: int, gate: float, duration: float =9.0, sr: int =44100) -> np.ndarray:
    vco_offsets: list[float] = [
          176.5,   872.7,  1593.0,  1791.0,  3035.0,  3928.0,  4817.0,  4873.0,  5345.0,  6856.0,
         7001.0,  8441.0,  8770.0, 10213.0, 11442.0, 11616.0, 13836.0, 14834.0, 15259.0, 15737.0,
        16961.0, 18217.0, 18227.0, 21691.0, 21818.0
    ]
    vca_params: list[dict[str, float]] = [
        {'A': 0.01, 'D': 8.0, 'S': 0.0, 'R': 8.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 8.0, 'S': 0.0, 'R': 8.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 6.0, 'S': 0.0, 'R': 6.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.06},
        {'A': 0.01, 'D': 6.0, 'S': 0.0, 'R': 6.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.12},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 4.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.04},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.03},
        {'A': 0.01, 'D': 4.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.02},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.41},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 1.00},
        {'A': 0.01, 'D': 2.0, 'S': 0.0, 'R': 2.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.39},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.57},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.43},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.04},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.06},
        {'A': 0.01, 'D': 1.0, 'S': 0.0, 'R': 1.0, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.04},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
        {'A': 0.01, 'D': 0.5, 'S': 0.0, 'R': 0.5, 'gate': gate, 'dur': duration, 'offset': 0.0, 'depth': 0.01},
    ]
    assert len(vco_offsets) == len(vca_params)

    adsr_args: list[str] = utils.get_func_kwargs(adsr)
    ys: np.ndarray = np.zeros((len(vco_offsets), int(duration*sr)))
    for i, (vco_offset, vca_param) in enumerate(zip(vco_offsets, vca_params)):
        vco: np.ndarray = np.repeat(vco_offset, int(duration*sr))
        vca: np.ndarray = vca_param['offset'] \
                        + vca_param['depth'] * adsr(**{k: v for k, v in vca_param.items() if k in adsr_args})
        if np.max(vco) < sr / 2:
            ys[i] = vca * sine(fs=vco, sr=sr)
    y: np.ndarray = np.sum(ys, axis=0)
    return (velocity / 127) / np.max(np.abs(y)) * y


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