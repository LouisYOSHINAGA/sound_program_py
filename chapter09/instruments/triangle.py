import sys
sys.path.append("....")
import numpy as np
import inspect
from env import adsr
from osc import sine


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

    ys: np.ndarray = np.zeros((len(vco_offsets), int(duration*sr)))
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()
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

    ys: np.ndarray = np.zeros((len(vco_offsets), int(duration*sr)))
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()
    for i, (vco_offset, vca_param) in enumerate(zip(vco_offsets, vca_params)):
        vco: np.ndarray = np.repeat(vco_offset, int(duration*sr))
        vca: np.ndarray = vca_param['offset'] \
                        + vca_param['depth'] * adsr(**{k: v for k, v in vca_param.items() if k in adsr_args})
        if np.max(vco) < sr / 2:
            ys[i] = vca * sine(fs=vco, sr=sr)

    y: np.ndarray = np.sum(ys, axis=0)
    return (velocity / 127) / np.max(np.abs(y)) * y