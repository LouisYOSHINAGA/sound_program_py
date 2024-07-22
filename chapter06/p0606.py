import sys
sys.path.append("..")
import numpy as np
import inspect
from env import adsr
from osc import sine
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    f0: float = 440
    gate: float = 3.0
    sec: float = 4.0

    partial_params: list[dict[str, dict[str, float]]] = [
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0}},
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': 2*f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0}},
    ]
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()

    vcos: np.ndarray = np.empty((len(partial_params), int(sec*sr)))
    vcas: np.ndarray = np.empty((len(partial_params), int(sec*sr)))
    for i, partial_param in enumerate(partial_params):
        po: dict[str, float] = partial_param['vco']
        vcos[i] = po['offset'] + po['depth'] * adsr(**{k: v for k, v in po.items() if k in adsr_args})
        pa: dict[str, float] = partial_param['vca']
        vcas[i] = pa['offset'] + pa['depth'] * adsr(**{k: v for k, v in pa.items() if k in adsr_args})
    y: np.ndarray = np.mean(vcas * sine(fs=vcos, sr=sr), axis=0)

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0606_output.wav", is_mono=True)