import sys
sys.path.append("..")
import numpy as np
import inspect
from env import adsr
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    f0: int = 440
    gate: float = 3.0
    sec: float = 4.0

    partial_params: list[dict[str, dict[str, float]]] = [
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': 3.5*f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0}},
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0}},
    ]
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()

    po_mod: dict[str, dict[str, float]] = partial_params[0]['vco']
    vco_mod: np.ndarray = po_mod['offset'] + po_mod['depth'] * adsr(**{k: v for k, v in po_mod.items() if k in adsr_args})

    pa_mod: dict[str, dict[str, float]] = partial_params[0]['vca']
    vca_mod: np.ndarray = pa_mod['offset'] + pa_mod['depth'] * adsr(**{k: v for k, v in pa_mod.items() if k in adsr_args})

    po_fc: dict[str, dict[str, float]] = partial_params[1]['vco']
    vco_fc: np.ndarray = po_fc['offset'] + po_fc['depth'] * adsr(**{k: v for k, v in po_fc.items() if k in adsr_args})

    pa_fc: dict[str, dict[str, float]] = partial_params[1]['vca']
    vca_fc: np.ndarray = pa_fc['offset'] + pa_fc['depth'] * adsr(**{k: v for k, v in pa_fc.items() if k in adsr_args})

    phases_mod: np.ndarray = np.cumsum(np.concatenate([np.zeros(1), vco_mod[:-1]/sr])) % 1
    phases_fc: np.ndarray = np.cumsum(np.concatenate([np.zeros(1), vco_fc[:-1]/sr])) % 1
    y: np.ndarray = vca_fc * np.sin(2 * np.pi * phases_fc + vca_mod * np.sin(2 * np.pi * phases_mod))

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0608_output.wav", is_mono=True)