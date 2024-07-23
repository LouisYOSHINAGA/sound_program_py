import sys
sys.path.append("..")
import numpy as np
import inspect
from env import adsr
from osc import sawtooth
from biquad import lpf_coef, biquad_filter
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    f0: int = 440
    gate: float = 3.0
    sec: float = 4.0

    partial_params: list[dict[str, dict[str, float]]] = [
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': f0, 'depth': 0.0},
         'vcf': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec, 'offset': 2*f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec, 'offset': 0.0, 'depth': 1.0}},
    ]
    partial_param: dict[str, dict[str, float]] = partial_params[0]
    adsr_args: list[str] = inspect.signature(adsr).parameters.keys()

    po: dict[str, float] = partial_param['vco']
    vco: np.ndarray = po['offset'] + po['depth'] * adsr(**{k: v for k, v in po.items() if k in adsr_args})
    y: np.ndarray = sawtooth(fs=vco, sr=sr)

    pf: dict[str, float] = partial_param['vcf']
    vcf: np.ndarray = pf['offset'] + pf['depth'] * adsr(**{k: v for k, v in pf.items() if k in adsr_args})
    y = biquad_filter(y, lpf_coef(fc=vcf, Q=1/np.sqrt(2), sr=sr))

    pa: dict[str, float] = partial_param['vca']
    vca: np.ndarray = pa['offset'] + pa['depth'] * adsr(**{k: v for k, v in pa.items() if k in adsr_args})
    y *= vca

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0607_output.wav", is_mono=True)