import sys
sys.path.append("..")
import numpy as np
from osc import sine
from env import adsr
from wavio import write_wave_16bit


if __name__ == "__main__":
    sr: int = 44100
    f0: float = 440
    gate: float = 3.0
    sec: float = 4.0

    partial_params: list[dict[str, dict[str, float]]] = [
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec,
                 'offset': f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec,
                 'offset': 0.0, 'depth': 1.0}},
        {'vco': {'A': 0.0, 'D': 0.0, 'S': 1.0, 'R': 0.0, 'gate': sec, 'dur': sec,
                 'offset': 2*f0, 'depth': 0.0},
         'vca': {'A': 0.01, 'D': 0.0, 'S': 1.0, 'R': 0.01, 'gate': gate, 'dur': sec,
                 'offset': 0.0, 'depth': 1.0}},
    ]

    vcos: np.ndarray = np.empty((len(partial_params), int(sec*sr)))
    for i, partial_param in enumerate(partial_params):
        p: dict[str, float] = partial_param['vco']
        vcos[i] = p['offset'] \
                + p['depth'] * adsr(sec_A=p['A'], sec_D=p['D'], amp_S=p['S'], sec_R=p['R'],
                                    sec_gate=p['gate'], dur=p['dur'], sr=sr)
    ys: np.ndarray = sine(fs=vcos, sr=sr)

    vcas: np.ndarray = np.empty((len(partial_params), int(sec*sr)))
    for i, partial_param in enumerate(partial_params):
        p: dict[str, float] = partial_param['vca']
        vcas[i] = p['offset'] \
                + p['depth'] * adsr(sec_A=p['A'], sec_D=p['D'], amp_S=p['S'], sec_R=p['R'],
                                    sec_gate=p['gate'], dur=p['dur'], sr=sr)
        ys[i] *= vcas[i]

    y: np.ndarray = np.mean(ys, axis=0)

    blank: float = 1.0
    vol: float = 0.5
    z: np.ndarray = np.zeros(int(blank*sr))
    out: np.ndarray = vol * np.concatenate([z, y, z])

    write_wave_16bit(out, sr, f"p0606_output.wav", is_mono=True)