import sys
sys.path.append("..")
import numpy as np
from wavio import write_wave_16bit


n2f: dict[str, float] = {
    'C2': 130.81,
    'E3': 164.81,
    'F3': 174.61,
    'G3': 196.00,
    'A3': 220.00,
    'C4': 261.63,
    'G4': 392.00,
    'A4': 440.00,
    'B4': 493.88,
    'C5': 523.25,
    'D5': 587.33,
    'E5': 659.26,
}


def gen_sine(amp: float, freq: float, duration: float, sr: int =44100, fade_sec: float =0.01) -> np.ndarray:
    ts: np.ndarray = np.arange(0, duration, 1/sr)
    x: np.ndarray = amp * np.sin(2 * np.pi * freq * ts)
    c: np.ndarray = np.arange(0, 1, 1/(fade_sec*sr))
    x[:int(fade_sec*sr)] *= c
    x[-int(fade_sec*sr):] *= c[: : -1]
    x = amp / np.max(np.abs(x)) * x
    return x

def render(score: list[dict[str, int|float]], volume: float, sr: int =44100, blank: int =2) -> np.ndarray:
    n_tracks: int = len(np.unique([s['track'] for s in score]))
    n_samples: int = int((max([s['onset'] for s in score]) + blank) * sr)
    tracks: np.ndarray = np.zeros((n_samples, n_tracks))

    for s in score:
        offset: int = int(s['onset'] * sr)
        dur_sample: int = int(s['dur'] * sr)
        tracks[offset:offset+dur_sample, s['track']-1] = gen_sine(amp=s['amp'], freq=s['freq'], duration=s['dur'], sr=sr)

    track: np.ndarray = np.sum(tracks, axis=1)
    track = volume / np.max(np.abs(track)) * track
    return track


def main() -> None:
    score: list[dict[str, int|float]] = [
        {'track': 1, 'onset': 2.0, 'amp': 0.5, 'freq': n2f['E5'], 'dur': 1.0},
        {'track': 1, 'onset': 3.0, 'amp': 0.5, 'freq': n2f['D5'], 'dur': 1.0},
        {'track': 1, 'onset': 4.0, 'amp': 0.5, 'freq': n2f['C5'], 'dur': 1.0},
        {'track': 1, 'onset': 5.0, 'amp': 0.5, 'freq': n2f['B4'], 'dur': 1.0},
        {'track': 1, 'onset': 6.0, 'amp': 0.5, 'freq': n2f['A4'], 'dur': 1.0},
        {'track': 1, 'onset': 7.0, 'amp': 0.5, 'freq': n2f['G4'], 'dur': 1.0},
        {'track': 1, 'onset': 8.0, 'amp': 0.5, 'freq': n2f['A4'], 'dur': 1.0},
        {'track': 1, 'onset': 9.0, 'amp': 0.5, 'freq': n2f['B4'], 'dur': 1.0},
        {'track': 2, 'onset': 2.0, 'amp': 0.5, 'freq': n2f['C4'], 'dur': 1.0},
        {'track': 2, 'onset': 3.0, 'amp': 0.5, 'freq': n2f['G3'], 'dur': 1.0},
        {'track': 2, 'onset': 4.0, 'amp': 0.5, 'freq': n2f['A3'], 'dur': 1.0},
        {'track': 2, 'onset': 5.0, 'amp': 0.5, 'freq': n2f['E3'], 'dur': 1.0},
        {'track': 2, 'onset': 6.0, 'amp': 0.5, 'freq': n2f['F3'], 'dur': 1.0},
        {'track': 2, 'onset': 7.0, 'amp': 0.5, 'freq': n2f['C2'], 'dur': 1.0},
        {'track': 2, 'onset': 8.0, 'amp': 0.5, 'freq': n2f['F3'], 'dur': 1.0},
        {'track': 2, 'onset': 9.0, 'amp': 0.5, 'freq': n2f['G3'], 'dur': 1.0},
    ]
    sr: int = 44100
    track: np.ndaray = render(score, volume=0.5, sr=sr)
    write_wave_16bit(track, sr=sr, filename="p0301_output.wav", is_mono=True)

if __name__ == "__main__":
    main()