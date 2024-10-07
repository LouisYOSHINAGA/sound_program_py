import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from effect import reverb


def load_tracks(dir: str, data: dict[str, dict[str, float|np.ndarray]]) -> dict[str, np.ndarray]:
    for file in data.keys():
        y, _ = read_wave_16bit(f"{dir}/{file}.wav", is_mono=True)
        data[file]['wave'] = y
    return data

def mixdown(data: dict[str, dict[str, float|np.ndarray]]) -> np.ndarray:
    samples: int = max([len(v['wave']) for v in data.values()])
    master: np.ndarray = np.zeros((samples, 2))
    for vpw in data.values():
        master[:, 0] += vpw['volume'] * np.cos(vpw['pan']*np.pi/2) * vpw['wave']
        master[:, 1] += vpw['volume'] * np.sin(vpw['pan']*np.pi/2) * vpw['wave']
    return master

def effect(master: np.ndarray, reverb_time: float, level: float) -> np.ndarray:
    master[:, 0] = reverb(master[:, 0], reverb_time=reverb_time, level=level)
    master[:, 1] = reverb(master[:, 1], reverb_time=reverb_time, level=level)
    return master

def save_tracks(master: np.ndarray, mvol: float, dir: str, title: str, sr: int =44100) -> None:
    master = mvol / np.max(np.abs(master)) * master
    write_wave_16bit(master, filename=f"{dir}/{title}", sr=sr)


def main() -> None:
    load_dir: str = "./p0804_tracks"
    save_dir: str = "./"
    data: dict[str, dict[str, float|np.ndarray]] = {
        'vocal': {'volume': 1.0, 'pan': 0.5},
        'oboe1': {'volume': 0.4, 'pan': 0.3},
        'oboe2': {'volume': 0.4, 'pan': 0.2},
        'clarinet1': {'volume': 0.4, 'pan': 0.7},
        'clarinet2': {'volume': 0.4, 'pan': 0.8},
        'tuba': {'volume': 0.7, 'pan': 0.5},
        'hihat': {'volume': 0.4, 'pan': 0.6},
        'crash': {'volume': 0.9, 'pan': 0.4},
        'snare': {'volume': 0.5, 'pan': 0.5},
        'kick': {'volume': 0.7, 'pan': 0.5},
    }
    data = load_tracks(dir=load_dir, data=data)
    master: np.ndarray = mixdown(data)
    master = effect(master, reverb_time=2, level=0.1)
    save_tracks(master, mvol=0.9, dir=save_dir, title="p0804_output.wav")

if __name__ == "__main__":
    main()