import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from wavio import read_wave_16bit


def plot_mono_wave(ts: np.ndarray, data: np.ndarray, sr: int, valid_sec: float, title: str):
    assert len(data.shape) == 1 and ts.shape == data.shape
    assert valid_sec <= len(data) / sr
    plt.figure(figsize=(12, 4))
    plt.axis([0, valid_sec, -1, 1])
    plt.xlabel("time [sec]")
    plt.ylabel("amplitude")
    plt.plot(ts, data)
    plt.tight_layout()
    plt.show()
    plt.savefig(title, dpi=320)


if __name__ == "__main__":
    data, sr = read_wave_16bit("p0501_input.wav", is_mono=True)
    ts: np.ndarray = np.arange(0, len(data)) / sr
    plot_mono_wave(ts, data, sr, valid_sec=0.008, title="p0501_amp.png")