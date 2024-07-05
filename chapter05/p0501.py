import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from wavio import read_wave_16bit
from window import hanning_window


def plot_mono_wave(ts: np.ndarray, data: np.ndarray, sr: int, valid_sec: float, title: str) -> None:
    assert len(data.shape) == 1 and ts.shape == data.shape
    assert valid_sec <= len(data) / sr
    plt.figure(figsize=(12, 4))
    plt.axis([0, valid_sec, -1, 1])
    plt.xlabel("time [sec]")
    plt.ylabel("amplitude")
    plt.plot(ts, data)
    plt.tight_layout()
    plt.savefig(title, dpi=320)
    plt.show()

def plot_spectrogram(fs: np.ndarray, amp_spec: np.ndarray, valid_freq: float, title: str) -> None:
    assert len(amp_spec.shape) == 1 and fs.shape == amp_spec.shape
    assert valid_freq <= fs[-1]
    plt.figure(figsize=(12, 4))
    plt.axis([0, valid_freq, 0, 200])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude")
    plt.plot(fs[:len(fs)//2], abs_X[:len(abs_X)//2])
    plt.tight_layout()
    plt.savefig(title, dpi=320)
    plt.show()


if __name__ == "__main__":
    data, sr = read_wave_16bit("p0501_input.wav", is_mono=True)
    ts: np.ndarray = np.arange(0, len(data)) / sr
    plot_mono_wave(ts, data, sr, valid_sec=0.008, title="p0501_time.png")

    N: int = 1024
    x: np.ndarray = hanning_window(N) * data[:N]
    X: np.ndarray = np.fft.fft(x, N)
    abs_X: np.ndarray = np.abs(X)

    fs: np.ndarray = sr * np.arange(N) / N
    plot_spectrogram(fs, abs_X, valid_freq=4000, title="p0501_freq.png")