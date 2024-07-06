import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from wavio import read_wave_16bit
from window import hanning_window

eps: float = 1e-5


def plot_spectrogram(spec: np.ndarray, n_frames: int, shift_size: int, N: int, sr: int):
    plt.figure(figsize=(8, 5))
    plt.xlabel("time [sec]")
    plt.ylabel("frequency [Hz]")
    plt.imshow(spec, aspect="auto", cmap="Grays", origin="lower", vmin=0, vmax=20,
               extent=[N / 2 / sr, ((n_frames - 1) * shift_size + N/2) / sr,
                       0, sr / 2]
              )
    plt.tight_layout()
    plt.savefig("p0502_spec.png")
    plt.show()


if __name__ == "__main__":
    data, sr = read_wave_16bit("p0502_input.wav", is_mono=True)

    N: int = 512
    shift_size: int = 64
    n_frames: int = (len(data) - (N - shift_size)) // shift_size

    spec: np.ndarray = np.zeros((N//2+1, n_frames))
    for frame in range(n_frames):
        offset: int = frame * shift_size
        x: np.ndarray = hanning_window(N) * data[offset:offset+N]
        X: np.ndarray = np.fft.fft(x, N)
        abs_X: np.ndarray = np.abs(X)
        spec[:, frame] = 20 * np.log10(abs_X+eps)[:N//2+1]

    plot_spectrogram(spec, n_frames, shift_size, N, sr)