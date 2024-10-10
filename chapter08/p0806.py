import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from window import hanning_window


def amp_diff(xs: np.ndarray) -> np.ndarray:
    assert len(xs.shape) == 2 and xs.shape[1] == 2
    diffs: np.ndarray = np.zeros(xs.shape)
    diffs[:, 0] = xs[:, 0] - xs[:, 1]
    diffs[:, 1] = xs[:, 1] - xs[:, 0]
    return diffs

def freq_diff(xs: np.ndarray, N: int =4096, min_freq: float =200, max_freq: float =8000, sr: int =44100) -> np.ndarray:
    assert len(xs.shape) == 2 and xs.shape[1] == 2
    diffs: np.ndarray = np.zeros(xs.shape)

    hop_length: int = N // 2
    n_frames: int = (len(xs) - (N - hop_length)) // hop_length

    min_bin: float = round(min_freq * N / sr)
    max_bin: float = round(max_freq * N / sr)

    for frame in range(n_frames):
        offset: int = hop_length * frame
        wxs: np.ndarray = hanning_window(N)[:, np.newaxis] * xs[offset:offset+N]
        Xs: np.ndarray = np.fft.fft(wxs, N, axis=0)
        absXs: np.ndarray = np.abs(Xs)

        for bin in range(min_bin, max_bin):
            if is_same_freq(Xs[bin]):
                absXs[bin] = absXs[N-bin] = 1e-6

        Ys: np.ndarray = absXs * np.exp(1j * np.angle(Xs))
        diffs[offset:offset+N] += np.real(np.fft.ifft(Ys, N, axis=0))
    return diffs

def is_same_freq(X: np.ndarray, threshold: float =0.001, eps: float =1e-8) -> bool:
    num: float = np.abs(X[0] - X[1]) ** 2
    den: float = np.abs(X[0] + X[1]) ** 2 + eps
    return num / den < threshold


def main() -> None:
    xs, sr = read_wave_16bit("p0806_input.wav")

    c: float = 0.2
    das: np.ndarray = amp_diff(xs)
    dfs: np.ndarray = freq_diff(xs, sr=sr)
    ys: np.ndarray = c * das + (1 - c) * dfs

    write_wave_16bit(ys, sr=sr, filename="p0806_output.wav")

if __name__ == "__main__":
    main()