import numpy as np
from wavio import write_wave_16bit
import midi as m


A4FREQ: float = 440.0
A4NOTE: int = 69


def sine(noteno: int, velocity: int, sec: float, sr: int = 44100) -> np.ndarray:
    freq: np.ndarray = np.full(int(sec * sr), 
                               A4FREQ * np.power(2, (noteno - A4NOTE)/12))
    sine: np.ndarray = np.sin(2 * np.pi * freq / sr)
    gain: float = np.max(np.abs(sine)) * (velocity / 127)
    return gain * sine


if __name__ == "__main__":
    div, tempo, n_tracks, eot, score = m.decode("canon.mid")
    tempo = 60 / (tempo / 1e6)
    n_tracks = int(n_tracks - 1)
    eot = (eot / div) * (60 / tempo)
    n_notes: int = score.shape[0]

    sr: int = 44100
    track: np.ndarray = np.zeros((int((eot+2)*sr), n_tracks))
    target_track: int = 0

    for note in range(n_notes):
        if int(score[note, 0] - 1) == target_track:
            onset: int = (score[note, 1] / div) * (60 / tempo)
            offset: int = int(onset * sr)

            noteno: int = score[note, 2]
            velocity: int = score[note, 3]
            sec: int = (score[note, 4] / div) * (60 / tempo)

            track[offset:offset+int(sec*sr), target_track] = sine(noteno, velocity, sec, sr)

    mvol: float = 0.5
    y: np.ndarray = mvol * np.max(np.abs(track[:, target_track]))

    write_wave_16bit(y, sr=sr, filename="p0801_output.wav", is_mono=False)
