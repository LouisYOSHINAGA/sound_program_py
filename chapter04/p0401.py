import sys
sys.path.append("..")
import numpy as np
import midi as m
from wavio import write_wave_16bit


A4FREQ: float = 440.0
A4NOTE: int = 69
FADEIN_SEC: float = 0.01
TAIL_BLANK: float = 2.0


def sine(noteno: int, velocity: int, sec: float, sr: int = 44100) -> np.ndarray:
    ts: np.ndarray = np.arange(0, sec, 1/sr)
    freq: float = A4FREQ * 2 ** ((noteno - A4NOTE) / 12)
    sine: np.ndarray = np.sin(2 * np.pi * freq * ts)

    fadein_sample: int = int(FADEIN_SEC * sr)
    sine[:fadein_sample] *= np.arange(0, 1, 1/fadein_sample)
    sine[-fadein_sample:] *= np.arange(1, 0, -1/fadein_sample)

    return (velocity / 127) * sine

def main(sr: int =44100) -> None:
    div, tempo, n_tracks, eot, score = m.decode("canon.mid", is_verbose=False)
    bpm: float = 60 / (tempo / 1e6)
    dur: float = (eot / div) * (60 / bpm)

    track: np.ndarray = np.zeros((int((dur+TAIL_BLANK)*sr), n_tracks))
    for note in range(len(score)):
        onset: float = (score[note, m.CURRENT_TIME_IN_SCORE] / div) * (60 / bpm)
        offset: int = int(onset * sr)

        noteno: int = score[note, m.NOTEON_IN_SCORE]
        velocity: int = score[note, m.VELOCITY_IN_SCORE]
        sec: float = (score[note, m.GATE_IN_SCORE] / div) * (60 / bpm)

        track[offset:offset+int(sec*sr), score[note, m.TRACK_IN_SCORE]] += sine(noteno, velocity, sec, sr)
    mvol: float = 0.5
    y: np.ndarray = np.sum(track, axis=1)
    y *= mvol / np.max(np.abs(y))
    write_wave_16bit(y, sr=sr, filename=f"p0401_output.wav", is_mono=True)


if __name__ == "__main__":
    main()