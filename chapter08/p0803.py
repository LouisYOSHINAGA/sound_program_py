import sys
sys.path.append("..")
import numpy as np
import midi as m
from wavio import write_wave_16bit


A4FREQ: float = 440.0
A4NOTE: int = 69
FADEIN_SEC: float = 0.01
TAIL_BLANK: float = 2.0


def sine(noteno: int, velocity: int, sec: float, sr: int =44100) -> np.ndarray:
    ts: np.ndarray = np.arange(0, sec, 1/sr)
    freq: float = A4FREQ * 2 ** ((noteno - A4NOTE) / 12)
    sine: np.ndarray = np.sin(2 * np.pi * freq * ts)

    fadein_sample: int = int(FADEIN_SEC * sr)
    sine[:fadein_sample] *= np.arange(0, 1, 1/fadein_sample)
    sine[-fadein_sample:] *= np.arange(1, 0, -1/fadein_sample)

    return (velocity / 127) * sine

def load(midi: str, target_tracks: list[int], sr: int =44100) -> np.ndarray:
    div, tempo, n_tracks, eot, score = m.decode(midi, is_verbose=False)
    bpm: float = 60 / (tempo / 1e6)
    dur: float = (eot / div) * (60 / bpm)
    assert len(target_tracks) <= n_tracks

    track: np.ndarray = np.zeros((int((dur+TAIL_BLANK)*sr), len(target_tracks)))
    for i, target_track in enumerate(target_tracks):
        for note in range(len(score)):
            if score[note, m.TRACK_IN_SCORE] == target_track:
                onset: float = (score[note, m.CURRENT_TIME_IN_SCORE] / div) * (60 / bpm)
                offset: int = int(onset * sr)

                noteno: int = score[note, m.NOTEON_IN_SCORE]
                velocity: int = score[note, m.VELOCITY_IN_SCORE]
                sec: float = (score[note, m.GATE_IN_SCORE] / div) * (60 / bpm)

                track[offset:offset+int(sec*sr), i] += sine(noteno, velocity, sec, sr)
    return track

def mixdown(track: np.ndarray, pans: list[float], mvol: float, title: str, sr: int =44100) -> None:
    mtrack: np.ndarray = np.zeros((track.shape[0], 2))
    mtrack[:, 0] = np.sum(np.cos(np.array(pans)*np.pi/2) * track, axis=1)
    mtrack[:, 1] = np.sum(np.sin(np.array(pans)*np.pi/2) * track, axis=1)
    mtrack = mvol / np.max(np.abs(mtrack), axis=0) * mtrack
    write_wave_16bit(mtrack, sr, title)


if __name__ == "__main__":
    pan: float = 0.7
    track: np.ndarray = load("canon.mid", target_tracks=[1, 2])
    mixdown(track, pans=[pan, 1-pan], mvol=0.5, title="p0803_output.wav")