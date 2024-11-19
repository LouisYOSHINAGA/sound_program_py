import sys
sys.path.append("..")
import numpy as np
from instruments.percussions import triangle_in, triangle_out
from wavio import write_wave_16bit


def main_in(velocity: int, gate: float =0.1, duration: float =9.0,
            blanks: list[float] =[1.0, 2.0], sr: int =44100) -> None:
    x: np.ndarray = triangle_in(velocity, gate, duration, sr)
    w: np.ndarray = np.zeros(int(blanks[0] * sr))
    z: np.ndarray = np.zeros(int(blanks[1] * sr))
    y: np.ndarray = np.concatenate([w, x, z])
    write_wave_16bit(y, sr=sr, filename="p0902_triangle_in.wav", is_mono=True)

def main_out(velocity: int, gate: float =0.1, duration: float =9.0,
             blanks: list[float] =[1.0, 2.0], sr: int =44100) -> None:
    x: np.ndarray = triangle_out(velocity, gate, duration, sr)
    w: np.ndarray = np.zeros(int(blanks[0] * sr))
    z: np.ndarray = np.zeros(int(blanks[1] * sr))
    y: np.ndarray = np.concatenate([w, x, z])
    write_wave_16bit(y, sr=sr, filename="p0902_triangle_out.wav", is_mono=True)


if __name__ == "__main__":
    main_in(velocity=100)
    main_out(velocity=100)