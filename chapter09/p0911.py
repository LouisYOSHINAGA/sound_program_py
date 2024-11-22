import sys
sys.path.append("..")
import numpy as np
from instruments.percussions import tom
from wavio import write_wave_16bit


def main(velocity: int, gate: float =0.1, duration: float =1.4,
         blanks: list[float] =[1.0, 2.0], sr: int =44100) -> None:
    x: np.ndarray = tom(velocity, gate, duration, sr)
    w: np.ndarray = np.zeros(int(blanks[0] * sr))
    z: np.ndarray = np.zeros(int(blanks[1] * sr))
    y: np.ndarray = np.concatenate([w, x, z])
    write_wave_16bit(y, sr=sr, filename="p0911_tom.wav", is_mono=True)


if __name__ == "__main__":
    main(velocity=100)