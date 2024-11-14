import sys
sys.path.append("..")
import numpy as np
from instruments.tubular_bells import tubular_bells
from wavio import write_wave_16bit


def main(note_no: int, velocity: int, gate: float =0.1, duration: float =5.0,
         blanks: list[float] =[1.0, 2.0], sr: int =44100) -> None:
    x: np.ndarray = tubular_bells(note_no, velocity, gate, duration, sr)
    w: np.ndarray = np.zeros(int(blanks[0] * sr))
    z: np.ndarray = np.zeros(int(blanks[1] * sr))
    y: np.ndarray = np.concatenate([w, x, z])
    write_wave_16bit(y, sr=sr, filename="p0903_tubular_bells.wav", is_mono=True)


if __name__ == "__main__":
    main(note_no=60, velocity=100)