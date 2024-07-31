import sys
sys.path.append("..")
from wavio import read_wave_16bit, write_wave_16bit
from effect import distortion

if __name__ == "__main__":
    y, sr = read_wave_16bit("p0702_input.wav", is_mono=True)
    gain: float = 1000
    level: float = 0.2
    y = distortion(y, gain=gain, level=level)
    write_wave_16bit(y, sr, "p0702_output.wav", is_mono=True)