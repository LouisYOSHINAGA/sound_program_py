import sys
sys.path.append("..")
import numpy as np
from wavio import read_wave_16bit, write_wave_16bit
from window import hanning_window


bounds: np.ndarray = np.array([
       0,    4,    7,   12,   16,   20,   25,   30,   35,   41,
      47,   53,   60,   67,   74,   82,   90,   99,  108,  118,
     128,  139,  150,  162,  175,  188,  202,  217,  233,  250,
     267,  286,  306,  326,  348,  371,  396,  421,  449,  477,
     508,  540,  574,  609,  647,  687,  729,  773,  820,  869,
     922,  977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545,
    1635, 1730, 1830, 1936, 2048
])
amps: np.ndarray = np.array([
     2.113102,  7.881102, 11.075727,  8.863704,  4.013235,  0.932089,  1.197000,  1.457407,  1.107974,  0.403404,
     0.357882,  0.831457,  0.998504,  0.977239,  0.803210,  0.446178,  0.149860,  0.585399,  0.968549,  1.216626,
     1.355936,  1.389370,  1.275824,  0.925896,  0.300999,  0.590605,  1.593167,  2.624420,  3.691557,  4.830407,
     6.157985,  7.791523,  9.669384, 11.712244, 13.612665, 14.868957, 15.269036, 15.070204, 14.665470, 14.323831,
    14.172306, 14.265161, 14.594980, 15.102259, 15.596862, 15.656709, 14.803574, 12.975877, 10.714214,  8.580678,
     6.829857,  5.485252,  4.450106,  3.657807,  3.042211,  2.555303,  2.168048,  1.856382,  1.607098,  1.407719,
     1.250445,  1.131747,  1.050209,  1.007424
])


if __name__ == "__main__":
    x, sr = read_wave_16bit("p0705_input.wav", is_mono=True)

    H: np.ndarray = np.zeros(bounds[-1])
    for i in range(len(amps)):
        H[bounds[i]:bounds[i+1]] = amps[i]
    H = np.concatenate([H, H[::-1]])

    h: np.ndarray = np.real(np.fft.ifft(H, 2*bounds[-1]))
    h = np.concatenate([h[len(h)//2:], h[:len(h)//2]])

    d: int = 128
    offset: int = (len(h) - d) // 2
    b: np.ndarray = hanning_window(d+1) * h[offset:offset+d+1]

    volume: float = 1.0
    y: np.ndarray = np.convolve(b, x, mode="same")
    y = volume * y / np.max(np.abs(y))

    write_wave_16bit(y, sr, "p0705_output.wav", is_mono=True)