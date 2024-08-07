import numpy as np
from typing import Any, Callable

"""
    omega := 2 pi fc / sr
    alpha := sin(omega) / (2 Q)

    fc' := tan(pi fc / sr) / (2 pi)
        = tan(omega/2) / (2 pi)

    den := 1 + 2 pi fc' / Q + (2 pi fc')^2
        = 1 + tan(omega/2) / Q + tan^2(omega/2)
           cos^2(omega/2) + sin(omega/2) cos(omega/2) / Q + sin^2(omega/2)
        = -----------------------------------------------------------------
                                cos^2(omega/2)
           1 + sin(omega) / (2 Q)
        = ------------------------
            (1 + cos(omega)) / 2
           2 (1 + alpha)
        = ----------------
           1 + cos(omega)

    num1 := 2 pi fc' / Q
            tan(omega/2)
         = --------------
                 Q
             sin(omega/2)
         = ----------------
            Q cos(omega/2)
            sin(omega/2) cos(omega/2)
         = ---------------------------
                Q cos^2(omega/2)
                sin(omega) / 2
         = ------------------------
            Q (1 + cos(omega)) / 2
            sin(omega)           2
         = ------------ x ----------------
               2 Q         1 + cos(omega)
               2 alpha
         = ----------------
            1 + cos(omega)

    num2 := (2 pi fc')^2
         = tan^2(omega/2)
         = 1 / cos^2(omega/2) - 1
                  2
         = ---------------- - 1
            1 + cos(omega)
            1 - cos(omega)
         = ----------------
            1 + cos(omega)
"""

def lpf_coef (fc: float|np.ndarray, Q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        a[0] := 1 + 2 pi fc' / Q + (2 pi fc')^2
             = den
        a[0]' = a[0] / a[0]
              = 1

        a[1] := 2 (2 pi fc)^2 - 2
             = 2 num2 - 2
                2 (1 - cos(omega))     2 (1 + cos(omega))
             = -------------------- - --------------------
                  1 + cos(omega)         1 + cos(omega)
                - 4 cos(omega)
             = ----------------
                1 + cos(omega)
        a[1]' = a[1] / a[0]
                 - 4 cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
              = -2 cos(omega) / (1 + alpha)

        a[2] := 1 - 2 pi fc' / Q + (2 pi fc')^2
             = 1 - num1 + num2
                1 + cos(omega)        2 alpha         1 - cos(omega)
             = ---------------- - ---------------- + ----------------
                1 + cos(omega)     1 + cos(omega)     1 + cos(omega)
                 2 - 2 alpha
             = ----------------
                1 + cos(omega)
        a[2]' = a[2] / a[0]
                  2 - 2 alpha       1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 1 - alpha
              = -----------
                 1 + alpha

        b[0] := (2 pi fc')^2
             = num2
        b[0]' = b[0] / a[0]
                 1 - cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 1 - cos(omega)
              = ----------------
                 2 (1 + alpha)

        b[1] := 2 (2 pi fc')^2
             = 2 num2
             = 2 b[0]
        b[1]' = b[1] / a[0]
              = 2 b[0] / a[0]
              = 2 b[0]'
                 1 - cos(omega)
              = ----------------
                   1 + alpha

        b[2] := (2 pi fc')^2
             = num2
             = b[0]
        b[2]' = b[2] / a[0]
              = b[0] / a[0]
              = b[0]'
                 1 - cos(omega)
              = ----------------
                 2 (1 + alpha)
    """
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = (2 * np.pi * fc)**2
    b[1] = 2 * (2 * np.pi * fc)**2
    b[2] = (2 * np.pi * fc)**2
    b /= a[0]
    a /= a[0]
    return a, b

def hpf_coef(fc: float|np.ndarray, Q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        a[0] := 1 + 2 pi fc' / Q + (2 pi fc')^2
             = den
        a[0]' = a[0] / a[0]
              = 1

        a[1] := 2 (2 pi fc')^2 - 2
             = 2 num2 - 2
                2 (1 - cos(omega))     2 (1 + cos(omega))
             = -------------------- - --------------------
                  1 + cos(omega)         1 + cos(omega)
                - 4 cos(omega)
             = ----------------
                1 + cos(omega)
        a[1]' = a[1] / a[0]
                 - 4 cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
              = -2 cos(omega) / (1 + alpha)

        a[2] := 1 - 2 pi fc' / Q + (2 pi fc')^2
             = 1 - num1 + num2
                1 + cos(omega)        2 alpha         1 - cos(omega)
             = ---------------- - ---------------- + ----------------
                1 + cos(omega)     1 + cos(omega)     1 + cos(omega)
                 2 - 2 alpha
             = ----------------
                1 + cos(omega)
        a[2]' = a[2] / a[0]
                  2 - 2 alpha       1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 1 - alpha
              = -----------
                 1 + alpha

        b[0] := 1
        b[0]' = b[0] / a[0]
                 1 + cos(omega)
              = ----------------
                 2 (1 + alpha)

        b[1] := -2
        b[1]' = b[1] / a[0]
              = -2 b[0] / a[0]
              = -2 b[0]'
                   1 + cos(omega)
              = - ----------------
                     1 + alpha

        b[2] := 1
        b[2]' = b[2] / a[0]
              = b[0] / a[0]
              = b[0]'
                 1 + cos(omega)
              = ----------------
                 2 (1 + alpha)
    """
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = 1
    b[1] = -2
    b[2] = 1
    b /= a[0]
    a /= a[0]
    return a, b

def bpf_coef(fc: float|np.ndarray, Q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        a[0] := 1 + 2 pi fc' / Q + (2 pi fc')^2
             = den
        a[0]' = a[0] / a[0]
              = 1

        a[1] := 2 (2 pi fc')^2 - 2
             = 2 num2 - 2
                2 (1 - cos(omega))     2 (1 + cos(omega))
             = -------------------- - --------------------
                  1 + cos(omega)         1 + cos(omega)
                - 4 cos(omega)
             = ----------------
                1 + cos(omega)
        a[1]' = a[1] / a[0]
                 - 4 cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
              = -2 cos(omega) / (1 + alpha)

        a[2] := 1 - 2 pi fc' / Q + (2 pi fc')^2
             = 1 - num1 + num2
                1 + cos(omega)        2 alpha         1 - cos(omega)
             = ---------------- - ---------------- + ----------------
                1 + cos(omega)     1 + cos(omega)     1 + cos(omega)
                 2 - 2 alpha
             = ----------------
                1 + cos(omega)
        a[2]' = a[2] / a[0]
                  2 - 2 alpha       1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 1 - alpha
              = -----------
                 1 + alpha

        b[0] := 2 pi fc' / Q
             = num1
        b[0]' = b[0] / a[0]
                    2 alpha         1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                   alpha
              = -----------
                 1 + alpha

        b[1] := 0
        b[1]' = b[1] / a[0]
              = 0

        b[2] := - 2 pi fc' / Q
             = - num1
             = - b[0]
        b[2]' = b[2] / a[0]
              = b[0] / a[0]
              = - b[0]'
                     alpha
              = - -----------
                   1 + alpha
   """
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = 2 * np.pi * fc / Q
    b[1] = 0
    b[2] = - 2 * np.pi * fc / Q
    b /= a[0]
    a /= a[0]
    return a, b

def bef_coef(fc: float|np.ndarray, Q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        a[0] := 1 + 2 pi fc' / Q + (2 pi fc')^2
             = den
        a[0]' = a[0] / a[0]
              = 1

        a[1] := 2 (2 pi fc')^2 - 2
             = 2 num2 - 2
                2 (1 - cos(omega))     2 (1 + cos(omega))
             = -------------------- - --------------------
                  1 + cos(omega)         1 + cos(omega)
                - 4 cos(omega)
             = ----------------
                1 + cos(omega)
        a[1]' = a[1] / a[0]
                 - 4 cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
              = -2 cos(omega) / (1 + alpha)

        a[2] := 1 - 2 pi fc' / Q + (2 pi fc')^2
             = 1 - num1 + num2
                1 + cos(omega)        2 alpha         1 - cos(omega)
             = ---------------- - ---------------- + ----------------
                1 + cos(omega)     1 + cos(omega)     1 + cos(omega)
                 2 - 2 alpha
             = ----------------
                1 + cos(omega)
        a[2]' = a[2] / a[0]
                  2 - 2 alpha       1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 1 - alpha
              = -----------
                 1 + alpha

        b[0] := (2 pi fc')^2 + 1
             = num2 + 1
                1 - cos(omega)     1 + cos(omega)
             = ---------------- + ----------------
                1 + cos(omega)     1 + cos(omega)
                     2
            = ----------------
               1 + cos(omega)
        b[0]' = b[0] / a[0]
                       2            1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                     1
              = -----------
                 1 + alpha

        b[1] := 2 (2 pi fc')^2 - 2
             = 2 num2 - 2
                2 (1 - cos(omega))     2 (1 + cos(omega))
             = -------------------- - --------------------
                  1 + cos(omega)         1 + cos(omega)
                - 4 cos(omega)
             = ----------------
                1 + cos(omega)
        b[1]' = b[1] / a[0]
                 - 4 cos(omega)     1 + cos(omega)
              = ---------------- x ----------------
                 1 + cos(omega)     2 (1 + alpha)
                 - 2 cos(omega)
              = ----------------
                   1 + alpha

        b[2] := (2 pi fc')^2 + 1
             = num2 + 1
             = b[0]
        b[2]' = b[2] / a[0]
              = b[0] / a[0]
              = b[0]'
                     1
              = -----------
                 1 + alpha
    """
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = (2 * np.pi * fc)**2 + 1
    b[1] = 2 * (2 * np.pi * fc)**2 - 2
    b[2] = (2 * np.pi * fc)**2 + 1
    b /= a[0]
    a /= a[0]
    return a, b

def lsf_coef(fc: float|np.ndarray, Q: float, g: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = 1 + 2 * np.pi * fc / Q * np.sqrt(1 + g) + (2 * np.pi * fc)**2 * (1 + g)
    b[1] = 2 * (2 * np.pi * fc)**2 * (1 + g) - 2
    b[2] = 1 - 2 * np.pi * fc / Q * np.sqrt(1 + g) + (2 * np.pi * fc)**2 * (1 + g)
    b /= a[0]
    a /= a[0]
    return a, b

def hsf_coef(fc: float|np.ndarray, Q: float, g: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = 1 + g + 2 * np.pi * fc / Q * np.sqrt(1 + g) + (2 * np.pi * fc)**2
    b[1] = 2 * (2 * np.pi * fc)**2 - 2 * (1 + g)
    b[2] = 1 + g - 2 * np.pi * fc / Q * np.sqrt(1 + g) + (2 * np.pi * fc)**2
    b /= a[0]
    a /= a[0]
    return a, b

def pf_coef(fc: float|np.ndarray, Q: float, g: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(fc, (int, float)) or len(fc.shape) == 1
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    a[0] = 1 + 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    a[1] = 2 * (2 * np.pi * fc)**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + (2 * np.pi * fc)**2
    b: np.ndarray = np.zeros(3) if isinstance(fc, (int, float)) else np.zeros((3, len(fc)))
    b[0] = 1 + 2 * np.pi * fc / Q * (1 + g) + (2 * np.pi * fc)**2
    b[1] = 2 * (2 * np.pi * fc)**2 - 2
    b[2] = 1 - 2 * np.pi * fc / Q * (1 + g) + (2 * np.pi * fc)**2
    b /= a[0]
    a /= a[0]
    return a, b


def apply_biquad_filter(data: np.ndarray, coefs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a, b = coefs
    if len(a.shape) == 1:
      a = np.repeat(a[:, np.newaxis], len(data), axis=1)
    assert len(data.shape) == 1 and a.shape == (3, len(data)) and b.shape == (3, ) or b.shape == (3, len(data))

    z1: np.ndarray = np.concatenate([np.zeros(1), data[:-1]])
    z2: np.ndarray = np.concatenate([np.zeros(2), data[:-2]])
    assert data.shape == z1.shape and z1.shape == z2.shape

    mid_data: np.ndarray = b[0] * data + b[1] * z1 + b[2] * z2
    assert mid_data.shape == data.shape

    lpf_data: np.ndarray = np.concatenate([
        np.array([mid_data[0], mid_data[1] - a[1, 1] * mid_data[0]]),
        np.zeros(len(mid_data) - 2)
    ])
    for i in range(2, len(lpf_data)):
        lpf_data[i] = mid_data[i] - a[1, i] * lpf_data[i-1] - a[2, i] * lpf_data[i-2]
    return lpf_data

def biquad_filter(data: np.ndarray, filter_type: str, **filter_kwargs: Any) -> np.ndarray:
    filter_coef_funcs: dict[str, Callable[[Any], tuple[np.ndarray, np.ndarray]]] = {
        'lowpass': lpf_coef,
        'highpass': hpf_coef,
        'bandpass': bpf_coef,
        'bandeliminate': bef_coef,
        'lowshelf': lsf_coef,
        'highshelf': hsf_coef,
        'peaking': pf_coef,
    }
    assert filter_type in filter_coef_funcs
    return apply_biquad_filter(data, filter_coef_funcs[filter_type](**filter_kwargs))