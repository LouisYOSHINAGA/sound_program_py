import numpy as np

def lpf_coef (fc: float, Q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        omega := 2 pi fc / sr
        alpha := sin(omega) / (2 Q)

        fc' := tan(pi fc / sr) / (2 pi)
            = tan(omega/2) / (2 pi)

        a[0] := 1 + 2 pi fc' / Q + 4 pi^2 fc^2
             = 1 + tan(omega/2) / Q + tan^2(omega/2)
        a[0]' = a[0] / a[0]
              = 1

        a[1] := 8 pi^2 fc'^2 - 2
        a[1]' = a[1] / a[0]
              = 2 (tan^2(omega/2) - 1) / a[0]
                        2 (tan^2(omega/2) - 1)
              = ---------------------------------------
                 1 + tan(omega/2) / Q + tan^2(omega/2)
                                2 ( sin^2(omega/2) - cos^2(omega/2) )
              = -----------------------------------------------------------------
                 cos^2(omega/2) + sin(omega/2) cos(omega/2) / Q + cos^2(omega/2)
                     sin^2(omega/2) - cos^2(omega/2)
              = 2 ------------------------------------
                   1 + sin(omega/2) cos(omega/2) / Q
                        - cos(omega)
              = 2 ------------------------
                   1 + sin(omega) / (2 Q)
              = -2 cos(omega) / (1 + alpha)

        a[2] := 1 - 2 pi fc' / Q + 4 pi^2 fc'^2
        a[2]' = a[2] / a[0]
                 1 - tan(omega/2) / Q + tan^2(omega/2)
              = ---------------------------------------
                 1 + tan(omega/2) / Q + tan^2(omega/2)
                 cos^2(omega/2) - sin(omega/2) cos(omega/2) / Q + sin^2(omega/2)
              = -----------------------------------------------------------------
                 cos^2(omega/2) + sin(omega/2) cos(omega/2) / Q + sin^2(omega/2)
                 1 - sin(omega/2) cos(omega/2) / Q
              = -----------------------------------
                 1 + sin(omega/2) cos(omega/2) / Q
                 1 - sin(omega) / (2 Q)
              = ------------------------
                 1 + sin(omega) / (2 Q)
              = (1 - alpha) / (1 + alpha)

        b[0] := 4 pi^2 fc'^2
        b[0]' = b[0] / a[0]
                           tan^2(omega/2)
              = ---------------------------------------
                 1 + tan(omega/2) / Q + tan^2(omega/2)
                                        sin^2(omega/2)
              = -----------------------------------------------------------------
                 cos^2(omega/2) + sin(omega/2) cos(omega/2) / Q + sin^2(omega/2)
                 (-cos(omega) + 1) / 2
              = ------------------------
                 1 + sin(omega) / (2 Q)
              = (1 - cos(omega)) / (2 * (1 + alpha))

        b[1] := 8 pi^2 fc'^2
        b[1]' = b[1] / a[0]
              = 2 b[0] / a[0]
              = (1 - cos(omega)) / (1 + alpha)

        b[2] := 4 pi^2 fc'^2
        b[2]' = b[2] / a[0]
              = b[0] / a[0]
              = (1 - cos(omega)) / (2 * (1 + alpha))
    """
    fc = np.tan(np.pi * fc / sr) / (2 * np.pi)
    a: np.ndarray = np.zeros(3)
    a[0] = 1 + 2 * np.pi * fc / Q + 4 * np.pi**2 * fc**2
    a[1] = 8 * np.pi**2 * fc**2 - 2
    a[2] = 1 - 2 * np.pi * fc / Q + 4 * np.pi**2 * fc**2
    b: np.ndarray = np.zeros(3)
    b[0] = 4 * np.pi**2 * fc**2
    b[1] = 8 * np.pi**2 * fc**2
    b[2] = 4 * np.pi**2 * fc**2
    b /= a[0]
    a /= a[0]
    return a, b