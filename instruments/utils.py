import numpy as np
import inspect
from typing import Callable, Any

A4_FREQ: float = 440.0
A4_NOTE: int = 69

def calc_freq(note_no: int) -> float:
    return A4_FREQ * 2 ** ((note_no - A4_NOTE) / 12)

def noise(n_sample: int, seed: int =0) -> np.ndarray:
    return 2 * (np.random.default_rng(seed).random(size=n_sample) - 0.5)

def get_func_kwargs(fn: Callable[[Any], Any]) -> list[str]:
    return inspect.signature(fn).parameters.keys()

def calc_delayar(offsets: np.ndarray|float, p1: float, p2: float, p3: float, p4: float, is_delay: bool =False) -> np.ndarray:
    delayar: np.ndarray = 1 / (1 + np.exp(-(offsets - p1) / p3)) * p4 + p2
    if is_delay and not isinstance(offsets, float):
        delayar = delayar - delayar[0]
    return delayar