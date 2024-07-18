import sys
sys.path.append("..")
import numpy as np
from env import adsr
import matplotlib.pyplot as plt


def plot_envepole(env: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.axis([0, 2, 0, 1])
    plt.xlabel("time [sec]")
    plt.ylabel("amplitude")
    plt.plot(np.arange(int(sec*sr))/sr, env)
    plt.tight_layout()
    plt.savefig(title, dpi=320)
    plt.show()


if __name__ == "__main__":
    sr: int = 44100
    offset: float = 0.0
    depth: float = 1.0
    sec_A: float = 0.1
    sec_D: float = 0.4
    amp_S: float = 0.5
    sec_R: float = 0.4
    sec_gate: float = 1.0
    sec: float = 2.0

    env: np.ndarray = adsr(sec_A, sec_D, sec_gate, amp_S, sec_R, sec, sr)
    env[:int(sec*sr)] = offset + depth * env[:int(sec*sr)]

    plot_envepole(env, title="p0605_adsr.png")