import sys
sys.path.append("..")
from instruments.percussions import snare
from instruments.utils import render_pad_save

render_pad_save(snare, save_title="p0912_snare.wav",
                velocity=100, gate=0.1, duration=1.2)